#!/usr/bin/env python3
# Times grobx and msolve on a set of .ms instances, one thread each.
# usage: run.py OUT.json SET_DIR [label=cmd-template ...] [--jobs N] [--timeout S]
# cmd-template uses {in} and {out}; e.g. msolve="/path/msolve -g 2 -t 1 -f {in} -o {out}"
import hashlib, json, os, re, signal, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

args = sys.argv[1:]
jobs, timeout, keep, repeat = 4, 120, False, 1
if "--repeat" in args:
    i = args.index("--repeat"); repeat = int(args[i + 1]); del args[i:i + 2]
if "--keep" in args:
    args.remove("--keep"); keep = True
if "--jobs" in args:
    i = args.index("--jobs"); jobs = int(args[i + 1]); del args[i:i + 2]
if "--timeout" in args:
    i = args.index("--timeout"); timeout = float(args[i + 1]); del args[i:i + 2]
out_json, set_dir, progs = args[0], args[1], dict(a.split("=", 1) for a in args[2:])
files = sorted(os.path.join(r, f) for r, _, fs in os.walk(set_dir) for f in fs if f.endswith(".ms"))
tmp = tempfile.mkdtemp(prefix="bench-", dir=os.path.dirname(os.path.abspath(out_json)))

def one(task):
    best = None
    for _ in range(repeat):
        r = one_run(task)
        if best is None or (r["status"] == "ok" and (best["status"] != "ok" or r["seconds"] < best["seconds"])):
            best = r
        if r["status"] != "ok":
            break
    return best

def one_run(task):
    label, path = task
    out = os.path.join(tmp, f"{label}-{os.path.basename(path)}.out")
    cmd = ["/usr/bin/time", "-f", "MAXRSS %M"] + progs[label].format(**{"in": path, "out": out}).split()
    t = time.monotonic()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        try:
            _, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.communicate(); raise
        p.stderr = err
        dt, status = time.monotonic() - t, ("ok" if p.returncode == 0 else f"exit{p.returncode}")
        m = re.search(r"MAXRSS (\d+)", p.stderr); rss = int(m.group(1)) * 1024 if m else None
    except subprocess.TimeoutExpired:
        dt, status, rss = timeout, "timeout", None
    digest = None
    if keep and status == "ok":
        h = hashlib.sha256()
        try:
            with open(out, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        except OSError:
            pass
    try: os.remove(out)
    except OSError: pass
    return {"program": label, "instance": os.path.relpath(path, set_dir), "status": status,
            "seconds": dt, "peak_bytes": rss, "sha256": digest}

tasks = [(l, f) for f in files for l in progs]
res = []
with ThreadPoolExecutor(jobs) as ex:
    for r in ex.map(one, tasks):
        res.append(r); print(r["program"], r["instance"], r["status"], f"{r['seconds']:.2f}", flush=True)
json.dump(res, open(out_json, "w"), indent=1)

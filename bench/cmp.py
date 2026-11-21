import json, math, sys, collections
R = json.load(open(sys.argv[1])); a, b = sys.argv[2], sys.argv[3]
T = collections.defaultdict(dict)
for r in R: T[r["instance"]][r["program"]] = r
both = [i for i in T if a in T[i] and b in T[i] and T[i][a]["status"] == "ok" == T[i][b]["status"]]
for p in ("p65521", "p0", "p2"):
    s = [i for i in both if i.endswith(p + ".ms")]
    if not s: continue
    lr = [math.log(T[i][b]["seconds"] / T[i][a]["seconds"]) for i in s]
    print(f"{p}: n={len(s)} geomean {b}/{a} = {math.exp(sum(lr)/len(lr)):.3f}, {a} faster on {sum(x>0 for x in lr)}, total {a} {sum(T[i][a]['seconds'] for i in s):.1f}s {b} {sum(T[i][b]['seconds'] for i in s):.1f}s")
for prog in (a, b):
    print(prog, "not ok:", [(i, T[i][prog]["status"]) for i in T if prog in T[i] and T[i][prog]["status"] != "ok"])
if len(sys.argv) > 4:
    s = sorted(both, key=lambda i: T[i][b]["seconds"] / T[i][a]["seconds"])
    for i in s[:int(sys.argv[4])] + ["..."] + s[-int(sys.argv[4]):]:
        if i == "...": print("..."); continue
        print(f"  {i:55s} {a} {T[i][a]['seconds']:8.2f}  {b} {T[i][b]['seconds']:8.2f}  ratio {T[i][b]['seconds']/T[i][a]['seconds']:.2f}")

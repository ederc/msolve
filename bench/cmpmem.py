import json, math, sys, collections
R = json.load(open(sys.argv[1])); a, b = sys.argv[2], sys.argv[3]
T = collections.defaultdict(dict)
for r in R: T[r["instance"]][r["program"]] = r
for p in ("p65521", "p0", "p2"):
    s = [i for i in T if i.endswith(p + ".ms") and a in T[i] and b in T[i] and T[i][a]["status"] == "ok" == T[i][b]["status"] and T[i][a]["peak_bytes"] and T[i][b]["peak_bytes"]]
    if not s: continue
    lr = [math.log(T[i][b]["peak_bytes"] / T[i][a]["peak_bytes"]) for i in s]
    w = sorted(s, key=lambda i: T[i][b]["peak_bytes"] / T[i][a]["peak_bytes"])
    print(f"{p}: peak memory geomean {b}/{a} = {math.exp(sum(lr)/len(lr)):.3f}; worst {w[-1]} {T[w[-1]][b]['peak_bytes']/T[w[-1]][a]['peak_bytes']:.2f}, best {w[0]} {T[w[0]][b]['peak_bytes']/T[w[0]][a]['peak_bytes']:.2f}")

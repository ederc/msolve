import json,collections,sys
R=json.load(open(sys.argv[1])); a,b=sys.argv[2],sys.argv[3]; T=collections.defaultdict(dict)
for r in R: T[r['instance']][r['program']]=r
both=[i for i,d in T.items() if a in d and b in d and d[a]['status']=='ok'==d[b]['status']]
bad=[i for i in both if T[i][a].get('sha256')!=T[i][b].get('sha256')]
print('compared',len(both),'differ',bad)

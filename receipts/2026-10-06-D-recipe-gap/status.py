import json,sys,collections
p=sys.argv[1]
c=collections.defaultdict(collections.Counter); errs=collections.Counter()
for line in open(p,encoding='utf-8'):
    r=json.loads(line)
    st=r.get('status') or r.get('response',{}).get('status') if isinstance(r.get('response'),dict) else r.get('status')
    keys=[k for k in r.keys() if k not in('payload',)]
    c[r['catalog_id']][str(st)]+=1
    if str(st)!='ok':
        e=str(r.get('error') or r.get('detail') or '')[:90]; errs[(r['catalog_id'],e)]+=1
print(keys)
for k in sorted(c): print(k,dict(c[k]))
for k,v in errs.most_common(15): print(v,k)

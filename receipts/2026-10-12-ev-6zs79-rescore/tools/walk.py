import json,sys,hashlib,re,collections
for fn in sys.argv[1:]:
    b=open(fn,'rb').read(); print('=====',fn,len(b),hashlib.sha256(b).hexdigest())
    j=json.loads(b)
    print('top keys', list(j.keys())[:40] if isinstance(j,dict) else type(j))
    paths=collections.Counter(); hits=[]; big=[]
    def w(o,p):
        if isinstance(o,dict):
            for k,v in o.items():
                q=p+'.'+k if not re.fullmatch(r'[0-9a-f]{6,}|.*-.*-.*',k) else p+'.<k>'
                paths[q]+=1
                if re.search(r'token|len|long|item|ctx|context|char',k,re.I): hits.append((p+'.'+k, str(v)[:160]))
                w(v,q)
        elif isinstance(o,list):
            paths[p+'[]']+=1
            for x in o: w(x,p+'[]')
        elif isinstance(o,(int,float)) and not isinstance(o,bool) and 60000<=o<=100000: big.append((p,o))
    w(j,'$')
    sys_paths=sorted(paths)
    print('n distinct paths',len(sys_paths))
    for q in sys_paths:
        if q.count('.')<=4: print('  ',q,paths[q])
    print('length-ish keys (first 60 unique paths):')
    seen=set()
    for p,v in hits:
        pp=re.sub(r'\.(rows|systems)\[\]?.*?\.', '.*.', p)
        if p in seen: continue
        seen.add(p)
        if len(seen)<=60: print('  ',p,'=',v)
    print('numbers in [60k,100k]:',len(big), collections.Counter(p for p,_ in big).most_common(20))

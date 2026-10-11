import json,sys,os,hashlib
d=sys.argv[1]
for f in sorted(os.listdir(d)):
    b=open(os.path.join(d,f),'rb').read(); print(f,len(b),hashlib.sha256(b).hexdigest())
c=json.load(open(os.path.join(d,'commits_head.json'),encoding='utf-8'))
print('n commits listed',len(c)); 
pin='b6b8fff7e345b98c060ad26c13308860ddc67004'
idx=[i for i,x in enumerate(c) if x['sha']==pin]; print('pin index (0=HEAD)',idx)
for x in c[:max(idx[0]+3 if idx else 15,15)]:
    print(x['sha'][:10],x['commit']['committer']['date'],x['commit']['message'].splitlines()[0][:110])
for t in ('tree_pin.json','tree_head.json'):
    j=json.load(open(os.path.join(d,t),encoding='utf-8')); print(t,j.get('sha'),'truncated',j.get('truncated'))
    ds=[(e['path'],e.get('size')) for e in j['tree'] if e['path'].startswith('datasets') and e['type']=='blob']
    for p,s in ds: print('  ',p,s)
    sp=set(e['path'] for e in j['tree'])
    if t=='tree_pin.json': P=sp
    else:
        print('added at HEAD:',sorted(sp-P)[:80]); print('removed at HEAD:',sorted(P-sp)[:40])

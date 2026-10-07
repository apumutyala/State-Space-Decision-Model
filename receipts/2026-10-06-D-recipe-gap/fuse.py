"""Fuse Clef-flash (decide path) and stock Qwen3.5-9B readout per question: log-linear pool
p ∝ pC^w * pS^(1-w) (choice); noul: logit pool. Missing/errored side -> other side. Diagnostic only."""
import json,sys,math,copy
S,w,mode,out=sys.argv[1],float(sys.argv[2]),sys.argv[3],sys.argv[4]
def load(p):
    d={}
    for line in open(p,encoding='utf-8'):
        r=json.loads(line); d[r['run_id']]=r
    return d
st=load(S+'/5cmat/s/out/index/stock/results.jsonl'); cf=load(S+'/clef-art1/x/clef-flash-jobA/index/decide/results.jsonl')
bench=None
if mode.startswith('route:'):  # route:<comma catalog ids that use stock>
    bench=set(mode.split(':',1)[1].split(','))
def lg(p): p=min(max(p,1e-6),1-1e-6); return math.log(p/(1-p))
miss=0
with open(out,'w',encoding='utf-8') as fo:
    for rid,s in st.items():
        c=cf.get(rid)
        if c is None: miss+=1; fo.write(json.dumps(s)+'\n'); continue
        if bench is not None:
            fo.write(json.dumps(s if str(s['catalog_id']) in bench else {**c,'run_id':rid})+'\n'); continue
        if s['status']!='ok': fo.write(json.dumps(c)+'\n'); continue
        if c['status']!='ok': fo.write(json.dumps(s)+'\n'); continue
        r=copy.deepcopy(s); A=r['response']['answers']; CA=c['response']['answers']
        for qk,a in A.items():
            b=CA.get(qk)
            if b is None: continue
            if a['type']=='noul':
                z=w*lg(b['noul'])+(1-w)*lg(a['noul']); a['noul']=round(1/(1+math.exp(-z)),6)
            elif a['type']=='choice':
                ps,pc=a['probabilities'],b['probabilities']
                z={k:w*math.log(max(pc.get(k,1e-9),1e-9))+(1-w)*math.log(max(ps[k],1e-9)) for k in ps}
                m=max(z.values()); e={k:math.exp(v-m) for k,v in z.items()}; t=sum(e.values())
                pr={k:v/t for k,v in e.items()}
                # round but keep sum ~1
                a['probabilities']={k:round(v,6) for k,v in pr.items()}
                a['choice']=max(pr,key=pr.get); a['confidence']=round(pr[a['choice']],6)
        fo.write(json.dumps(r)+'\n')
print('missing clef rows',miss,file=sys.stderr)

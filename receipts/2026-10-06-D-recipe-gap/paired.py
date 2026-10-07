import json,sys,gzip,re,collections
S=sys.argv[1]; dev=sys.argv[2]
def load(p):
    d={}
    for line in open(p,encoding='utf-8'):
        r=json.loads(line); d[r['run_id']]=r
    return d
st=load(S+'/5cmat/s/out/index/stock/results.jsonl'); cv=load(S+'/5cmat/s/out/index/conv/results.jsonl')
cf=load(S+'/clef-art1/x/clef-flash-jobA/index/decide/results.jsonl')
rows={}
for line in gzip.open(dev,'rt',encoding='utf-8'):
    r=json.loads(line)
    if r['_evaluation']['run_id'] in st: rows[r['_evaluation']['run_id']]=r
def idclass(keys):
    if all(re.fullmatch(r'[A-Z]',k) for k in keys): return 'letter'
    if all(re.fullmatch(r'option_\d+',k) for k in keys): return 'option_N'
    if all(re.fullmatch(r'k_[0-9a-f]+',k) for k in keys): return 'hash'
    if set(keys)<= {'true','false'}: return 'noul'
    return 'semantic'
def ans(res):
    if res is None or res.get('status')!='ok': return {}
    return res['response'].get('answers',{}) if isinstance(res['response'],dict) else {}
B=collections.defaultdict(lambda: collections.Counter()); C=collections.defaultdict(lambda: collections.Counter())
cls_of={}
for rid,r in rows.items():
    c=r['_evaluation']['catalog_id']
    a_s,a_c,a_v=ans(st.get(rid)),ans(cf.get(rid)),ans(cv.get(rid))
    for qk,q in r['questions'].items():
        exp=r['expected'].get(qk)
        if exp is None or q.get('type')!='choice': continue
        keys=list(q['criteria'].keys()); k=idclass(keys); cls_of[c]=k
        n=len(keys)
        for name,a in (('stock',a_s),('clef',a_c),('conv',a_v)):
            x=a.get(qk); ok= x is not None and str(x.get('choice'))==str(exp)
            B[c][name]+=ok
            C[k][name]+=ok
        B[c]['n']+=1; B[c]['chance']+=1/n; C[k]['n']+=1; C[k]['chance']+=1/n
        okS=str((a_s.get(qk) or {}).get('choice'))==str(exp); okC=str((a_c.get(qk) or {}).get('choice'))==str(exp)
        B[c]['both']+=okS and okC; B[c]['clef_only']+=okC and not okS; B[c]['stock_only']+=okS and not okC
print('cat | idclass | nq | chance% | stock% | conv% | clef% | clef_only | stock_only')
for c in sorted(B,key=lambda c:int(c)):
    b=B[c]; n=b['n']
    print(f"{c} | {cls_of[c]} | {n} | {100*b['chance']/n:.0f} | {100*b['stock']/n:.0f} | {100*b['conv']/n:.0f} | {100*b['clef']/n:.0f} | {b['clef_only']} | {b['stock_only']}")
print('--- by id class (question-weighted)')
for k,b in C.items():
    n=b['n']; ch=b['chance']/n
    sk=lambda x:(x/n-ch)/(1-ch)*100
    print(k,n,f"chance {100*ch:.0f} stock {100*b['stock']/n:.1f} (skill {sk(b['stock']):.1f}) clef {100*b['clef']/n:.1f} (skill {sk(b['clef']):.1f})")

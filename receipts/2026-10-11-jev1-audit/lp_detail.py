import gzip, json, re, hashlib
from collections import Counter, defaultdict
P = r'C:\Users\apumu\research\CHLabs\ssdm-data\m2\build\bschema-draw-jev1\records.jsonl.gz'
g = defaultdict(Counter); ins = defaultdict(Counter); pd_h = defaultdict(Counter); ex = {}
casekeys = Counter(); pdlen = []; caselen = []; nl = Counter(); casetype = Counter()
noul_empty = Counter(); first = {}
for l in gzip.open(P, 'rt', encoding='utf-8'):
    r = json.loads(l)
    if r['aug']['jev1']['family8'] != 'long_policy':
        if r['questions']['q']['type'] == 'noul':
            noul_empty[(r['aug']['jev1']['family8'], bool(r['questions']['q'].get('criteria')))] += 1
        continue
    q = r['questions']['q']; s = r['family']; t = q['type']
    g[(s, t)][str(r['labels']['q'])] += 1
    ins[s][q['instructions'][:110]] += 1
    st = r['state']; pd = st['policy_document']; cs = st['case']
    casetype[type(cs).__name__] += 1; casetype['pd:' + type(pd).__name__] += 1
    if isinstance(cs, dict): casekeys[tuple(cs)] += 1
    pdlen.append(len(pd) if isinstance(pd, str) else len(json.dumps(pd)))
    caselen.append(len(json.dumps(cs)))
    nl['pd_has_newline'] += isinstance(pd, str) and '\n' in pd
    pd_h[s][hashlib.sha1(re.sub(r'\d+', '#', json.dumps(pd, sort_keys=True)).encode()).hexdigest()] += 1
    noul_empty[('long_policy', bool(q.get('criteria')))] += 1 if t == 'noul' else 0
    if s not in first: first[s] = r
for k, c in sorted(g.items()):
    tot = sum(c.values()); print(k, tot, [(a, round(b / tot, 3)) for a, b in c.most_common(6)])
for s, c in ins.items(): print('\nINSTR', s, c.most_common())
print('\ncase/pd types', casetype, 'case keys', casekeys.most_common(5))
pdlen.sort(); caselen.sort()
print('pd chars p50/max', pdlen[len(pdlen)//2], pdlen[-1], 'case chars p50/max', caselen[len(caselen)//2], caselen[-1], nl)
for s, c in pd_h.items(): print('distinct policy docs (digits masked)', s, len(c), 'top reuse', c.most_common(3)[0][1])
print('noul (family, has criteria)', sorted(noul_empty.items()))
for s, r in first.items():
    st = r['state']
    print('\n---', s, 'case=', json.dumps(st['case'])[:400])
    print(' pd head=', json.dumps(st['policy_document'])[:700])
    print(' q=', json.dumps(r['questions']['q'])[:500], 'gold', r['labels'])

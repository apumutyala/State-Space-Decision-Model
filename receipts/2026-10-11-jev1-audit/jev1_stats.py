import gzip, json, hashlib, re, sys
from collections import Counter, defaultdict

P = r'C:\Users\apumu\research\CHLabs\ssdm-data\m2\build\bschema-draw-jev1\records.jsonl.gz'
fam = Counter(); sub = Counter(); typ = defaultdict(Counter); K = defaultdict(Counter)
gold = defaultdict(Counter); gpos = defaultdict(Counter); gposK = defaultdict(Counter)
lsrc = defaultdict(Counter); instr = defaultdict(Counter); skeys = defaultdict(Counter)
crit = defaultdict(Counter); src = defaultdict(Counter); tgt = defaultdict(Counter)
exact = Counter(); near = Counter(); nearfam = {}
order = []  # family8 sequence (run-length)
chars = defaultdict(list)
sortedcrit = defaultdict(Counter)
famsub_gold = defaultdict(Counter)
for line in gzip.open(P, 'rt', encoding='utf-8'):
    r = json.loads(line)
    j = r.get('aug', {}).get('jev1', {})
    f8 = j.get('family8', '?'); s = r['family']
    fam[f8] += 1; sub[(f8, s)] += 1
    if not order or order[-1][0] != f8: order.append([f8, 1])
    else: order[-1][1] += 1
    q = r['questions']['q']; c = q.get('criteria') or {}
    keys = list(c)
    typ[f8][q['type']] += 1; K[f8][len(keys)] += 1
    g = r['labels']['q']
    gold[f8][str(g)] += 1; famsub_gold[s][str(g)] += 1
    if g in keys:
        gpos[f8][keys.index(g)] += 1; gposK[(f8, len(keys))][keys.index(g)] += 1
    else:
        gpos[f8]['NA:' + str(g)[:20]] += 1
    lsrc[f8][j.get('label_source')] += 1
    instr[f8][q.get('instructions', '')[:90]] += 1
    skeys[f8][tuple(sorted(r['state']))] += 1
    crit[f8][tuple(keys) if len(keys) <= 6 else ('K=%d' % len(keys),)] += 1
    sortedcrit[f8][keys == sorted(keys)] += 1
    src[f8][r['source']] += 1; tgt[f8][j.get('target')] += 1
    body = json.dumps({'s': r['state'], 'q': q}, sort_keys=True)
    chars[f8].append(len(body))
    exact[hashlib.sha1(body.encode()).hexdigest()] += 1
    # near-dup: normalized concatenated state text (lowercase, digits stripped, whitespace) with sorted criteria
    t = re.sub(r'\d+', '#', re.sub(r'\s+', ' ', json.dumps(r['state'], sort_keys=True).lower()))
    t = t + '|' + q.get('instructions', '') + '|' + '|'.join(sorted(keys))
    h = hashlib.sha1(t.encode()).hexdigest(); near[h] += 1; nearfam[h] = f8

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]

out = {}
print('TOTAL', sum(fam.values())); print('FAM', dict(fam))
print('ORDER runs (first 40):', order[:40], 'nruns', len(order))
for f8 in sorted(fam):
    print('\n=====', f8, fam[f8])
    print(' subs', {k[1]: v for k, v in sub.items() if k[0] == f8})
    print(' sources', dict(src[f8])); print(' type', dict(typ[f8])); print(' target', dict(tgt[f8]))
    print(' label_source', dict(lsrc[f8]))
    print(' K', sorted(K[f8].items()))
    print(' gold top', gold[f8].most_common(12), 'ndistinct', len(gold[f8]))
    print(' gold pos', sorted(gpos[f8].items(), key=lambda x: str(x[0]))[:20])
    for (ff, k), c in sorted(gposK.items(), key=lambda x: (x[0][0], x[0][1])):
        if ff == f8 and sum(c.values()) > 200 and k <= 8:
            print('   K=%d pos' % k, sorted(c.items()))
    print(' criteria-sorted?', dict(sortedcrit[f8]))
    print(' crit sigs', crit[f8].most_common(6))
    print(' instr', instr[f8].most_common(6), 'ndistinct', len(instr[f8]))
    print(' state keys', skeys[f8].most_common(5))
    cs = chars[f8]; print(' chars p50/p90/p99/max', pct(cs, .5), pct(cs, .9), pct(cs, .99), max(cs))
dupe_ex = sum(v - 1 for v in exact.values() if v > 1)
nd = defaultdict(int)
for h, v in near.items():
    if v > 1: nd[nearfam[h]] += v - 1
print('\nEXACT dup rows (beyond first):', dupe_ex)
print('NEAR dup rows by family:', dict(nd))
print('\nSUB gold:')
for s, c in sorted(famsub_gold.items()):
    tot = sum(c.values()); print(' ', s, tot, c.most_common(4))

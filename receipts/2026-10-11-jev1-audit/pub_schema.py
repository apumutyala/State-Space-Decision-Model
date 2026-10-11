"""Schema-level only summary of the jevkit public items (eval-only; no content is printed or saved)."""
import json, sys, statistics
from collections import Counter, defaultdict
D = r'C:\Users\apumu\research\CHLabs\ssdm-data\jevbench\b6b8fff7e345b98c060ad26c13308860ddc67004'
rows = []
for f in ('original', 'easy', 'hard'):
    for l in open(D + '\\' + f + '.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l); r['_file'] = f; rows.append(r)
print('n', len(rows), 'top keys', Counter(tuple(sorted(r)) for r in rows).most_common(3))
by = defaultdict(list)
for r in rows: by[r.get('family')].append(r)
def stat(xs):
    xs = sorted(xs); return (xs[0], xs[len(xs)//2], xs[-1]) if xs else None
for fam, rs in sorted(by.items()):
    t = Counter(); K = Counter(); stype = Counter(); nkeys = []; slen = []; ilen = []; noulk = Counter(); epos = Counter()
    depth = Counter(); qkeys = Counter(); exp_t = Counter(); files = Counter(); instr_type = Counter(); lab = Counter()
    for r in rs:
        q = r.get('question') or {}
        qkeys[tuple(sorted(q))] += 1
        t[q.get('type')] += 1; files[r['_file']] += 1
        c = q.get('criteria')
        K[len(c) if c is not None else 0] += 1
        s = r.get('state'); stype[type(s).__name__] += 1
        if isinstance(s, dict):
            nkeys.append(len(s))
            depth[max((1 + (isinstance(v, (dict, list))) for v in s.values()), default=0)] += 1
        slen.append(len(json.dumps(s, ensure_ascii=False, separators=(',', ':'))))
        ins = q.get('instructions'); instr_type[type(ins).__name__] += 1
        ilen.append(len(ins if isinstance(ins, str) else json.dumps(ins)))
        if q.get('type') == 'noul': noulk[tuple(sorted(c)) if isinstance(c, dict) else type(c).__name__] += 1
        if q.get('type') == 'choice' and isinstance(c, dict) and r.get('expected') in list(c):
            epos[(len(c), list(c).index(r['expected']))] += 1
        exp_t[type(r.get('expected')).__name__] += 1
        lab[tuple(r.get('labels'))[:0] if q.get('type') != 'noul' else tuple(sorted(r.get('labels')))] += 1
    print(f'\n== {fam} n={len(rs)} files={dict(files)}')
    print('  qtype', dict(t), 'K', sorted(K.items()), 'question keys', dict(qkeys))
    print('  state type', dict(stype), 'n top keys (min/med/max)', stat(nkeys), 'nested-depth', dict(depth))
    print('  state chars (min/med/max)', stat(slen), ' instr chars', stat(ilen), 'instr type', dict(instr_type))
    print('  noul criteria keys', dict(noulk), ' expected type', dict(exp_t))
    print('  choice expected (K,pos)', sorted(epos.items()))

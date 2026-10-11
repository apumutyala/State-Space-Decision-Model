"""Reproduce the rows r2 saw: draw order = sha256('0:'+id) (train.run_cell.split_ids via train.train_set.draw),
cell_items keeps order (all 96,000 pass K<=255 / T<=2048), Trainer._order(epoch 0) = torch.randperm(n, Generator(seed*100003+0))."""
import gzip, json, hashlib
from collections import Counter, defaultdict
import torch
P = r'C:\Users\apumu\research\CHLabs\ssdm-data\m2\build\bschema-draw-jev1\records.jsonl.gz'
meta = {}
for l in gzip.open(P, 'rt', encoding='utf-8'):
    r = json.loads(l); q = r['questions']['q']
    meta[r['id']] = (r['aug']['jev1']['family8'], r['family'], q['type'], str(r['labels']['q']), r['source'])
ids = sorted(meta, key=lambda i: hashlib.sha256(f"0:{i}".encode()).hexdigest())
perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0 * 100_003 + 0)).tolist()
for n_rows, tag in ((8968, '1,121 steps x 8'), (8000, 'best ckpt step 1000 x 8')):
    seen = [ids[i] for i in perm[:n_rows]]
    fam = Counter(meta[i][0] for i in seen); src = Counter('synth' if meta[i][4] == 'jev1synth' else 'dataset' for i in seen)
    print(f'\n{tag}: {n_rows} rows', dict(sorted(fam.items())), dict(src))
    # first/last thirds of the run: family mix drift
    for a, b in ((0, n_rows // 3), (n_rows // 3, 2 * n_rows // 3), (2 * n_rows // 3, n_rows)):
        print('  rows %d-%d' % (a, b), dict(sorted(Counter(meta[i][0] for i in seen[a:b]).items())))
    lp = Counter((meta[i][1], meta[i][2]) for i in seen if meta[i][0] == 'long_policy')
    print('  long_policy sub/type', sorted(lp.items()))
    g = Counter(meta[i][3] for i in seen if meta[i][0] == 'long_policy' and meta[i][2] == 'noul')
    print('  long_policy noul gold', dict(g))
    # longest same-family run in the seen sequence
    best, cur = 1, 1
    for x, y in zip(seen, seen[1:]):
        cur = cur + 1 if meta[x][0] == meta[y][0] else 1; best = max(best, cur)
    print('  longest same-family run', best)

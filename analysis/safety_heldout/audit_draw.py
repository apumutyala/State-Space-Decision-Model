import json, sys, gzip, collections, hashlib, re
sys.path.insert(0, '.')
from train.train_set import read_pointer, files
def split_ids(ids,n,ne,seed):
    o=sorted(ids,key=lambda i: hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest()); return o[ne:ne+n],o[:ne]
from train.cohort_v3 import POINTERS, POOL_150K, cohort_v3, training_exclusion
from train.cohort import read_content_map
def read_manifest(path):
    with gzip.open(path,'rt',encoding='utf-8') as fh: return {e['id']:e['sha256'] for e in (json.loads(x) for x in fh if x.strip())}
from train.train_set import heldout_excluded
cfg = json.load(open('jobs/m2_first_cell/m2_first_cell_rwkv7.json'))
TD = 'C:/Users/apumu/research/CHLabs/ssdm-data/m2/build/sample-150000-seed20261006'
ptr = read_pointer(cfg['train_set']['pointer']); fs = files(TD, ptr)
man = read_manifest(fs['manifest']); content = read_content_map(fs['content_map'])
v3x = training_exclusion(man, content, cohort_v3(), pool=POOL_150K)
pool = sorted(set(man) - heldout_excluded(cfg) - v3x['ids'])
tr, _ = split_ids(pool, int(cfg['n_train_records']), 0, cfg['seed'])
print('drawn', len(tr), 'sha', hashlib.sha256('\n'.join(tr).encode()).hexdigest())
want = set(tr)
inj = collections.Counter(); fam = collections.Counter(); pos = collections.Counter(); field = collections.Counter()
fc = 0; pat = collections.Counter()
PATS = {'ignore_prev': r'ignore (all )?(previous|prior|above) instructions', 'im_start': r'<\|im_start\|>', 'system_colon': r'(^|\n)\s*SYSTEM:',
        'roleplay': r'\b(pretend|role-?play|you are now|act as)\b', 'zero_width': '[\u200b\u200c\u200d\u2060\ufeff]', 'bidi': '[\u202a-\u202e\u2066-\u2069]',
        'base64ish': r'\bbase64\b', 'tool_result_tag': r'</?(tool|function)_?(result|response|output)>', 'assistant_tag': r'\[\[assistant\]\]|^assistant:',
        'new_instructions': r'new instructions', 'override': r'\boverrid'}
with gzip.open(fs['records'], 'rt', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r['id'] not in want: continue
        a = r.get('aug') or {}
        if 'fact_change' in a: fc += 1
        if 'injection' in a:
            i = a['injection']; inj[f"{i['family']}:{i['template']}"] += 1; fam[i['family']] += 1; pos[i['inserted_at']] += 1
            field[','.join(r.get('untrusted_fields') or [])] += 1
        txt = json.dumps(r['state'], ensure_ascii=False) + json.dumps(r['questions'], ensure_ascii=False)
        for k, p in PATS.items():
            if re.search(p, txt, re.I | re.M): pat[k] += 1
print('fact_change', fc); print('families', dict(fam))
for k, v in sorted(inj.items()): print(' ', k, v)
print('insert pos', dict(sorted(pos.items())[:12]))
print('fields', field.most_common(12))
print('regex hits (any record)', dict(pat))

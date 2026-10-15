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

import re
from data.injection import ngram_sha1s
from data import text as T
H=json.load(open(r"analysis/safety_heldout/heldout_v1_templates.json"))
G={}
for k,v in H.items():
    for t in (v['text'], v.get('payload','')):
        for f in re.split(r'\$[A-Z0-9_]+', t):
            if f.strip():
                for g in ngram_sha1s(f): G[g]=k
hits={}
n=0
with gzip.open(fs['records'],'rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line)
        if r['id'] not in want: continue
        n+=1
        for seg in T.raw_segments(r['state'], r['questions']):
            for g in ngram_sha1s(seg) & G.keys(): hits.setdefault(G[g],set()).add(r['id'])
print('scanned',n,'templates with any 6-gram in drawn records:',{k:len(v) for k,v in hits.items()})

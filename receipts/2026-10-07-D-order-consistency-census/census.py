
import os,sys,json,hashlib,time,functools,collections,math
from pathlib import Path
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
# Pin the measured training implementation; a changed source requires a new receipt.
EXPECTED_SOURCE_SHA256={
  "heads/layout.py": "b908078f44c854da747cb362436b6904fc4fa53a0a67c80621cf20a34da58412",
  "jobs/m2_first_cell/m2_first_cell_rwkv7.json": "60c6e45253164bc40a052964c3f6de674b473fa886aa6ad15dfe27c3846f8ec2",
  "third_party/rwkv7_hf/tokenization_rwkv7.py": "58406b0aa5f51d5ca3a698602825ae9b299f6621e6c20bb0a2915d43c9f8bf65",
  "train/cohort.py": "aa1b5871aeaf968ed76ed3dcb32a6aa6d7c7b4276cdfd740dbf4ff203690bb24",
  "train/cohort_v3.py": "5f802963fe7931e6935e8cc3d6ba73aba3ca5e6238ee87616ee95dc8876531a9",
  "train/data.py": "83fa6e04827b61077c26cd0e1fd02d8c8d01d32bfcbf4af3b2e849ec349b8f21",
  "train/order_views.py": "e2e823399dc7ccd5dbebda636c99622ddd57a2cb6aeb773da1b60a9f2bcba456",
  "train/run_cell.py": "7f4f072fdc080b76d99ed5f205158457ade2a227c772c2d84b07c29cede77be1",
  "train/train_set.py": "b5d483194047499c567545e11c7deeecee049e55630b94ec30573b6329d308be"
}
for source_path, expected_hash in EXPECTED_SOURCE_SHA256.items():
    assert hashlib.sha256((ROOT/source_path).read_bytes()).hexdigest()==expected_hash, source_path
from train.train_set import draw
from train.guard import SplitGuard
from train.run_cell import parent_encoder,keep_item
from train.order_views import reversed_record
from train.data import items_from_records
from decide.schema import canonical_options
cfg_path=ROOT/'jobs/m2_first_cell/m2_first_cell_rwkv7.json'
cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
train_dir=Path('C:/Users/apumu/research/CHLabs/ssdm-data/m2/build/sample-150000-seed20261006')
tokdir=Path('C:/Users/apumu/research/CHLabs/ssdm-data/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba')
assert hashlib.sha256((tokdir/'tokenizer.json').read_bytes()).hexdigest()=='1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf'
print('CENSUS_START train-only draw=24000 seed=0; no model construction',flush=True)
recs,ids,ident=draw(cfg,train_dir)
# Aggregate-only export: the verifier carries three pinned record IDs internally.
excluded_ids=ident.pop('heldout_excluded')
ident['heldout_excluded_count']=len(excluded_ids)
ident['heldout_excluded_ordered_ids_sha256']=hashlib.sha256('\n'.join(excluded_ids).encode()).hexdigest()
print('DRAW_VERIFIED '+json.dumps(ident,sort_keys=True),flush=True)
guard=SplitGuard.from_manifests([ROOT/'harness/manifests/dev-manifest.jsonl.gz',ROOT/'harness/manifests/test-manifest.jsonl.gz'],content_index='C:/Users/apumu/m2fc_uhxb5_out/x_final/out/guard/index-content-hashes.txt.gz')
assert guard.identity()['content_index']['sha256']=='7cbe6c89e005d0eecf6d163af1ff3f621c867392574a8f4eb5bff79fad052d8d'
rawenc=parent_encoder('rwkv7',str(tokdir))
@functools.lru_cache(maxsize=4096)
def cached(text,trusted=True): return tuple(rawenc(text,trusted))
def enc(text,trusted=True): return list(cached(text,trusted))
enc.identity=rawenc.identity
counts=collections.Counter();byk=collections.defaultdict(collections.Counter);bysource=collections.defaultdict(collections.Counter)
rungs={t:collections.Counter() for t in [512,1024,2048,4096]}
lengths={'base_k_le_16':[],'eligible_base':[],'eligible_view':[],'eligible_max':[]}
basehash={t:hashlib.sha256() for t in rungs};pairhash={t:hashlib.sha256() for t in rungs}
start=time.monotonic()
for ix,rid in enumerate(ids,1):
    rec=recs[rid]
    guard.check(rec)
    src=rid.split(':')[1] if ':' in rid else 'unknown'
    # Same integrated items_from_records/build_row path. Nothing from held-out roles is rendered.
    its=items_from_records([rec],enc,layout=cfg['layout'],template=cfg['template'],guard=guard)
    for it in its:
        qid=it.row.qid;k=it.row.n_options;bl=len(it.row.ids)
        counts['all_items']+=1;counts['type_'+it.row.qtype]+=1
        byk[k]['all']+=1
        if k>cfg['max_k']:
            counts['k_gt_16']+=1;continue
        counts['k_le_16']+=1;byk[k]['k_le_16']+=1;bysource[src]['base_k_le_16']+=1
        lengths['base_k_le_16'].append(bl)
        contok=max(map(len,it.row.continuations))<=16
        if not contok: counts['continuation_gt_16']+=1
        for t in rungs:
            if keep_item(it.row,cfg['max_k'],t):
                rungs[t]['base']+=1;basehash[t].update((it.id+'\n').encode())
        viewgot=reversed_record(rec,qid)
        if viewgot is None:
            counts['no_eligible_view']+=1;continue
        vrec,perm=viewgot
        vit=items_from_records([{**vrec,'labels':{qid:vrec['labels'][qid]}}],enc,layout=cfg['layout'],template=cfg['template'],guard=guard)
        assert len(vit)==1
        v=vit[0];vl=len(v.row.ids)
        assert v.row.option_ids==it.row.option_ids
        assert it.row.option_ids[perm[v.label]]==it.row.option_ids[it.label]
        counts['eligible_view_before_T']+=1;byk[k]['eligible_view_before_T']+=1;bysource[src]['eligible_view_before_T']+=1
        if v.label!=it.label: counts['eligible_moved_gold']+=1
        lengths['eligible_base'].append(bl);lengths['eligible_view'].append(vl);lengths['eligible_max'].append(max(bl,vl))
        for t in rungs:
            if keep_item(it.row,cfg['max_k'],t):
                if keep_item(v.row,cfg['max_k'],t):
                    rungs[t]['paired']+=1;byk[k]['paired_T'+str(t)]+=1;bysource[src]['paired_T'+str(t)]+=1
                    rungs[t]['base_tokens_paired']+=bl;rungs[t]['view_tokens']+=vl
                    pairhash[t].update((it.id+'\n').encode())
                    if v.label!=it.label:rungs[t]['moved_gold']+=1
                else:rungs[t]['base_kept_view_dropped']+=1
    if ix%1000==0:print('PROGRESS '+json.dumps({'records':ix,'elapsed_s':round(time.monotonic()-start,1),'counts':dict(counts)}),flush=True)
def summary(xs):
    xs=sorted(xs);n=len(xs)
    return {'n':n,'min':xs[0] if n else None,'p50':xs[math.floor((n-1)*.5)] if n else None,'p80':xs[math.floor((n-1)*.8)] if n else None,'p95':xs[math.floor((n-1)*.95)] if n else None,'p99':xs[math.floor((n-1)*.99)] if n else None,'max':xs[-1] if n else None,'sum':sum(xs)}
source_files=['train/train_set.py','train/cohort.py','train/cohort_v3.py','train/order_views.py','train/data.py','heads/layout.py','train/run_cell.py','third_party/rwkv7_hf/tokenization_rwkv7.py','jobs/m2_first_cell/m2_first_cell_rwkv7.json']
report={'counts':dict(counts),'by_k':dict(sorted(byk.items())),'by_source':dict(sorted(bysource.items())),'rungs':{t:dict(c)|{'base_ordered_ids_sha256':basehash[t].hexdigest(),'paired_ordered_ids_sha256':pairhash[t].hexdigest()} for t,c in rungs.items()},'lengths':{k:summary(v) for k,v in lengths.items()},'train_identity':ident,'guard':guard.identity(),'encoder':enc.identity(),'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},'wall_s':round(time.monotonic()-start,3),'cached_segments':cached.cache_info()._asdict()}
print('CENSUS_RESULT '+json.dumps(report,sort_keys=True),flush=True)

"""Temporary exact alpha-renaming probe; no repository edits or model imports."""
import ast
import builtins
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import string
import subprocess
import sys
import time
import types

ROOT = Path('C:/Users/apumu/research/CHLabs/ssdm-wt/agent-c-multirecord-synth')
TRAINER = Path('C:/Users/apumu/research/CHLabs/ssdm-wt/b-train-v3w')
COMMIT = '9c3e4350ccb41bec1385d58b9fe7002184f69626'
TOK = Path('C:/Users/apumu/research/CHLabs/ssdm-data/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba')
OUT = Path(os.environ['TEMP'])/'c-k77-alpha-rename-prototype.json'
sys.path.insert(0, str(ROOT))
sys.path.insert(1, 'C:/Users/apumu/research/CHLabs/ssdm-data/decision-index')
os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TORCH='0', USE_TF='0', USE_FLAX='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
old_import = builtins.__import__
def no_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('model/Torch imports forbidden')
    return old_import(name, *args, **kwargs)
builtins.__import__ = no_torch
from data.c_multirecord_synth import BoundedEncodingCache, TOKENIZER_PINS, _sha_file, _load_encoder, _load_layout_without_package_initializers, canonical, make_record, verify_record
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=TRAINER,text=True).strip() == COMMIT
assert {n:_sha_file(TOK/n) for n in TOKENIZER_PINS} == TOKENIZER_PINS
assert _sha_file(ROOT/'third_party/rwkv7_hf/tokenization_rwkv7.py') == _sha_file(TRAINER/'third_party/rwkv7_hf/tokenization_rwkv7.py')
names = {'RwkvSegmentEncoder','parent_encoder','host_provenance','to_train_record'}
tree = ast.parse((TRAINER/'train/run_cell.py').read_text(encoding='utf-8'))
nodes = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
assert {n.name for n in nodes} == names
env = {'hashlib':hashlib,'json':json}
build_row = _load_layout_without_package_initializers(TRAINER)
package = types.ModuleType('train')
package.__path__ = [str(TRAINER/'train')]
sys.modules['train'] = package
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(TRAINER/'train/run_cell.py'),'exec'),env)
tokenizer, _ = _load_encoder(TOK)
encoder = env['RwkvSegmentEncoder'](tokenizer,set(tokenizer.all_special_ids))
cached = BoundedEncodingCache(encoder)
cached.token_bytes, cached.identity, cached.is_rwkv7_g1 = encoder.token_bytes, encoder.identity, encoder.is_rwkv7_g1
from heads.layout import V3W_ANSWER_CTX, v3w_first_token
from decide.prompt_g1v3 import load_v3_codes, word_ids_ok
from train.data import items_from_records
from harness.heldout import check_record
entries, code_sha = load_v3_codes()
base = cached(V3W_ANSWER_CTX,True)
special = set(tokenizer.all_special_ids)|{0}
alias_values = set()
def collect_aliases(value, inside_alias=False):
    if isinstance(value,dict):
        for k,v in value.items():
            collect_aliases(v,inside_alias or 'alias' in k.lower())
    elif isinstance(value,list):
        for v in value: collect_aliases(v,inside_alias)
    elif inside_alias and isinstance(value,str):
        alias_values.add(value.casefold())
metadata_paths = [TRAINER/'decide/data/rwkv_g1_v3_codes.json',TRAINER/'decide/data/rwkv_g1_filtered_v1.json']
for path in metadata_paths:
    if path.exists(): collect_aliases(json.loads(path.read_bytes()))
def single(oid):
    ids = cached(V3W_ANSWER_CTX+' '+oid,True)
    token = v3w_first_token(cached,oid)
    return token if ids[:len(base)]==base and len(ids)==len(base)+1 and ids[-1]==token and token not in special else None
available = {'letters':[],'words':[],'opaque':[]}
safe_symbols = set(string.punctuation) - set('<>`\"\'\\')
for entry in entries:
    oid=entry['code']
    if re.fullmatch('[A-Z]{1,2}',oid) and oid.casefold() not in alias_values and single(oid) is not None:
        assert single(oid)==entry['token_id']
        available['letters'].append(oid)
for token,raw in tokenizer._token_bytes.items():
    if not raw.startswith(b' '): continue
    try: oid=raw[1:].decode('ascii')
    except UnicodeDecodeError: continue
    if oid.casefold() in alias_values or oid in {'true','false','null','none'}: continue
    if re.fullmatch('[a-z]{3,8}',oid): style='words'
    elif 2<=len(oid)<=8 and all(ch in safe_symbols for ch in oid): style='opaque'
    else: continue
    if single(oid) is not None: available[style].append(oid)
pools = {}
for style,candidates in available.items():
    candidates = list(dict.fromkeys(candidates))
    if style!='letters': candidates.sort(key=lambda oid:(len(oid),oid))
    seen, chosen = set(), []
    for oid in candidates:
        token=single(oid)
        if token not in seen:
            seen.add(token); chosen.append(oid)
        if len(chosen)==77: break
    pools[style]=sorted(chosen)
report = {'scope':'temporary synthetic K77 alpha-renaming prototype; no model/Index claim','trainer_commit':COMMIT,
          'trainer_source_sha256':{name:_sha_file(TRAINER/name) for name in ('train/run_cell.py','train/data.py','heads/layout.py','decide/prompt.py','decide/prompt_g1v3.py','third_party/rwkv7_hf/tokenization_rwkv7.py')},
          'tokenizer_sha256':TOKENIZER_PINS,'v3_code_table_sha256':code_sha,
          'metadata_sha256':{str(p.relative_to(TRAINER)):_sha_file(p) for p in metadata_paths if p.exists()},
          'excluded_alias_values':sorted(alias_values),'available_pool_sizes':{s:len(set(v)) for s,v in available.items()},
          'pools':{s:[{'id':oid,'answer_token':single(oid)} for oid in ids] for s,ids in pools.items()},'cases':[]}
if any(len(ids)<77 for ids in pools.values()):
    report.update(success=False,reason='fewer than 77 valid unique complete single-token IDs in a required pool')
    OUT.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'success':False,'available_pool_sizes':report['available_pool_sizes'],'report':str(OUT)}),flush=True)
    raise SystemExit(2)

def rename(old):
    row=deepcopy(old)
    qmap=dict(zip(sorted(old['questions']),string.ascii_lowercase[:7]))
    old_roles=old['aug']['question_roles']
    choice=next(q for q,m in old_roles.items() if m['kind']=='top_profile')
    style=old_roles[choice]['id_style']
    oids=sorted(old['questions'][choice]['criteria'])
    newids=pools[style]
    omap=dict(zip(oids,newids,strict=True))
    row['questions'][choice]['criteria']={omap[oid]:deepcopy(old['questions'][choice]['criteria'][oid]) for oid in oids}
    row['labels'][choice]=omap[old['labels'][choice]]
    row['questions']={qmap[q]:row['questions'][q] for q in sorted(old['questions'])}
    row['labels']={qmap[q]:row['labels'][q] for q in sorted(old['labels'])}
    row['aug']['question_roles']={qmap[q]:deepcopy(old_roles[q]) for q in sorted(old_roles)}
    for name in ('case_exact','multi_select'):
        row['aug'][name]['question_ids']=[qmap[q] for q in old['aug'][name]['question_ids']]
    row['id']=old['id']+'~view:alpha-k77-v1'
    row['aug']['alpha_view']={'version':1,'source_record_id':old['id'],'pool_sha256':hashlib.sha256(canonical(pools[style])).hexdigest()}
    # Invert both identifier maps and prove all model-visible meanings/labels/order unchanged.
    inverse={v:k for k,v in omap.items()}
    restored={q:deepcopy(row['questions'][qmap[q]]) for q in old['questions']}
    restored[choice]['criteria']={inverse[oid]:value for oid,value in restored[choice]['criteria'].items()}
    restored_labels={q:row['labels'][qmap[q]] for q in old['labels']}
    restored_labels[choice]=inverse[restored_labels[choice]]
    assert restored==old['questions'] and restored_labels==old['labels'] and row['state']==old['state']
    assert all(row['questions'][qmap[q]]['instructions']==old['questions'][q]['instructions'] for q in old['questions'])
    assert sorted(row['questions']).index(qmap[choice])==sorted(old['questions']).index(choice)
    assert sorted(row['questions'][qmap[choice]]['criteria']).index(row['labels'][qmap[choice]])==old_roles[choice]['gold_position']
    verify_record(row); check_record(row)
    return row,qmap

started=time.perf_counter()
for i in range(32):
    index=75+76*(i%3)+228*(i*11)
    old=make_record(20261008,index,revision=2)
    verify_record(old); check_record(old)
    row,qmap=rename(old)
    choice=next(q for q,m in old['aug']['question_roles'].items() if m['kind']=='top_profile')
    case={'index':index,'style':old['aug']['question_roles'][choice]['id_style'],'shape':old['aug']['schema_shape'],'gold_position':old['aug']['question_roles'][choice]['gold_position'],'old':{},'new':{}}
    for label,record in (('old',old),('new',row)):
        case[label]['serialized_record_tokens']=len(cached(canonical(record).decode('utf-8'),True))
        tr=env['to_train_record'](record)
        for key,template in (('v2f','jrt-rwkv-g1-v2f'),('v3w','jrt-rwkv-g1-v3w')):
            items=items_from_records([tr],cached,layout='g1v2',template=template)
            assert len(items)==7
            vals=[]
            for item in items:
                direct=build_row(tr['request'],item.row.qid,cached,layout='g1v2',template=template)
                assert direct==item.row or all(getattr(direct,n)==getattr(item.row,n) for n in ('ids','option_ids','continuations','option_spans','state_span','field_span'))
                vals.append({'qid':item.row.qid,'prompt':len(item.row.ids),'max_cont':max(map(len,item.row.continuations)),
                             'full':len(item.row.ids)+max(map(len,item.row.continuations)),'k':item.row.n_options,'words':bool(item.row.meta.get('words',False)),
                             'catalog_chars':item.row.meta.get('catalog_chars')})
            assert max(v['k'] for v in vals)==77
            case[label][key]={'max_full':max(v['full'] for v in vals),'rows':vals}
    report['cases'].append(case)
    if (i+1)%8==0: print('K77 alpha prototype: '+str(i+1)+'/32',flush=True)
report.update(success=all(c['new'][key]['max_full']<=2048 for c in report['cases'] for key in ('v2f','v3w')) and all(c['new']['serialized_record_tokens']<=2048 for c in report['cases']),
              semantic_roundtrip_equal=True,verify_record_check_record_passed=True,direct_trainer_row_equality=True,
              seconds=time.perf_counter()-started,torch_imported='torch' in sys.modules)
report['max_full']={label:{key:max(c[label][key]['max_full'] for c in report['cases']) for key in ('v2f','v3w')} for label in ('old','new')}
report['max_serialized_record_tokens']={label:max(c[label]['serialized_record_tokens'] for c in report['cases']) for label in ('old','new')}
OUT.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ('success','max_full','semantic_roundtrip_equal','seconds','available_pool_sizes','torch_imported')},sort_keys=True),flush=True)

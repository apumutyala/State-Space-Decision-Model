#!/usr/bin/env python3
"""Recompute the pinned operation diagnostic from JSON artifacts, without inference.

Run with Python -B so importing the semantic planner writes no bytecode files.
Only the dedicated independent-remeasurement.json is written. Model, checkpoint
and feature-cache binaries are never opened. Assertion failures refuse an audit
result rather than silently treating incomplete evidence as a complete run.
"""
from __future__ import annotations

import argparse
import sys
import json
import gzip
import hashlib
import random
import math
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from decimal import Decimal
from collections import Counter, defaultdict

# Keep the script callable from any working directory, without importing models.
sys.dont_write_bytecode = True
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))
from language_growth.planner import plan_checked_answer
from language_growth.kernel import check_program
from language_growth.realize import render_checked
from language_growth.types import Fact, Comparison, SpeechAct


def main() -> None:
    if not __debug__:
        raise RuntimeError("Assertions are required; do not run Python with -O.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SCRIPT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    expected_out = root / "receipts/2026-10-07-L-operation-only-run1/independent-remeasurement.json"
    out = args.out.resolve() if args.out else expected_out
    if root != SCRIPT_ROOT:
        raise ValueError("--root must identify this script's worktree and pinned semantic source.")
    if out != expected_out:
        raise ValueError("Only the dedicated independent-remeasurement.json may be written.")
    assert 'torch' not in sys.modules
    P=root/'receipts/2026-10-07-L-operation-only-run1'
    C=root/'receipts/2026-10-06-L-operation-only-cache-prepared/corpus.json.gz'
    def read(p): return json.loads(p.read_text(encoding='utf-8'))
    def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    def dig(x): return hashlib.sha256(canon(x).encode()).hexdigest()
    def filehash(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    R=read(P/'receipt.json'); G=read(P/'registration.json')
    D=json.loads(gzip.decompress(C.read_bytes())); O=read(P/'operation-records.json'); T=read(P/'targets.json')
    E=[json.loads(s) for s in (P/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    assert R['schema']=='ssdm.operation-only-cache-result.v1' and R['status']=='complete'
    assert E[-1]['event']=='launcher_exit' and E[-1]['exit_code']==0
    assert not (P/'failure.json').exists()
    assert filehash(P/'registration.json')==R['registration_sha256']
    assert dig(D)==G['corpus_sha256']=='cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466'
    diff=[]
    def compare(label,a,b):
        if a!=b: diff.append({'field':label,'recomputed':a,'saved':b})
    for split,n in [('development_train',24),('development_validation',12)]:
        assert len(D[split])==n and len({e['task_id'] for e in D[split]})==n
        reconstructed=[]
        for e in D[split]:
            req=e['model_inputs'][0]; tar=e['training_only']['targets'][0]
            assert tar['slot']=='operation' and set(req['questions'])=={'slot-001-operation'}
            reconstructed.append({'id':e['task_id'],'request':req,'labels':{'slot-001-operation':tar['option_id']}})
        assert reconstructed==O[split]
        assert dig(O[split])==G['operation_records_sha256'][split]
        assert [r['id']+'/slot-001-operation' for r in O[split]]==G['operation_row_ids'][split]
        target_map={r['row_id']:r['label'] for r in T[split]}
        assert len(target_map)==n and len(T[split])==n
        for r in O[split]:
            q=r['request']['questions']['slot-001-operation']
            expected=sorted(q['criteria']).index(r['labels']['slot-001-operation'])
            assert target_map[r['id']+'/slot-001-operation']==expected

    examples=D['development_validation']; byid={e['task_id']:e for e in examples}
    assert Counter(e['training_only']['kind'] for e in examples)=={'read':4,'explain':4,'compare':4}
    state_map=defaultdict(list)
    for ex in examples: state_map[canon(json.loads(ex['model_inputs'][0]['state']))].append(ex['task_id'])
    assert all(len(v)==1 for v in state_map.values())
    reqs={}; responses={}; exceptions={}
    groups=defaultdict(list)
    for event in E:
        if event['event']=='request':
            req=event['request']; key=(event['phase'],req['ssdm']['request_id'])
            assert key not in reqs
            state=json.loads(req['state']); state['proposed_slot_choices']=[]
            task=state_map[canon(state)][0]
            reqs[key]={'request':req,'task_id':task}; groups[(event['phase'],task)].append(key)
        elif event['event']=='response':
            response=event['response']; key=(event['phase'],response['ssdm']['request_id'])
            assert key not in responses
            responses[key]=response
        elif event['event']=='provider_exception':
            key=(event['phase'],event['request_id']); assert key not in exceptions; exceptions[key]=event
        else: assert event['event']=='launcher_exit'
    assert set(responses)<=set(reqs) and set(exceptions)<=set(reqs)
    assert set(responses).isdisjoint(exceptions)
    assert set(reqs)==set(responses)|set(exceptions)
    assert not exceptions

    def asserted(out):
        explained=out.kind=='explain'
        act=out.content if explained else out
        if isinstance(act,SpeechAct) and act.kind=='assert' and isinstance(act.content,(Fact,Comparison)):
            return explained,act.content
        return None
    def useful(ex,result):
        if result.status!='accepted' or result.program is None or result.text is None:return False
        tr=ex['training_only']; host,cat=tr['host'],tr['catalog']
        gold=check_program(tr['program'],host,cat); actual=check_program(result.program,host,cat)
        assert gold.status=='accepted'
        if actual.status!='accepted' or result.text!=render_checked(actual,host,cat):return False
        if len(gold.outputs)!=1 or len(actual.outputs)!=1:return False
        target,answer=asserted(gold.outputs[0]),asserted(actual.outputs[0])
        if target is None or answer is None:return False
        expected_explained,ec=target; got_explained,ac=answer
        if tr['kind']=='explain' and not got_explained:return False
        if tr['kind'] in ('read','explain'):return isinstance(ec,Fact) and isinstance(ac,Fact) and ac==ec
        if tr['kind']=='compare' and isinstance(ec,Comparison) and isinstance(ac,Comparison):
            if ac.left==ec.left and ac.right==ec.right:return ac.relation==ec.relation
            if ac.left==ec.right and ac.right==ec.left:return ac.relation=={'equal':'equal','greater_than':'less_than','less_than':'greater_than'}[ec.relation]
        return False

    ALL={}; audit_summary={}; mismatches=[]
    for phase in ['stock','zero_update','trained']:
        saved=read(P/(phase+'.json'))
        assert saved==R[phase] and len(saved)==12 and {r['task_id'] for r in saved}==set(byid)
        savedby={r['task_id']:r for r in saved}; rows=[]
        for ex in examples:
            tid=ex['task_id']; keys=groups[(phase,tid)]; assert keys
            first=reqs[keys[0]]['request']; firstresponse=responses[keys[0]]
            class Replay:
                cursor=0
                def decide(self,request):
                    assert self.cursor<len(keys),('extra call',phase,tid)
                    key=keys[self.cursor]; self.cursor+=1
                    assert request==reqs[key]['request'],('request replay differs',phase,tid,self.cursor)
                    if key in exceptions:raise RuntimeError('recorded provider exception')
                    return deepcopy(responses[key])
            replay=Replay(); tr=ex['training_only']; prefix=first['ssdm']['request_id'].rsplit('-',1)[0]
            cfg={k:firstresponse['ssdm']['served_config'][k] for k in ['calibration','policy_masking']}
            with patch('language_growth.planner.uuid.uuid4',return_value=SimpleNamespace(hex=prefix)):
                result=plan_checked_answer(json.loads(first['state'])['task_input'],tr['host'],tr['catalog'],replay,model=first['model'],parent_revision=first['ssdm']['parent_revision'],expected_served_config=cfg,allow_repair=False,candidate_style='record_ref')
            assert replay.cursor==len(keys),('unconsumed calls',phase,tid)
            row={'task_id':tid,'status':result.status,'reason':result.reason,'kernel_accepted':result.status=='accepted','task_correct':useful(ex,result),'text':result.text,'program':result.program,'provider_calls':result.provider_calls,'repair_requests':result.repair_requests,'forward_calls':result.forward_calls}
            for k,v in row.items():compare(phase+'/'+tid+'/'+k,v,savedby[tid][k])
            reference={}
            for request,target in zip(ex['model_inputs'],tr['targets']):
                q=next(iter(request['questions'].values()))
                reference[target['slot']]=q['criteria'][target['option_id']]
            slots=[]
            for key in keys:
                request=reqs[key]['request']; qid,q=next(iter(request['questions'].items())); slot=qid.split('-',2)[-1]
                answer=(responses.get(key) or {}).get('answers',{}).get(qid,{}).get('choice')
                actual=q['criteria'].get(answer) if isinstance(answer,str) else None; expected=reference.get(slot)
                scored=slot in reference and actual is not None and expected is not None
                slots.append({'slot':slot,'selected':answer,'selected_description':actual,'expected_description':expected,'scored':scored,'matches_reference_slot':scored and actual==expected})
            row['slot_attribution']=slots; row['missing_reference_slots']=sorted(set(reference)-{s['slot'] for s in slots})
            compare(phase+'/'+tid+'/slots',slots,savedby[tid]['slot_attribution'])
            compare(phase+'/'+tid+'/missing',row['missing_reference_slots'],savedby[tid]['missing_reference_slots'])
            rows.append(row)
        ALL[phase]=rows
        observed=[s for r in rows for s in r['slot_attribution'] if s['slot']!='operation']
        downstream={'worlds':12,'failed_worlds':sum(r['status']!='accepted' for r in rows),'accepted_wrong_task':sum(r['kernel_accepted'] and not r['task_correct'] for r in rows),'observed_later_slots':len(observed),'correct_later_slots':sum(s['matches_reference_slot'] for s in observed),'missing_reference_slots':[{'task_id':r['task_id'],'slots':r['missing_reference_slots']} for r in rows],'wrong_relation_slots':sum(s['slot']=='relation' and not s['matches_reference_slot'] for s in observed)}
        compare(phase+'/downstream',downstream,R['downstream'][phase])
        kind_counts={kind:{'n':4,'operation':sum(r['slot_attribution'][0]['matches_reference_slot'] for r in rows if byid[r['task_id']]['training_only']['kind']==kind),'useful':sum(r['task_correct'] for r in rows if byid[r['task_id']]['training_only']['kind']==kind),'accepted':sum(r['kernel_accepted'] for r in rows if byid[r['task_id']]['training_only']['kind']==kind)} for kind in ['read','explain','compare']}
        audit_summary[phase]={'operation':sum(r['slot_attribution'][0]['matches_reference_slot'] for r in rows),'useful':sum(r['task_correct'] for r in rows),'accepted':sum(r['kernel_accepted'] for r in rows),'kind':kind_counts,'downstream':{k:v for k,v in downstream.items() if k!='missing_reference_slots'},'missing_reference_slot_count':sum(len(r['missing_reference_slots']) for r in rows),'missing_reference_worlds':sum(bool(r['missing_reference_slots']) for r in rows),'failures':[{'task_id':r['task_id'],'status':r['status'],'reason':r['reason']} for r in rows if r['status']!='accepted'],'expected_later_slots':sum(len(e['training_only']['targets'])-1 for e in examples)}
        mismatches.extend({'phase':phase,'task_id':r['task_id'],'operation':r['slot_attribution'][0],'later':r['slot_attribution'][1:]} for r in rows if not r['slot_attribution'][0]['matches_reference_slot'])

    fit={}
    for arm,field in [('zero_update','fit_before'),('trained','fit_after')]:
        fit[arm]={}
        for split,n in [('development_train',24),('development_validation',12)]:
            f=R[field][split]; targets={r['row_id']:r['label'] for r in T[split]}
            assert len(f['rows'])==n and {r['row_id'] for r in f['rows']}==set(targets)
            results=[]
            for row in f['rows']:
                assert row['target']==targets[row['row_id']]
                correct=row['prediction']==targets[row['row_id']]
                compare(field+'/'+split+'/'+row['row_id']+'/correct',correct,row['correct'])
                results.append(correct)
            accuracy=sum(results)/n; ce=math.fsum(r['ce'] for r in f['rows'])/n
            compare(field+'/'+split+'/accuracy',accuracy,f['accuracy']);compare(field+'/'+split+'/n',n,f['n'])
            assert math.isclose(ce,f['ce'],rel_tol=1e-14,abs_tol=1e-14)
            fit[arm][split]={'correct':sum(results),'n':n,'accuracy':accuracy,'saved_ce_mean':ce}

    def ci(rows,base,metric):
        a={r['task_id']:r for r in rows};b={r['task_id']:r for r in base};assert set(a)==set(b)
        delta=[int(a[k][metric])-int(b[k][metric]) for k in sorted(a)]
        rng=random.Random(9203);draws=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(2000))
        return {'n_worlds':len(delta),'estimate':sum(delta)/len(delta),'ci95':[draws[50],draws[1950]],'unit':'independent generated world/task','method':'paired percentile bootstrap','seed':9203,'samples':2000}
    paired={};per_kind={}
    for baseline in ['stock','zero_update']:
        name='trained_minus_'+baseline
        paired[name]={m:ci(ALL['trained'],ALL[baseline],m) for m in ['task_correct','kernel_accepted']}
        per_kind[name]={kind:{m:ci([r for r in ALL['trained'] if byid[r['task_id']]['training_only']['kind']==kind],[r for r in ALL[baseline] if byid[r['task_id']]['training_only']['kind']==kind],m) for m in ['task_correct','kernel_accepted']} for kind in ['read','explain','compare']}
    compare('paired',paired,R['paired']);compare('paired_by_kind',per_kind,R['paired_by_kind'])

    direct_ids={e['task_id'] for e in examples if e['training_only']['kind']=='compare'};direct={}
    for arm in ['direct_before','direct_after']:
        rows=R[arm]; assert len(rows)==4 and {r['task_id'] for r in rows}==direct_ids
        correct=0; detail=[]
        for row in rows:
            ex=byid[row['task_id']];tr=ex['training_only'];nodes=tr['program']['nodes'];nodeby={n['id']:n for n in nodes}
            cmp=[n for n in nodes if n['op']=='Compare']; assert len(cmp)==1
            cmp=cmp[0]; records={r['record_ref']:r for r in tr['catalog']['records']}
            def value(ref):
                node=nodeby[ref['node']]
                assert node['op']=='ReadFact'
                return Decimal(str(records[node['record_ref']]['fields'][node['field']]['value']))
            left,right=value(cmp['left']),value(cmp['right'])
            numerical='equal' if left==right else ('greater_than' if left>right else 'less_than')
            assert numerical==cmp['relation']
            qid='slot-001-direct-comparison-relation';choice=row['answers'][qid]['choice'];p=row['served_probabilities'][qid]
            assert choice==max(sorted(p),key=p.get)
            got={'r001':'greater_than','r002':'less_than','r003':'equal'}[choice];is_correct=got==numerical
            compare(arm+'/'+row['task_id']+'/correct',is_correct,row['correct']);correct+=is_correct
            detail.append({'task_id':row['task_id'],'prediction':got,'target':numerical,'left':str(left),'right':str(right)})
        direct[arm]={'correct':correct,'n':4,'rows':detail}
    equal=R['direct_before']==R['direct_after'];compare('direct_equal',equal,R['direct_equal'])
    eligible=fit['trained']['development_train']['accuracy']>=.8 and paired['trained_minus_stock']['task_correct']['estimate']>0 and per_kind['trained_minus_stock']['read']['task_correct']['estimate']>=0 and per_kind['trained_minus_stock']['read']['kernel_accepted']['estimate']>=0
    compare('eligible_for_review',eligible,R['eligible_for_review'])
    assert 'torch' not in sys.modules
    names=['registration.json','operation-records.json','targets.json','raw.jsonl','stock.json','zero_update.json','trained.json','receipt.json']
    hashes={str(P/n):filehash(P/n) for n in names};hashes[str(C)]=filehash(C);hashes['corpus_semantic_sha256']=dig(D)
    source_hash_check={}
    for name in ['language_growth/planner.py','language_growth/kernel.py','language_growth/realize.py','language_growth/types.py','language_growth/free_running_dev.py','language_growth/parent_bridge.py','language_growth/operation_cache.py','language_growth/rwkv_operation_cache.py']:
        actual=filehash(root/name);assert actual==G['source_sha256'][name];source_hash_check[name]=actual

    # Verify the fixed training recipe from JSON alone; cache/checkpoint binaries are excluded.
    h=R['history']; orders=R['orders']; keys=R['lineage']['cache_keys']
    assert len(h)==96 and [v['step'] for v in h]==list(range(1,97))
    assert [v['epoch'] for v in h]==[i//12 for i in range(96)]
    assert len(keys)==24 and len(set(keys))==24 and len(orders)==8
    for epoch,order in enumerate(orders):
        idx=list(range(24)); random.Random(G['config']['seed']+epoch).shuffle(idx)
        assert order==[keys[i] for i in idx]
    assert R['lineage']['targets_sha256']==dig(T['development_train'])
    recipe={'steps':96,'complete_deterministic_passes':8,
            'targets_sha256':R['lineage']['targets_sha256'],'targets_lineage_verified':True}

    # Make discordant world counts explicit for all predeclared metrics/cohorts.
    discordances={}
    for baseline in ['stock','zero_update']:
        name='trained_minus_'+baseline
        a={r['task_id']:r for r in ALL['trained']}
        b={r['task_id']:r for r in ALL[baseline]}
        discordances[name]={}
        for cohort in ['all','read','explain','compare']:
            ids=sorted(k for k in a if cohort=='all' or byid[k]['training_only']['kind']==cohort)
            discordances[name][cohort]={}
            for metric in ['task_correct','kernel_accepted']:
                delta=[int(a[k][metric])-int(b[k][metric]) for k in ids]
                discordances[name][cohort][metric]={
                    'n_worlds':len(ids),'task_ids':ids,'paired_deltas':delta,
                    'baseline_count':sum(b[k][metric] for k in ids),
                    'trained_count':sum(a[k][metric] for k in ids),
                    'gains':sum(d==1 for d in delta),'losses':sum(d==-1 for d in delta),
                    'both_true':sum(a[k][metric] and b[k][metric] for k in ids),
                    'both_false':sum(not a[k][metric] and not b[k][metric] for k in ids)}

    # The corpus can rename/reorder choices. Compare descriptions, never raw IDs.
    alignment={}
    for phase,field in [('zero_update','fit_before'),('trained','fit_after')]:
        recs={r['id']+'/slot-001-operation':r for r in O['development_validation']}
        live={r['task_id']:r['slot_attribution'][0]['selected_description'] for r in ALL[phase]}
        alignment[phase]=[]
        for row in R[field]['development_validation']['rows']:
            rec=recs[row['row_id']]
            criteria=rec['request']['questions']['slot-001-operation']['criteria']
            description=criteria[sorted(criteria)[row['prediction']]]
            if description!=live[rec['id']]:
                alignment[phase].append({'task_id':rec['id'],'cached_description':description,
                                         'live_description':live[rec['id']]})

    expected_input_hashes={
        'registration.json':'6cb4c68de5f528f6dd033941ad7eee5dadf88d403fe181dbfd5f4243edf38d88',
        'operation-records.json':'9b32ee60ff3bd89d9de2c3995b17633da5badf463c00f22507a8195d1288af43',
        'targets.json':'220155b9a070d6569d991632ffe4e8630abdd9acad28e27989aa080e72061aa9',
        'raw.jsonl':'40fda1510a0d7f398104b9bb6290114fa6204d78069519c57faa77b2ae7ded38',
        'stock.json':'794665512e49afdc7ff944297341abbc8a7eac79dd9d6e9e32acba14e5d4f8e3',
        'zero_update.json':'32c58676f82aef733677da0c15279bd1195567a4796424d36bf52602793220a9',
        'trained.json':'6ad0afa7aa162df44deeb558344797eaa9252aed69200f4a4c3da50ae21c7ea2',
        'receipt.json':'b4f49bcb1cdfcb70ca497d4a97ee2ec3c93662a32db6753adf4d7996c6629e60'}
    input_hashes={name:filehash(P/name) for name in expected_input_hashes}
    assert input_hashes==expected_input_hashes
    corpus_file_sha=filehash(C)
    assert corpus_file_sha=='0d3be636b0b2d7b19823c4c42f083e8cddd5196348a8b5f110930e0e4752b65b'
    assert not diff, diff
    assert 'torch' not in sys.modules
    command=(
        '& "'+Path(sys.executable).resolve().as_posix()+'" -B "'+
        Path(__file__).resolve().as_posix()+'" --root "'+root.as_posix()+
        '" --out "'+out.as_posix()+'"')
    result={
        'schema':'ssdm.operation-only-independent-remeasurement.v1',
        'conclusion':{
            'strong_train_operation_fit':True,
            'strict_useful_answer_gain_vs_stock':False,
            'read_useful_answer_and_acceptance_nonregression':True,
            'direct_saved_outputs_unchanged_but_accuracy_zero_of_four':True,
            'eligible_for_review':False,
            'interpretation':'Train operation fit reaches 24/24, but dev useful answers remain 1/12 versus stock and acceptance falls from 12/12 to 11/12. The one correctly selected comparison reaches correct operands then fails on the stock greater_than relation. No adoption or qualification follows.'},
        'status':'pass','command':command,
        'script_sha256':filehash(Path(__file__)),
        'inputs':{
            'run_directory':P.as_posix(),'file_sha256':input_hashes,
            'frozen_corpus_path':C.as_posix(),'frozen_corpus_file_sha256':corpus_file_sha,
            'corpus_canonical_json_sha256':dig(D),
            'operation_records_canonical_json_sha256':{s:dig(v) for s,v in O.items()},
            'source_sha256_verified_against_registration':source_hash_check},
        'historical_ephemeral_audit':{
            'inline_body_sha256':'6e17ea08f885a7d06f0d25544d3881b8c1f2b7fecbaeb5df2c85b8fda89ad2ab',
            'command':'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -B -c "import sys,hashlib; code=sys.stdin.read(); print(\'AUDIT_SCRIPT_SHA256=\'+hashlib.sha256(code.encode()).hexdigest()); exec(compile(code,\'<operation-raw-audit>\',\'exec\'))"',
            'tool_execution_chunk':'6a7ad1'},
        'validation':{
            'complete_terminal':E[-1],'registration_file_hash_matches_receipt':True,
            'corpus_and_operation_record_pins_match':True,'targets_rederived_from_sorted_option_ids':True,
            'all_three_arms_replayed':True,'regenerated_requests_exactly_equal_recorded_requests':True,
            'all_recorded_calls_consumed_once':True,'torch_imported':False,
            'model_or_feature_or_checkpoint_files_opened':False,
            'original_receipt_and_inputs_unchanged':True},
        'raw_counts':{'events':len(E),'requests':len(reqs),'responses':len(responses),
                      'exceptions':len(exceptions),'requests_by_phase':dict(Counter(k[0] for k in reqs))},
        'saved_vs_recomputed_differences':diff,
        'recomputed':audit_summary,'replayed_observations':ALL,'cached_fit':fit,
        'cached_live_dev_prediction_description_differences':alignment,'training_recipe':recipe,
        'paired':paired,'paired_by_kind':per_kind,'discordances':discordances,
        'direct':direct,'direct_equal':equal,'eligible_for_review':eligible,
        'bootstrap':{
            'seed':9203,'samples':2000,'reset_rng_for_each_metric_and_cohort':True,
            'world_order':'sorted task_id','unit':'one independent generated world/task',
            'paired_delta':'int(trained_outcome_i) - int(baseline_outcome_i)',
            'estimate':'sum(paired_deltas) / n_worlds',
            'draw':'sum(rng.choice(paired_deltas) for _ in paired_deltas) / n_worlds',
            'rng':'Python random.Random(9203)',
            'ci95':'sort(2000 bootstrap means)[50], sort(2000 bootstrap means)[1950]',
            'n_all':12,'n_per_kind':4,'method':'paired percentile bootstrap'},
        'scope_limits':[
            'Reused synthetic development diagnostics only; no Index, heldout, qualification, or adoption evidence.',
            'Every arm retains all 12 worlds; each predeclared kind retains all four worlds.',
            'Direct retention is exact saved status/answer/full-precision probability equality over only four comparison cases, not all 12 worlds.',
            'Direct requests/responses are absent from raw.jsonl; direct request identity and backbone immutability cannot be independently verified from these artifacts.',
            'All four direct comparison references are less_than; predictions are greater_than; bookend accuracy is 0/4 before and after.',
            'Cached-fit argmax accuracy is independently recomputed from saved integer predictions and separate labels; full score vectors are absent.',
            'CE values are saved per-row CE aggregated again, not independently rederived from raw score vectors.',
            'Reference labels must be resolved by descriptions because frozen augmented option IDs can differ from live production IDs.',
            'Reference-slot agreement is stricter than useful meaning: read tasks may accept a correct explanation; comparisons may reverse operands with inverse relation.',
            'Observed downstream calls include wrong-operation nonreference branches; 20 expected later reference slots per arm provide the coverage denominator.',
            'This replays recorded decisions through pinned semantic code; it does not reproduce model inference or independently verify feature/checkpoint binaries.']}
    # Recheck all input bytes immediately before writing only the dedicated audit result.
    assert {name:filehash(P/name) for name in expected_input_hashes}==input_hashes
    assert filehash(C)==corpus_file_sha
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    assert {name:filehash(P/name) for name in expected_input_hashes}==input_hashes
    print(json.dumps({'status':'pass','differences':len(diff),'script':str(Path(__file__).resolve()),
                      'script_sha256':result['script_sha256'],'result':str(out),
                      'result_sha256':filehash(out),'raw_sha256':input_hashes['raw.jsonl'],
                      'receipt_sha256':input_hashes['receipt.json'],'command':command,
                      'torch_imported':'torch' in sys.modules},indent=2))



if __name__ == "__main__":
    main()

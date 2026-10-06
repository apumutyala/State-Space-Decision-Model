# Independent frozen validation remeasurement

Reviewer: `/root/baseline_receipt_recompute`, distinct from implementation.
Audited source: `d5663891e26defaa2e450511cf1a55437b3cd1a5`, clean at audit.
Verdict: verified within the declared frozen engineered-feature development
scope. No edits, training rerun, GPU, held-out/Index access or external access.

Each seed independently yields 200/300 exact reference programs and 300/300
accepted outputs: read 100/100, explain 100/100, compare 0/100. Every comparison
request selects `Assert one selected fact`. All 600 trained rollout calls use
the model's own previous selections; zero repair. Teacher-forced diagnostic
slots are operation 200/300, fact 200/200, left/right/relation 100/100; no-sign
relation 43/100. These diagnostic slots do not rescue free-running task accuracy.

Initial models solve 0/300 each. Initial accepted counts are 300, 100, 159 for
seeds 17/29/43; provider-rejected counts are 0, 200, 141. The implementation
explicitly retains unsupported-feature outcomes and all task denominators.
All checkpoint bytes match the original experiment. Evaluation regeneration
matches exactly; names, ranges and templates are disjoint from fitting.

The extended independent stdin audit reconstructed contributions, logits,
initial seeded weights and expected task meaning from requests/evidence.
Maximum logit discrepancy: 1.3877787807814457e-17; raw-probability discrepancy:
5.551115123125783e-17. It checked kernel/rendering and rollout prefixes, with
no invalid labels or reference leakage. Gold prefixes exist only in the
disclosed teacher-forced slot diagnostics.

Safe conclusion: frozen engineered-feature linear rankers score 200/300 on
one shared iterative synthetic development-validation set, with a wording
failure at operation selection. The kernel accepts sound factual answers to
the wrong requested task; it does not verify relevance. No untouched held-out,
raw-language/numerical learning, retention, causal mediation or promotion claim.

| Artifact | Independently verified SHA-256 |
| --- | --- |
| receipt.json | `51b6b17728d594377ea7228f711ffb093743d27568ab8eb43a282108c3df33d7` |
| evaluation_tasks.jsonl | `d34df20c1448998791071833cc9154589b526e2ce0a68e7f98e307028d0a827f` |
| predictions-17.json | `309ae0d96d78b85dd19fde32d93f26add11f2488233bd49c415e99b3832455ac` |
| predictions-29.json | `69c3c3bc1f88fc41cb762c2df3d8939a6ef01698fc02e7b373c3a6a68c1c1627` |
| predictions-43.json | `83c25d5050febc3c48575a5e169bc314014607309067510cf0e6d8fc4811faeb` |
| model-17.json | `79a48dd4727208ac75470bc085e73b552ed7d89b71f22c8f8e9efca5905fcd3e` |
| model-29.json | `418b674bcb229a19926475efed32a1874f3e785ee80488c7aa41e71c65da450f` |
| model-43.json | `080c2b7e63ea2d5cecf1c4dbca163df0997ffb9e63afb3c7256d71ff88600b04` |
| training_records.jsonl | `eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c` |

The auditor supplied this compact replay command. Its extended independent
feature audit was a separate stdin execution. Run from
`C:\Users\apumu\research\CHLabs\ssdm-codex-l0-transparent-cpu-baseline`:

```powershell
@'
import json,pathlib,hashlib,collections
from language_growth.kernel import check_program
b=pathlib.Path(r'C:\Users\apumu\research\CHLabs\ssdm-data')
v=b/'l0-transparent-cpu-validation-20261006'
old=b/'l0-transparent-cpu-20261006'
tasks={t['task_id']:t for t in map(json.loads,(v/'evaluation_tasks.jsonl').read_text().splitlines())}
gold={}
for tid,t in tasks.items():
    for req,target in zip(t['model_inputs'],t['training_only']['targets']):
        gold[tid,next(iter(req['questions']))]=target['option_id']
for seed in (17,29,43):
    assert (v/f'model-{seed}.json').read_bytes()==(old/f'model-{seed}.json').read_bytes()
    raw=json.loads((v/f'predictions-{seed}.json').read_text())
    for arm in ('slot_predictions','initial_slot_predictions','without_numeric_sign_slot_predictions'):
        counts=collections.defaultdict(lambda:[0,0])
        for row in raw[arm]:
            selected=max(sorted(row['logits']),key=row['logits'].get)
            counts[row['slot']][0]+=selected==gold[row['task_id'],row['qid']]
            counts[row['slot']][1]+=1
        print(seed,arm,dict(counts))
    for arm in ('report','initial_report'):
        rows=raw[arm]['observations'];assert len(rows)==len(tasks)==300
        exact=sum(r['program']==tasks[r['task_id']]['training_only']['program'] for r in rows)
        accepted=0
        for r in rows:
            t=tasks[r['task_id']]['training_only']
            if r['program']:
                accepted+=check_program(r['program'],t['host'],t['catalog']).status=='accepted'
        print(seed,arm,'exact',exact,'accepted',accepted,'statuses',dict(collections.Counter(r['status'] for r in rows)))
for p in sorted(v.glob('*')):
    if p.is_file():print(p.name,hashlib.sha256(p.read_bytes()).hexdigest())
'@ | & 'C:\Users\apumu\AppData\Roaming\uv\tools\awscli\Scripts\python.exe' -B -
```

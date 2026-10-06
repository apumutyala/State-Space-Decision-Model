# Independent re-derivation of the original same-domain baseline

Reviewer: fresh-context subagent `/root/baseline_receipt_recompute`, distinct
from the implementation author. Verdict relayed by the coordinator: confirmed
within the declared engineered-feature, same-template development scope.
Audit source `d2963fe`; receipt-preservation HEAD `b989d11`; the auditor checked
`git diff d2963fe HEAD -- language_growth` was empty.

The independent command used inline PowerShell here-strings piped to
`& 'C:\Users\apumu\AppData\Roaming\uv\tools\awscli\Scripts\python.exe' -B -`.
The inline audit script was not supplied as a standalone artifact. The auditor
rederived checkpoint/raw-request logits without importing this baseline's
feature/scoring/task helpers. Maximum reported logit discrepancy was 8.88e-16,
raw-probability discrepancy 2.22e-16. No training rerun, GPU, spend or sealed
held-out access occurred.

Recomputed: each seed 17/29/43 improves 0/300 to 300/300 task-correct; all 800
fitted slots are correct; numeric-sign inference ablation gives 40/100 relation
accuracy. No overlap was found in task IDs, complete query+world identities or
structured worlds. All evaluation wordings occur in the fitting templates;
this does not demonstrate transfer to new wording. Reference metadata did not
leak into model requests. The later development-validation measurement is
separate and is not covered by this verdict.

Verified SHA-256 evidence:

- Receipt: `74ebc3eb4471ec4751863c3b270c632db13f9579228eabd90b2ea02b0ca4f7c6`
- Evaluation tasks: `a409c652b26cbbccb46974ebd8c4564c6e1026cfb2ae4e01811b5cd018ebde4a`
- Training records: `eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c`
- Model 17: `79a48dd4727208ac75470bc085e73b552ed7d89b71f22c8f8e9efca5905fcd3e`
- Model 29: `418b674bcb229a19926475efed32a1874f3e785ee80488c7aa41e71c65da450f`
- Model 43: `080c2b7e63ea2d5cecf1c4dbca163df0997ffb9e63afb3c7256d71ff88600b04`
- Predictions 17: `ea383bb945e2dd9ff7381a7f4c2c5f4ca072f43b893da519b8766e350d119c37`
- Predictions 29: `d9441190b6f9914471bab71b62ceea29c164a0144b92fe645bb4623bbfc03856`
- Predictions 43: `ab9e44e061f288a3cb47fee43564339e842c8bf90335141791749cb2d87be4f7`

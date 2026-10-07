# Issue #36 margin review sensitivity receipt

This is a CPU-only Gaussian sensitivity calculation used to assess promotion-gate operating characteristics. It is not a model evaluation, bootstrap reproduction, or measured power study. The script uses a seeded normal generator, known standard errors, and two-sided normal 95% intervals. It uses no evaluation rows or model outputs.

Reproduce from this directory with Python 3.12.13 and the standard library:

```powershell
python simulate.py | Out-File -Encoding utf8 simulation.json
```

Configuration: 200,000 draws; seed 20261006; billed cost $0; CPU only. Standard errors are specified in `simulate.py`: measured Index subset SE 0.752, projected full-DEV SE 0.31, retention-accuracy SE from assumed discordance 0.10 and n=1,420, and retention-ECE half-width 0.026 assumed from the independent review. Projected/assumed values are explicitly labelled in the JSON keys and must not be presented as measured operating characteristics.

The alarm example is a deterministic counterexample showing that a paired change test alone does not enforce an absolute false-alarm cap. No dataset, model, training, or GPU is used.

Source SHA256: `c217afa09a33ca575b959b63c10d24ec96211bb7a7db22841e7a36be16b2ed8e`.
Original PowerShell output SHA256: `6080e711218149be41424b0d9a52a6fdcdcaf156e6c380491ef6227a155ee46d`. The committed LF-normalized `simulation.json` SHA256 is `af71dec3c7c51434fb36bfd17890c3ea1ab93dec40a3f43eb427a16f96cd4a7b`; rerunning in this worktree reproduced that committed hash.

Committed evidence used for the Index SE / area example is `receipts/2026-10-06-D-index03-rescore/rescore.json` at the Track D review's referenced commit. Its 95% area interval is not sufficient for the proposed 99% Bonferroni gate; recompute before use.


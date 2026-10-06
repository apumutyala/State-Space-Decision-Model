"""Independent re-derivation from the written records (not the build's in-memory state):
(1) rebuild with the same seed into a second directory and compare the record-file sha256 (determinism);
(2) re-read records.jsonl.gz, re-validate every record, re-run the held-out template guard, recompute the
manifest hashes, and re-run the whole-payload dedup check (must equal kept + kept_with_context_overlap split)."""
import gzip, hashlib, json, sys
from pathlib import Path
from ssdm_data import augment as A, build as B, config, dedup as D, injection as I
from ssdm_data.text import canonical_json

rd = Path(sys.argv[1])
summ = json.loads((rd / "build-summary.json").read_text(encoding="utf-8"))
log = json.loads((rd / "dedup-log.json").read_text(encoding="utf-8"))
fp = D.Fingerprint.load(config.FINGERPRINT)
out = {}
res2 = B.build_sample(n_target=summ["n_target"], seed=summ["seed"], out=str(config.BUILD_DIR / "recheck-rebuild"),
                      fingerprint=fp, log=lambda *a: None)
out["rebuild_sha256_equal"] = res2["summary"]["records_uncompressed_sha256"] == summ["records_uncompressed_sha256"]
man = {}
with gzip.open(config.MANIFEST_DIR / f"sample-{summ['n_target']}-seed{summ['seed']}.manifest.jsonl.gz", "rt",
               encoding="utf-8") as f:
    for line in f:
        m = json.loads(line); man[m["id"]] = m["sha256"]
n = ctx = mism = 0
with gzip.open(summ["records_path"], "rt", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n"); rec = json.loads(line); n += 1
        A.validate(rec); I.assert_no_heldout(rec["state"], rec["questions"], rec["id"])
        mism += man.get(rec["id"]) != hashlib.sha256(line.encode()).hexdigest()
        ctx += any(fp.check(rec["state"], rec["questions"]).values())
out.update(records=n, manifest_mismatches=mism, whole_payload_overlap_recomputed=ctx,
           whole_payload_overlap_logged=log["totals"].get("kept_with_context_overlap", 0))
out["ok"] = out["rebuild_sha256_equal"] and mism == 0 and ctx == out["whole_payload_overlap_logged"]
print(json.dumps(out, indent=1))

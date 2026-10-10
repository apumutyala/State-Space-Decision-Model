"""Answer-independent split of DEV (edition 0.2.1) for B-PREREG-GUAR-FULLDEV. Streams the materialised DEV rows,
checks run_id order == the committed dev manifest (the frozen loader's check), and prints the id-set shas."""
import gzip, hashlib, json, sys
sys.path.insert(0, sys.argv[1])  # read-only agent-main checkout
from harness import slicing, config
from jevkit.guarantees import split_calibration, Scored, ids_sha256

SALT = "B-PREREG-GUAR-FULLDEV|v1"
def u(key): return int(hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)

def stream_rows():
    with gzip.open(config.SLICE_ROWS_DIR / "dev-rows.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":
    man = [m["run_id"] for m in slicing.read_manifest(config.MANIFEST_DIR / "dev-manifest.jsonl.gz")]
    ids, cal_rows, test_rows, cal_dec, test_dec = [], [], [], [], []
    for r in stream_rows():
        rid = r["_evaluation"]["run_id"]; ids.append(rid)
        side = cal_rows if u(rid) < 0.5 else test_rows
        side.append(rid)
        for k in r["questions"]:
            (cal_dec if side is cal_rows else test_dec).append(f"{rid}#{k}")
    assert ids == man, "materialised DEV rows do not match the manifest"
    fit, ltt, rec = split_calibration([Scored(i, "x", {}) for i in cal_dec], seed=0, fit_frac=0.5)
    print(json.dumps({"n_rows": len(ids), "manifest_sha256_ids": ids_sha256(ids),
                      "cal_rows": len(cal_rows), "cal_run_ids_sha256": ids_sha256(cal_rows),
                      "test_rows": len(test_rows), "test_run_ids_sha256": ids_sha256(test_rows),
                      "cal_decisions": len(cal_dec), "cal_decision_ids_sha256": ids_sha256(cal_dec),
                      "test_decisions": len(test_dec), "test_decision_ids_sha256": ids_sha256(test_dec),
                      "fit_ltt": rec}, indent=1))

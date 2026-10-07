"""Edition 0.3 GSM8K rows: rebuild, and the sealed-TEST-safe DEV/TEST slice addenda.

Rebuild ($0, CPU): kit PR #74 head 62d2f51, `decision_index.suite.build.gsm8k_v3.build(test_parquet, selected_rows)`
over the pinned GSM8K test parquet (openai/gsm8k @740312a, sha256 ee7b8da9...) and our byte-identical 0.2 rows.
Output `gsm8k-rows.jsonl` is byte-identical to the board's (sha256 75933489..., pinned in config).

Slice amendment (`build_addenda`). Every rebuilt row keeps the 0.2 row's id, group_id, track and state, and its
run_id is `v0.3-gsm8k:` + the 0.2 run_id (`_evaluation.source_run_id`). So the 0.3 GSM8K slice membership is a
pure function of the committed manifests' IDs: DEV gets the rebuilt twins of DEV's GSM8K rows, TEST those of
TEST's. No new sampling, no case reassignment, no TEST row is ever selected by content. The 0.2.1 manifests are
not modified; the addenda are new files `{dev,test}-manifest-0.3-gsm8k.jsonl.gz`.

Seal: the TEST addendum manifest stores IDs and hashes only, computed from the full rebuilt suite file (the same
way `harness slice` hashed every TEST row when the slices were cut). TEST rows are never written to disk,
printed, scored or materialised; only counts and hashes are reported. Loading them for scoring still goes through
`cli.load_slice("test", ...)`, which requires SSDM_UNSEAL_TEST=1 + SSDM_PREREG and logs the unseal.
"""

import collections
import json

from harness import cases, config, slicing
from harness.suite import dump_json, read_rows, write_rows

PREFIX = config.GSM8K_03_RUN_ID_PREFIX
DEV_ROWS = config.SLICE_ROWS_DIR / "dev-rows-0.3-gsm8k.jsonl.gz"


def manifest_path(name):
    return config.MANIFEST_DIR / f"{name}-manifest-0.3-gsm8k.jsonl.gz"


def load_rows(path=None):
    """All 2,638 rebuilt rows, refusing a file whose uncompressed hash is not the pinned one."""
    from decision_index.suite.io import sha256_file

    path = path or config.GSM8K_03_ROWS
    got = sha256_file(path, gunzip=True)
    if got != config.GSM8K_03_SHA256:
        raise RuntimeError(f"{path}: uncompressed sha256 {got} != pinned {config.GSM8K_03_SHA256}")
    return read_rows(path)


def twin_ids(manifest):
    """0.2.1 manifest entries -> the run_ids of their 0.3 GSM8K twins (IDs only)."""
    return [PREFIX + m["run_id"] for m in manifest if m["catalog_id"] == 30]


def _select(rows, manifest):
    by = {r["_evaluation"]["run_id"]: r for r in rows}
    out = []
    for m in manifest:
        if m["catalog_id"] != 30:
            continue
        r = by.get(PREFIX + m["run_id"])
        if r is None:
            raise ValueError(f"no rebuilt twin for {m['run_id']}")
        e = r["_evaluation"]
        if e["source_run_id"] != m["run_id"] or str(e["group_id"]) != m["group_id"] or e["track"] != m["track"]:
            raise ValueError(f"rebuilt twin of {m['run_id']} changed id/group/track")
        out.append(r)
    return sorted(out, key=lambda r: r["_evaluation"]["run_id"])


def build_addenda(rows_path=None):
    rows = load_rows(rows_path)
    old = {n: slicing.read_manifest(config.MANIFEST_DIR / f"{n}-manifest.jsonl.gz") for n in ("dev", "test")}
    sel = {n: _select(rows, old[n]) for n in old}
    out = {"edition": "0.3", "kit_pr": config.KIT_03_PR, "kit_commit": config.KIT_03_COMMIT, "gsm8k_rows_sha256": config.GSM8K_03_SHA256,
           "rule": "membership = rebuilt twins (run_id prefix v0.3-gsm8k:) of each 0.2.1 slice's GSM8K rows; 0.2.1 manifests unchanged",
           "rebuilt_rows": len(rows)}
    for n in sel:
        out[n] = {"manifest": manifest_path(n).name, "manifest_sha256": slicing.write_manifest(manifest_path(n), sel[n]), "rows": len(sel[n]),
                  "old_gsm8k_rows": sum(m["catalog_id"] == 30 for m in old[n]),
                  "by_track": dict(sorted(collections.Counter(r["_evaluation"]["track"] for r in sel[n]).items()))}
    ids = {n: {r["_evaluation"]["run_id"] for r in sel[n]} for n in sel}
    out["disjoint_rows"] = not (ids["dev"] & ids["test"])
    out["source_overlap_dev_test"] = cases.overlap_report(sel["dev"], sel["test"])
    # 0.3 options are other test problems' golds: an answer group (10 problems sharing a value set) can have
    # members on both sides. Only numbers are shared, never a question; reported, not a disjointness violation.
    grp = {n: {r["metadata"]["answer_group"]["id"] for r in sel[n]} for n in sel}
    out["answer_groups"] = {"dev": len(grp["dev"]), "test": len(grp["test"]), "shared": len(grp["dev"] & grp["test"])}
    dump_json(config.MANIFEST_DIR / "slices-0.3-gsm8k.json", out)
    config.SLICE_ROWS_DIR.mkdir(parents=True, exist_ok=True)
    write_rows(DEV_ROWS, sel["dev"])  # DEV only; TEST rows are never materialised
    return out


def addendum_rows(name):
    """Rebuilt GSM8K rows of a slice, hash-checked against its addendum manifest. Callers gate TEST."""
    manifest = slicing.read_manifest(manifest_path(name))
    if name == "dev" and DEV_ROWS.exists():
        rows = read_rows(DEV_ROWS)
    else:
        rows = load_rows()
    return slicing.select(rows, manifest)


def map_ids(run_ids):
    """Extend an ID list (e.g. the m1 DEV subset) with the 0.3 twins of its GSM8K rows."""
    ids = list(run_ids)
    return ids + [PREFIX + i for i in ids if i.startswith("30:")]


if __name__ == "__main__":
    print(json.dumps(build_addenda(), indent=1))

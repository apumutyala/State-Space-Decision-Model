"""Verify the combined jev2 pointer and the jev1+jev2 mix manifest against the git blobs and the records on disk.

    python -X utf8 scripts/regate_mix_check.py --ref 282a4867 --out mix-check.json
"""
import argparse, collections, gzip, hashlib, json, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import regate_jev2a as RG
from train.cohort import record_content_hash

BUILD = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\m2\build")


def blob(ref, path):
    b = subprocess.run(["git", "cat-file", "blob", f"{ref}:{path}"], capture_output=True, cwd=REPO).stdout
    return b, hashlib.sha256(b).hexdigest()


def scan(build, skip_fams=()):
    p = BUILD / build / "records.jsonl.gz"
    h = hashlib.sha256(); n = 0; fams = collections.Counter(); rows = []
    with gzip.open(p, "rb") as fh:
        for line in fh:
            h.update(line)
            if not line.strip():
                continue
            r = json.loads(line); n += 1
            f = r["aug"]["jev1"]["family8"]; fams[f] += 1
            if f in skip_fams:
                continue
            full = " | ".join(RG.norm_leaf(x) for x in RG.row_leaves(r["state"], r["questions"]))
            rows.append((r["id"], record_content_hash(r), hashlib.sha256(full.encode()).hexdigest()[:20]))
    return {"records_uncompressed_sha256": h.hexdigest(), "n_records": n, "families": dict(fams)}, rows


def dups(groups):
    """groups: {build: rows}. Cross-build duplicates by id, content hash, normalised text."""
    out = {}
    for k, name in ((0, "id"), (1, "content_hash"), (2, "normalised_text")):
        owner = {}; cross = collections.Counter(); within = 0; ex = []
        for b, rows in groups.items():
            seen_b = set()
            for row in rows:
                v = row[k]
                if v in seen_b:
                    within += 1
                seen_b.add(v)
                if v in owner and owner[v] != b:
                    cross[f"{owner[v]}|{b}"] += 1
                    if len(ex) < 10:
                        ex.append(row[0])
                owner.setdefault(v, b)
        out[name] = {"cross_build": dict(cross), "cross_build_total": sum(cross.values()), "within_build": within,
                     "examples": ex}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"ref": a.ref}
    cb, cs = blob(a.ref, "data/manifests/bschema-draw-jev2.pointer.json"); comb = json.loads(cb)
    mb, ms = blob(a.ref, "data/manifests/bschema-mix-jev1-jev2.json"); mix = json.loads(mb)
    out["combined_pointer_sha256"] = cs; out["mix_manifest_sha256"] = ms
    disk, rows = {}, {}
    for b in ("bschema-draw-jev1", "bschema-draw-jev2a-v2", "bschema-draw-jev2b", "bschema-draw-jev2c-v2",
              "bschema-draw-jev2a", "bschema-draw-jev2c"):
        disk[b], rows[b] = scan(b, skip_fams=("long_policy",) if b == "bschema-draw-jev1" else ())
        print(b, disk[b]["n_records"], flush=True)
    out["disk"] = disk

    def check_components(comps):
        res = []
        for c in comps:
            pb, psha = blob(a.ref, c["pointer"]); ptr = json.loads(pb)
            d = disk[c["build"]]
            fam_claim = {k: v for k, v in c["families"].items()}
            res.append({"build": c["build"], "pointer_sha256_claimed": c["pointer_sha256"], "pointer_sha256_git": psha,
                        "pointer_ok": psha == c["pointer_sha256"],
                        "records_ok_pointer": ptr["records_uncompressed_sha256"] == c["records_uncompressed_sha256"],
                        "records_ok_disk": d["records_uncompressed_sha256"] == c["records_uncompressed_sha256"],
                        "n_ok": d["n_records"] == c["n_records"],
                        "families_ok": all(d["families"].get(k.replace(" (excluded: replaced)", "")) == v
                                           for k, v in fam_claim.items())})
        return res

    out["combined_components"] = check_components(comb["components"])
    out["combined_builds"] = [c["build"] for c in comb["components"]]
    out["combined_n_ok"] = comb["n_records"] == sum(disk[b]["n_records"] for b in out["combined_builds"])
    sup = comb.get("superseded_do_not_mix", {})
    out["superseded_listed"] = {b: {"records_sha256_matches_disk": v.get("records_uncompressed_sha256") ==
                                    disk[b]["records_uncompressed_sha256"]} for b, v in sup.items() if b in disk}
    out["superseded_not_a_component"] = not (set(sup) & set(out["combined_builds"]))
    out["combined_dups"] = dups({b: rows[b] for b in out["combined_builds"]})
    out["combined_vs_superseded_overlap"] = dups({**{b: rows[b] for b in out["combined_builds"]},
                                                  "bschema-draw-jev2a": rows["bschema-draw-jev2a"],
                                                  "bschema-draw-jev2c": rows["bschema-draw-jev2c"]})
    out["mix_components"] = check_components(mix["components"])
    mb_builds = [c["build"] for c in mix["components"]]
    out["mix_builds"] = mb_builds
    out["mix_jev1_is_gated"] = any(c["build"] == "bschema-draw-jev1" and
                                   c["pointer_sha256"].startswith("582ffe1c") and
                                   c["records_uncompressed_sha256"].startswith("45735109") for c in mix["components"])
    out["mix_superseded_absent"] = not ({"bschema-draw-jev2a", "bschema-draw-jev2c"} & set(mb_builds))
    n_mix = sum(len(rows[b]) for b in mb_builds)
    out["mix_rows_counted"] = n_mix; out["mix_n_ok"] = n_mix == mix["n_records"]
    out["mix_families"] = {f: v["rows"] for f, v in mix["families"].items()}
    out["mix_n_families"] = len(mix["families"])
    out["mix_long_policy_owner"] = mix["families"].get("long_policy", {}).get("draw")
    out["mix_dups"] = dups({b: rows[b] for b in mb_builds})
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("disk",)})[:6000])


if __name__ == "__main__":
    main()

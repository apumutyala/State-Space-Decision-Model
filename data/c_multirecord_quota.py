"""Exact C-only quota from the verified union; preserve K/style coverage and bytes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from .c_multirecord_refresh import NAMES, V2, _triples
from .c_multirecord_subset import _pinned_json, _rank, _sha, stratum
from .c_multirecord_synth import _sha_file

ROOT = Path(__file__).resolve().parents[1]


def select_quota(rows, count, seed):
    """Reserve one per K/style; allocate remaining capacity proportionally.

    Within each K/style, round-robin available gold positions in seeded order.
    No promise of all gold positions when a group's quota is smaller than K.
    """
    if type(count) is not int or type(seed) is not int or seed < 0:
        raise ValueError("integer count and nonnegative integer seed required")
    groups = defaultdict(lambda: defaultdict(list))
    for rid, bucket in rows:
        groups[bucket[:2]][bucket[2]].append(rid)
    sizes = {b: sum(map(len, positions.values())) for b, positions in groups.items()}
    if not sizes or not len(sizes) <= count <= sum(sizes.values()):
        raise ValueError("quota cannot cover every available K/style group")
    if len({rid for rid, _ in rows}) != len(rows):
        raise ValueError("duplicate record IDs")
    remaining = count - len(sizes)
    capacity = sum(sizes.values()) - len(sizes)
    quotas = {b: 1 + (remaining * (n - 1) // capacity if capacity else 0) for b, n in sizes.items()}
    if capacity:
        order = sorted(sizes, key=lambda b: (-(remaining * (sizes[b] - 1) % capacity), _rank(seed, "quota", b), b))
        for b in order[:count - sum(quotas.values())]:
            quotas[b] += 1
    selected = []
    for bucket in sorted(groups):
        positions = groups[bucket]
        queues = {p: deque(sorted(ids, key=lambda rid: (_rank(seed, "record", rid), rid)))
                  for p, ids in positions.items()}
        active = deque(sorted(positions, key=lambda p: (_rank(seed, "position", (*bucket, p)), p)))
        for _ in range(quotas[bucket]):
            p = active.popleft()
            selected.append(queues[p].popleft())
            if queues[p]:
                active.append(p)
    selected.sort(key=lambda rid: (_rank(seed, "output", rid), rid))
    return selected, quotas


def export_quota(source, source_pointer, source_pointer_sha256, verification,
                 verification_sha256, out, *, count=7000, seed=20261009, data_root=None):
    from .config import DATA_ROOT

    source, out = Path(source), Path(out)
    data_root = Path(data_root or DATA_ROOT).resolve()
    relative = out.resolve().relative_to(data_root)
    if out.exists():
        raise FileExistsError("refusing existing quota output")
    pointer = _pinned_json(source_pointer, source_pointer_sha256)
    proof = _pinned_json(verification, verification_sha256)
    if (proof.get("schema") != "ssdm.c_multirecord_union_independent_verification.v1"
            or proof.get("status") != "verified" or proof.get("pointer_sha256") != source_pointer_sha256
            or proof.get("b_inputs_read") is not False or proof.get("protected_rows_read") is not False
            or proof.get("refused") != 0 or proof.get("check_record_passed") != pointer["n_records"]
            or proof.get("counts") != {name: pointer["n_records"] for name in NAMES}
            or proof.get("uncompressed_sha256") != {name: pointer[name + "_uncompressed_sha256"] for name in NAMES}):
        raise ValueError("requires the pinned independent verified C-only union")
    bounds = proof["token_bounds_verified"]
    if (bounds.get("layout") != "g1v2" or set(bounds["max_effective_tokens"]) != {"v2f", "v3w"}
            or any(type(n) is not int or not 1 <= n <= 2048 for n in bounds["max_effective_tokens"].values())):
        raise ValueError("verified complete-record token bounds missing")
    # One selection pass over already verified bytes; no re-tokenization or semantic re-audit.
    payload, buckets, worlds, contents = {}, {}, set(), set()
    bytes_retained = 0
    prefix_k, prefix_groups = set(), set()
    for record, content, raw in _triples(source, pointer):
        rid, world = record["id"], record["source_id"]
        if (record["source"] != V2 or record["source_split"] != "synthetic"
                or len(record["questions"]) != 7 or set(record["questions"]) != set(record["labels"])
                or rid in payload or world in worlds or content in contents):
            raise ValueError("source C identity/whole-record uniqueness differs from proof")
        bucket = stratum(record)
        if len(payload) < count:
            prefix_k.add(bucket[0]); prefix_groups.add(bucket[:2])
        payload[rid], buckets[rid] = raw, bucket
        worlds.add(world); contents.add(content)
        bytes_retained += sum(map(len, raw))
        if bytes_retained > 80 * 1024 * 1024:
            raise MemoryError("quota source payload exceeds 80 MiB")
    if sorted({b[0] for b in buckets.values()}) != proof["all_k"]:
        raise ValueError("source K coverage differs from proof")
    selected, quotas = select_quota(list(buckets.items()), count, seed)
    if _sha_file(Path(source_pointer)) != source_pointer_sha256 or _sha_file(Path(verification)) != verification_sha256:
        raise ValueError("source metadata changed during selection")
    out.mkdir(parents=True, exist_ok=False)
    train_dir = out / "train_set"
    train_dir.mkdir()
    names = {"records": "records.jsonl.gz", "manifest": out.name + ".manifest.jsonl.gz",
             "content_map": out.name + ".content.jsonl.gz"}
    hashes = {name: hashlib.sha256() for name in NAMES}
    with ExitStack() as stack:
        streams = {name: stack.enter_context(gzip.GzipFile(filename="", mode="wb", mtime=0,
                   fileobj=stack.enter_context((train_dir / names[name]).open("xb")))) for name in NAMES}
        for rid in selected:
            for name, raw in zip(NAMES, payload[rid]):
                streams[name].write(raw); hashes[name].update(raw)
    result_pointer = {"build": out.name + " (python -m data.c_multirecord_quota)", "n_records": count,
                      "location": (Path(data_root.name) / relative / "train_set").as_posix() + "/",
                      "manifest": names["manifest"], "content_map": names["content_map"],
                      "builder": {"module": "data.c_multirecord_quota", "commit": subprocess.check_output(
                          ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
                      "receipt": f"receipts/2026-10-09-C-multirecord-quota/{out.name}/summary.json",
                      "excludes_cohort": "verified C-only synthetic union subset; consumer must use clean B v2d and retain guards",
                      **{name + "_uncompressed_sha256": h.hexdigest() for name, h in hashes.items()}}
    pointer_raw = json.dumps(result_pointer, indent=1).encode() + b"\n"
    (train_dir / "pointer.json").write_bytes(pointer_raw)
    per_k = Counter(buckets[rid][0] for rid in selected)
    report = {"schema": "ssdm.c_multirecord_quota.v1", "records": count, "questions": 7 * count,
              "source_directory": str(source.resolve()), "source_pointer_sha256": source_pointer_sha256,
              "source_verification_sha256": verification_sha256, "source_records": pointer["n_records"],
              "source_uncompressed_sha256": proof["uncompressed_sha256"], "seed": seed,
              "selection": "one per available K/style then capacity-proportional largest remainder; seeded gold-position round robin within groups",
              "selected_ids": selected, "per_k_records": dict(sorted(per_k.items())),
              "group_quotas": [{"k": b[0], "style": b[1], "records": n} for b, n in sorted(quotas.items())],
              "source_groups": len(set(b[:2] for b in buckets.values())), "selected_groups": len(quotas),
              "prefix_baseline": {"records": count, "k": sorted(prefix_k), "groups": len(prefix_groups)},
              "selected_gold_strata": len({buckets[rid] for rid in selected}),
              "record_and_sidecar_bytes_preserved": True, "token_bounds": bounds,
              "bound_scope": "inherited source upper bounds, not remeasured subset maxima",
              "consumer_contract": f"REPLACE old C input in fresh clean B v2d build; quota=limit={count}; consume all selected IDs exactly once and fail on any guard/admission drop; never append overlapping C pools",
              "b_inputs_read": False, "protected_rows_read": False, "retokenized": False,
              "index_gain_measured": False, "payload_bytes": bytes_retained,
              "pointer": result_pointer, "pointer_sha256": _sha(pointer_raw),
              "compressed_sha256": {name: _sha_file(train_dir / names[name]) for name in NAMES}}
    (out / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-pointer", "source-pointer-sha256", "verification", "verification-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--count", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = vars(parser.parse_args())
    result = export_quota(**args)
    print(json.dumps({k: result[k] for k in ("records", "questions", "selected_groups", "selected_gold_strata", "pointer_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()

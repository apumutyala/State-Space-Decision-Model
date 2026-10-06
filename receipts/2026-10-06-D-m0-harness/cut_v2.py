"""Apply the official 0.2 retrieval cut to the (HLE-less) rebuilt 0.1 rows and stage suite-0.2/."""
import json, shutil, sys
from pathlib import Path
from decision_index.suite.build import release_v2
from decision_index.suite.download import hub_file
from decision_index.suite.io import sha256_file

work = Path(sys.argv[1]); out = Path(sys.argv[2])
v1 = work / "artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl"
excluded = set(json.loads(Path(hub_file("0.2", "excluded-questions.json")).read_text())["rows"])
dst = work / "artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl"
rep = release_v2.cut(v1, dst, excluded)
release_v2.gz(dst)
out.mkdir(parents=True, exist_ok=True)
for name in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
    shutil.copyfile(dst.parent / name, out / name)
shutil.copyfile(hub_file("0.2.1", "excluded-questions.json"), out / "excluded-questions.json")
shutil.copyfile(hub_file("0.2.1", "manifest.json"), out / "manifest.json")
rep["v1_sha256"] = sha256_file(v1)
print(json.dumps(rep))

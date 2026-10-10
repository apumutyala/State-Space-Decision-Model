set -u
S=/c/Users/apumu/AppData/Local/Temp/claude/C--Users-apumu-research-CHLabs/7d3e88d6-0884-4aac-8c4d-3a4ed67e2bfc/scratchpad
cd /c/Users/apumu/research/CHLabs/ssdm-wt/b-jev1
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""
P=../m2-heads/.venv/Scripts/python.exe
W="C:/Users/apumu/research/CHLabs/ssdm-data/m2/build/bschema-draw-jev1-work"
REC="C:/Users/apumu/research/CHLabs/ssdm-data/m2/build/bschema-draw-jev1/records.jsonl.gz"
R=receipts/2026-10-10-jev1-draw/bschema-draw-jev1
step() { name=$1; shift; timeout 1900 bash $S/waitram.sh 1100 1800 > /dev/null; echo "[$(date +%T)] start $name" >> $S/pipe.log; "$@" > $S/p_$name.log 2>&1; rc=$?; echo "[$(date +%T)] $name exit $rc" >> $S/pipe.log; return $rc; }
step gen timeout 900 $P -X utf8 -I scripts/jev1_build.py gen --work $W || exit 1
step check timeout 7000 $P -X utf8 -I scripts/jev1_build.py check --work $W || exit 1
step tok timeout 5400 $P -X utf8 -I scripts/jev1_tokcheck.py --shim-repo "$S/sizecliff" --records "$W/checked.jsonl.gz" --out "$W/lens-shim.jsonl" || exit 1
step final timeout 1200 $P -X utf8 -I scripts/jev1_build.py final --work $W --lens "$W/lens-shim.jsonl" --receipt $R --overwrite || exit 1
step doc timeout 3600 $P -X utf8 -m train.doc_overlap_guard scan --records $REC --out $R/doc-overlap-guard.json
step atk timeout 3600 $P -X utf8 -m train.attack_guard scan --records $REC --out $R/attack-guard.json
step jev timeout 3600 $P -X utf8 -m train.jevbench_guard rescan $REC --out $R/jevbench-guard-rescan.json --min-free-mb 1000
step rows timeout 3600 $P -X utf8 -I scripts/jev1_gates.py --records $REC --out $R/row-gates.json
echo "[$(date +%T)] ALL DONE" >> $S/pipe.log

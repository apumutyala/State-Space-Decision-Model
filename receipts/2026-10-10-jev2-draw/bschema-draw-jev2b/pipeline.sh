set -u
STAGE=${STAGE:-b}
S=/c/Users/apumu/AppData/Local/Temp/claude/C--Users-apumu-research-CHLabs/8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a/scratchpad/jev2
cd /c/Users/apumu/research/CHLabs/ssdm-wt/ev-jev2
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""
P=../m2-heads/.venv/Scripts/python.exe
PA=../m2-data/.venv/Scripts/python.exe   # the gen stage needs pyarrow (parquet sources)
SHIM=../jev2-4b                           # decide/shim_format.py, pinned jrt-h2o-shim:qwen3.5-4b (read only)
W="C:/Users/apumu/research/CHLabs/ssdm-data/m2/build/bschema-draw-jev2${STAGE}-work"
REC="C:/Users/apumu/research/CHLabs/ssdm-data/m2/build/bschema-draw-jev2${STAGE}/records.jsonl.gz"
R=receipts/2026-10-10-jev2-draw/bschema-draw-jev2${STAGE}
L=$S/pipe-$STAGE.log
step() { name=$1; shift; timeout 1900 bash $S/waitram.sh 1100 1800 >> $L; echo "[$(date +%T)] start $name" >> $L; "$@" > $S/p${STAGE}_$name.log 2>&1; rc=$?; echo "[$(date +%T)] $name exit $rc" >> $L; return $rc; }
step gen timeout 2400 $PA -X utf8 -I scripts/jev2_build.py gen --stage $STAGE --work $W || exit 1
step check timeout 7000 $P -X utf8 -I scripts/jev2_build.py check --work $W || exit 1
step tok timeout 5400 $P -X utf8 -I scripts/jev1_tokcheck.py --shim-repo "$SHIM" --records "$W/checked.jsonl.gz" --out "$W/lens-shim.jsonl" || exit 1
step final timeout 1200 $P -X utf8 -I scripts/jev2_build.py final --stage $STAGE --work $W --lens "$W/lens-shim.jsonl" --receipt $R --overwrite || exit 1
step doc timeout 3600 $P -X utf8 -m train.doc_overlap_guard scan --records $REC --out $R/doc-overlap-guard.json
step atk timeout 3600 $P -X utf8 -m train.attack_guard scan --records $REC --out $R/attack-guard.json
step jev timeout 3600 $P -X utf8 -m train.jevbench_guard rescan $REC --out $R/jevbench-guard-rescan.json --min-free-mb 1000
step rows timeout 3600 $P -X utf8 -I scripts/jev1_gates.py --generator data.jev2 --records $REC --out $R/row-gates.json
echo "[$(date +%T)] ALL DONE" >> $L

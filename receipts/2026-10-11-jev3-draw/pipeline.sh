set -u
# usage: PHASE=pool|pre|final|post bash receipts/2026-10-11-jev3-draw/pipeline.sh
#   pool : merge per-family pools (scripts/jev3_build.py gen --families f, run first) -> check -> shim token lengths
#   pre  : draw-level gates on the CHECKED pool -> drop-ids (provenance, strict full-row Index fp, verbatim >= 6-word
#          leaves, doc_overlap (Index + JevBench), attack_guard, jevbench rescan)
#   final: final (window, state dedupe, balance, slice) with the drop-ids
#   post : every gate again on the FINAL train + slice records (0 hits required) + jev1 row gates + shape gate -> GATES.json
PHASE=${PHASE:?}
S=/c/Users/apumu/AppData/Local/Temp/claude/C--Users-apumu-research-CHLabs/8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a/scratchpad/jev3
cd /c/Users/apumu/research/CHLabs/ssdm-wt/ev-jev3
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""
P=../m2-heads/.venv/Scripts/python.exe
PA=../m2-data/.venv/Scripts/python.exe   # pyarrow (MATH parquet reference)
SHIM=../jev2-4b
B=C:/Users/apumu/research/CHLabs/ssdm-data/m2/build
W=$B/bschema-draw-jev3-work
R=receipts/2026-10-11-jev3-draw
TR=$B/bschema-draw-jev3/records.jsonl.gz
SL=$B/bschema-draw-jev3-slice/records.jsonl.gz
L=$S/pipe-$PHASE.log
mkdir -p $R/pre $R/train $R/slice
step() { name=$1; shift; timeout 1900 bash $S/../jev2/waitram.sh 900 1800 >> $L; echo "[$(date +%T)] start $name" >> $L; "$@" > $S/p_${PHASE}_$name.log 2>&1; rc=$?; echo "[$(date +%T)] $name exit $rc" >> $L; return $rc; }
case $PHASE in
pool)
  step merge $P -X utf8 -I scripts/jev3_build.py merge --work $W || exit 1
  step check timeout 9000 $P -X utf8 -I scripts/jev3_build.py check --work $W || exit 1
  step tok timeout 5400 $P -X utf8 -I scripts/jev1_tokcheck.py --shim-repo "$SHIM" --records "$W/checked.jsonl.gz" --out "$W/lens-shim.jsonl" || exit 1
  ;;
pre)
  C=$W/checked.jsonl.gz
  step prov timeout 3600 $PA -X utf8 -I scripts/jev3_prov_gate.py --records $C --out $R/pre/prov-gate.json --drop-ids-out $W/drop-prov.txt
  step strictfp timeout 5400 $P -X utf8 -I scripts/jev3_strict_fp.py --records $C --out $R/pre/strict-fp.json --ids-out $W/drop-strictfp.txt
  step leaves timeout 5400 $P -X utf8 scripts/regate_exact_leaves.py --records $C --out $R/pre/exact-leaves.json --ids-out $W/drop-leaves.txt
  step doc timeout 5400 $P -X utf8 -m train.doc_overlap_guard scan --records $C --out $R/pre/doc-overlap-guard.json
  step atk timeout 3600 $P -X utf8 -m train.attack_guard scan --records $C --out $R/pre/attack-guard.json
  step jev timeout 5400 $P -X utf8 -m train.jevbench_guard rescan $C --out $R/pre/jevbench-guard-rescan.json --min-free-mb 900
  step hs3 timeout 1800 $P -X utf8 -I scripts/jev3_hs3_dedupe.py --records $C --out $W/drop-hs3.txt --report $R/pre/hs3-dedupe.json
  step ids $P -X utf8 -I scripts/jev3_gates_summary.py dropids --pre $R/pre --out $W/drop-pre.txt --checked $C
  ;;
final)
  step final timeout 1800 $P -X utf8 -I scripts/jev3_build.py final --work $W --lens "$W/lens-shim.jsonl" --receipt $R --drop-ids $W/drop-pre.txt --overwrite || exit 1
  ;;
post)
  for role in train slice; do
    RR=$([ $role = train ] && echo $TR || echo $SL)
    step prov_$role timeout 3600 $PA -X utf8 -I scripts/jev3_prov_gate.py --records $RR --out $R/$role/prov-gate.json
    step strictfp_$role timeout 5400 $P -X utf8 -I scripts/jev3_strict_fp.py --records $RR --out $R/$role/strict-fp.json
    step leaves_$role timeout 5400 $P -X utf8 scripts/regate_exact_leaves.py --records $RR --out $R/$role/exact-leaves.json
    step doc_$role timeout 5400 $P -X utf8 -m train.doc_overlap_guard scan --records $RR --out $R/$role/doc-overlap-guard.json
    step atk_$role timeout 3600 $P -X utf8 -m train.attack_guard scan --records $RR --out $R/$role/attack-guard.json
    step jev_$role timeout 5400 $P -X utf8 -m train.jevbench_guard rescan $RR --out $R/$role/jevbench-guard-rescan.json --min-free-mb 900
    step rows_$role timeout 5400 $P -X utf8 -I scripts/jev1_gates.py --generator data.jev3 --records $RR --out $R/$role/row-gates.json
  done
  zcat $B/bschema-draw-jev3/bschema-draw-jev3.lens.jsonl.gz > $W/lens-train.jsonl
  step shape timeout 1200 $P -X utf8 -I scripts/jev2_shape_gate.py check --fp receipts/2026-10-10-jev2-draw/public-shape-fingerprint.json --records $TR --lens $W/lens-train.jsonl --out $R/shape-gate.json
  step summary timeout 1200 $P -X utf8 -I scripts/jev3_gates_summary.py summary --receipt $R
  ;;
esac
echo "[$(date +%T)] PHASE $PHASE DONE" >> $L

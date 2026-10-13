set -u
# jev-heldout gates (laptop CPU only, $0). usage: PHASE=pre|check|final|post bash receipts/2026-10-13-jev-heldout/pipeline.sh
#   pre  : draw-level gates on the candidate POOL -> drop-ids (jev disjointness, provenance, strict full-row Index fp,
#          verbatim >= 6-word leaves, exact lines, 13-gram overlap incl. proxy600 + GSM8K test, doc_overlap, attack_guard,
#          jevbench rescan)
#   check: jev1 cmd_check on the pool (gold-once, Checker cohort v2/v3 + SplitGuard + harness.heldout, Index fp item
#          tier, JevBench fp + exact segment, attack_guard) -> checked.jsonl.gz (needs >= 1 GB free RAM)
#   final: quotas + seal (scripts/jevho_build.py final)
#   post : every gate again on the FINAL records (0 hits required) + jev1 row gates + shape gate -> GATES.json
PHASE=${PHASE:?}
S=/c/Users/apumu/AppData/Local/Temp/claude/C--Users-apumu-research-CHLabs/8ba1e3cb-efb0-46e2-9fd6-7683f7008f8a/scratchpad/jevho
cd /c/Users/apumu/research/CHLabs/ssdm-wt/ev-jev-heldout
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""
P=../m2-heads/.venv/Scripts/python.exe
PA=../m2-data/.venv/Scripts/python.exe   # pyarrow (GSM8K test parquet in the 13-gram overlap)
B=C:/Users/apumu/research/CHLabs/ssdm-data/m2/build
W=$B/jev-heldout-work
R=receipts/2026-10-13-jev-heldout
FIN=$B/jev-heldout/records.jsonl.gz
L=$S/pipe-$PHASE.log
mkdir -p $S $R/pre $R/final
freemb() { powershell -NoProfile -c "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" | tr -d '\r'; }
waitram() { local need=$1 t=0; while [ "$(freemb)" -lt "$need" ] && [ $t -lt 3600 ]; do sleep 30; t=$((t+30)); done; echo "[$(date +%T)] free $(freemb) MB (need $need, waited ${t}s)" >> $L; }
step() { name=$1; need=$2; shift 2; waitram $need; echo "[$(date +%T)] start $name" >> $L; "$@" > $S/p_${PHASE}_$name.log 2>&1; rc=$?; echo "[$(date +%T)] $name exit $rc" >> $L; return $rc; }
gates() {  # $1 = records, $2 = receipt dir, $3 = ids prefix ("" = no drop files)
  local REC=$1 RD=$2 X=$3 D=$S/draw-$(basename $RD)
  mkdir -p $D && cp $REC $D/records.jsonl.gz
  step disjoint 400 timeout 5400 $P -X utf8 -I scripts/jevho_disjoint.py --records $REC --out $RD/disjoint-jev.json ${X:+--ids-out ${X}disjoint.txt}
  step prov 400 timeout 3600 $P -X utf8 -I scripts/jevho_prov_gate.py --records $REC --out $RD/prov-gate.json ${X:+--drop-ids-out ${X}prov.txt}
  step strictfp 700 timeout 5400 $P -X utf8 -I scripts/jev3_strict_fp.py --records $REC --out $RD/strict-fp.json ${X:+--ids-out ${X}strictfp.txt}
  step leaves 700 timeout 5400 $P -X utf8 scripts/regate_exact_leaves.py --records $REC --out $RD/exact-leaves.json ${X:+--ids-out ${X}leaves.txt}
  step lines 700 timeout 5400 $P -X utf8 scripts/regate_line_exact.py --records $REC --out $RD/line-exact.json
  step overlap13 900 timeout 7200 $PA -X utf8 scripts/regate_jev2a.py overlap --draw $D --out $RD/overlap-13gram.json ${X:+--dump-hit-ids ${X}overlap13.txt} --min-free-mb 600
  step doc 700 timeout 5400 $P -X utf8 -m train.doc_overlap_guard scan --records $REC --out $RD/doc-overlap-guard.json
  step atk 500 timeout 3600 $P -X utf8 -m train.attack_guard scan --records $REC --out $RD/attack-guard.json
  step jev 900 timeout 5400 $P -X utf8 -m train.jevbench_guard rescan $REC --out $RD/jevbench-guard-rescan.json --min-free-mb 600
}
case $PHASE in
pre)
  gates $W/pool.jsonl.gz $R/pre $W/drop-
  step ids 300 $P -X utf8 -I scripts/jevho_gates.py dropids --pre $R/pre --work $W --out $W/drop-pre.txt
  ;;
check)
  step check 1024 timeout 9000 $P -X utf8 -I scripts/jevho_build.py check --work $W || exit 1
  ;;
final)
  step final 400 timeout 1800 $P -X utf8 -I scripts/jevho_build.py final --work $W --lens $W/lens-shim.jsonl --drop-ids $W/drop-pre.txt --receipt $R --overwrite || exit 1
  ;;
post)
  gates $FIN $R/final ""
  step rows 900 timeout 5400 $P -X utf8 -I scripts/jev1_gates.py --generator data.jev_heldout --records $FIN --out $R/final/row-gates.json
  zcat $B/jev-heldout/jev-heldout.lens.jsonl.gz > $W/lens-final.jsonl
  step shape 300 timeout 1200 $P -X utf8 -I scripts/jev2_shape_gate.py check --fp receipts/2026-10-10-jev2-draw/public-shape-fingerprint.json --records $FIN --lens $W/lens-final.jsonl --out $R/final/shape-gate.json
  step summary 300 $P -X utf8 -I scripts/jevho_gates.py summary --receipt $R
  ;;
esac
echo "[$(date +%T)] PHASE $PHASE DONE" >> $L

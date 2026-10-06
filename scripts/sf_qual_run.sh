#!/usr/bin/env bash
# First paid job: <=$20 H100 qualification run (SF Compute autoresearch, one h100-1 node). Runs ON the node.
#
# Orchestration (from the agent, NOT from this script):
#   0. LEDGER planned row + `spend` issue acked by Codex; code merged to agent-main (COORDINATION: merge before paid run).
#   1. open_mission("m1-qual-h100")
#   2. create_node(name="claude-m1-qual", chip="h100-1", image=<catalog pytorch-* entry>, mission=...)
#   3. upload three code archives (git archive) to /home/dev/m1q/code/{m1,decide,steer}.tgz via write_file / autor cp:
#        m1     = claude/m1-conversion-pilot HEAD (this repo)
#        decide = claude/m0-decide-adapter @ a5031f7
#        steer  = claude/m0-steer-eval   @ 944fa02
#      plus this script at /home/dev/m1q/sf_qual_run.sh
#   4. run_command(node, "bash /home/dev/m1q/sf_qual_run.sh", detach=true, max_duration_minutes=345, mission=...)
#      (platform-enforced 5.75 h -> max $17.08 at $2.97/h; this script also stops the node itself on EXIT)
#   5. poll get_command; on each new line in /home/dev/m1q/export/MANIFEST call export_file(node, that .tgz)
#   6. after exit: export_file(node, "/home/dev/m1q/export"); confirm get_node == stopped; record actual $ in LEDGER.
#
# Phases (each exports /home/dev/m1q/export/<phase>.tgz):
#   setup  pip pins; fetch 0.8B + data (fg); RWKV7 / Mamba-3 / 9B (bg from t=0)
#   A      tests/test_gdn.py + tests/test_init_shapes.py on the H100            (timeout 10m; FAIL -> exit 2)
#   0      m0 GPU items: RWKV7 + Mamba-3 debias controls, fixture regen (injection-03), steer canonical smoke (<=20m)
#   1      0.8B curve, configs/m1_pilot_0p8b_h100.json                             (timeout 60m)
#   2      9B throughput/memory probe, configs/m1_9b_qual.json                     (timeout = remaining budget)
#   3      decision rule (scripts/decide_9b.py), plots, final export, POST /v1/stop
set -uo pipefail
ROOT=/home/dev/m1q
mkdir -p $ROOT/{out,export,logs,data,extras}
export SSDM_DATA_ROOT=$ROOT/data HF_HOME=$ROOT/data/hfhome PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
T0=$(date +%s)
BUDGET_MIN=${BUDGET_MIN:-345}   # must equal run_command max_duration_minutes
RESERVE_MIN=12                  # final export + stop
LOG=$ROOT/logs/run.log
exec > >(tee -a $LOG) 2>&1

elapsed_min() { echo $(( ($(date +%s) - T0) / 60 )); }
remaining_min() { echo $(( BUDGET_MIN - RESERVE_MIN - $(elapsed_min) )); }
say() { echo "[$(date -u +%FT%TZ) +$(elapsed_min)m] $*"; }
export_phase() {
  tar czf $ROOT/export/$1.tgz -C $ROOT out logs 2>/dev/null || true
  echo "$(date -u +%FT%TZ) $1 $ROOT/export/$1.tgz" >> $ROOT/export/MANIFEST
  say "exported phase $1"
}
finish() {
  rc=$?
  say "EXIT rc=$rc"
  echo "{\"exit_code\": $rc, \"elapsed_min\": $(elapsed_min)}" > $ROOT/out/run_exit.json
  export_phase final
  curl -s -m 20 -X POST -H 'Metadata-Flavor: givemeanode' http://169.254.42.1/v1/stop || true
}
trap finish EXIT

# ------------------------------------------------------------------ setup
say "setup"
for a in m1 decide steer; do mkdir -p $ROOT/code/$a && tar xzf $ROOT/code/$a.tgz -C $ROOT/code/$a; done
nvidia-smi | tee $ROOT/out/nvidia-smi.txt
python -m pip install -q "transformers==5.17.0" "flash-linear-attention==0.5.2" "fla-core==0.5.2" einops pyarrow \
  huggingface_hub safetensors accelerate matplotlib numpy pytest 2>&1 | tail -3
python - <<'EOF' | tee $ROOT/out/env.json
import json, torch, transformers, importlib.metadata as m
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                  "fla": m.version("flash-linear-attention"), "gpu": torch.cuda.get_device_name(0)}))
EOF
# Mamba-3 (m0 phase 0): mamba_ssm at the commit PR #6 used, Triton path only (no CUDA build), in its own target dir
(MAMBA_SKIP_CUDA_BUILD=TRUE python -m pip install -q --no-deps --target $ROOT/extras/mamba \
   "mamba_ssm @ git+https://github.com/state-spaces/mamba@e9594ce" > $ROOT/logs/mamba_install.log 2>&1 \
   && touch $ROOT/extras/MAMBA_OK) &
cd $ROOT/code/m1
python -I scripts/sf_fetch.py --stage first || exit 5
(python -I scripts/sf_fetch.py --stage rest > $ROOT/logs/fetch_rest.log 2>&1) &   # 9B etc. in the background
source $SSDM_DATA_ROOT/paths.env
export_phase setup

# ------------------------------------------------------------------ phase A: kernel tests on the H100
say "phase A"
mkdir -p $ROOT/out/tests
timeout 10m python -m tests.test_gdn > $ROOT/out/tests/test_gdn_h100.json; RA=$?
timeout 5m python -m tests.test_init_shapes > $ROOT/out/tests/test_init_shapes.json
export_phase A
[ $RA -eq 0 ] || { say "test_gdn FAIL ($RA) -> stop"; exit 2; }

# ------------------------------------------------------------------ phase 0: deferred m0 GPU items (<= 20 min total)
say "phase 0"
P0_END=$(( $(date +%s) + 20*60 ))
p0() {  # p0 <name> <max minutes> <cmd...> : bounded by both its own cap and the phase-0 deadline
  local name=$1 cap=$2; shift 2
  local left=$(( (P0_END - $(date +%s)) / 60 )); [ $left -lt $cap ] && cap=$left
  [ $cap -le 0 ] && { say "phase0 $name skipped (time)"; echo skipped > $ROOT/out/m0/$name.status; return; }
  ( timeout ${cap}m "$@" ) > $ROOT/out/m0/$name.out 2> $ROOT/logs/m0-$name.err; echo $? > $ROOT/out/m0/$name.status
  say "phase0 $name rc=$(cat $ROOT/out/m0/$name.status)"
}
mkdir -p $ROOT/out/m0
for i in $(seq 1 12); do [ -f $SSDM_DATA_ROOT/DONE_MAMBA3_TOK_DIR ] && break; sleep 10; done
source $SSDM_DATA_ROOT/paths.env
cd $ROOT/code/decide
p0 debias-rwkv7-0.1b-g1 4 python -I scripts/run_debias_controls.py --family rwkv7 --model-dir "${RWKV7_DIR:-missing}" \
   --model-id fla-hub/rwkv7-0.1B-g1 --device cuda
p0 debias-mamba3-siso-187m 4 python -I scripts/run_debias_controls.py --family mamba3 --model-dir "${MAMBA3_DIR:-missing}" \
   --tokenizer-dir "${MAMBA3_TOK_DIR:-missing}" --model-id state-spaces/mamba3-siso-187m --device cuda --extras $ROOT/extras/mamba
p0 fixture-v0-regen 6 python -I scripts/gen_replay_fixture.py --qwen-dir "$Q08_DIR" --out $ROOT/out/m0/fixture-v0-regen \
   --runs 2 --device cpu --dtype float32
cd $ROOT/code/steer
p0 steer-canonical-qwen3.5-0.8b 8 env SSDM_QWEN_DIR="$Q08_DIR" SSDM_DECIDE_PATH=$ROOT/code/decide \
   python -m safety_eval.run_smoke --backend decide --device cuda --out $ROOT/out/m0/steer-canonical-qwen3.5-0.8b
export_phase 0

# ------------------------------------------------------------------ phase 1: 0.8B pipeline-qualification curve
say "phase 1"
cd $ROOT/code/m1
export PHASE_HOOK="bash $ROOT/code/m1/scripts/sf_phase_hook.sh m1-0p8b"
DEADLINE_MINUTES=55 timeout 60m python -m convert.pipeline --config configs/m1_pilot_0p8b_h100.json \
  --out $ROOT/out/m1-0p8b > $ROOT/logs/m1-0p8b.log 2>&1; R1=$?
echo $R1 > $ROOT/out/m1-0p8b/exit_code
python scripts/plot_curve.py $ROOT/out/m1-0p8b > /dev/null 2>&1 || true
export_phase 1
say "phase 1 rc=$R1"

# ------------------------------------------------------------------ phase 2: 9B throughput/memory probe
say "phase 2"
while [ ! -f $SSDM_DATA_ROOT/DONE_rest ] && [ $(remaining_min) -gt 100 ]; do sleep 30; done
source $SSDM_DATA_ROOT/paths.env
REM=$(remaining_min)
if [ -z "${Q9B_DIR:-}" ] || [ $REM -lt 90 ]; then
  say "9B not available or <90 min left (REM=$REM) -> skip phase 2"; echo skipped > $ROOT/out/phase2.status
else
  export PHASE_HOOK="bash $ROOT/code/m1/scripts/sf_phase_hook.sh m1-9b"
  DEADLINE_MINUTES=$((REM - 15)) timeout ${REM}m python -m convert.pipeline --config configs/m1_9b_qual.json \
    --out $ROOT/out/m1-9b > $ROOT/logs/m1-9b.log 2>&1; R2=$?
  echo $R2 > $ROOT/out/m1-9b/exit_code
  say "phase 2 rc=$R2 (3 = throughput stop, 4 = memory stop, 124 = timeout)"
fi
export_phase 2

# ------------------------------------------------------------------ phase 3: pre-registered decision
say "phase 3"
[ -f $ROOT/out/m1-9b/results.json ] && python scripts/decide_9b.py $ROOT/out/m1-9b > /dev/null
exit 0

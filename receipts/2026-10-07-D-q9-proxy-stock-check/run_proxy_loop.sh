#!/bin/bash
# usage: q9fcl_pl.sh <subcmd> [extra args]
cd /c/Users/apumu/research/CHLabs/ssdm-wt/q9fcl-run || exit 9
SP=/c/Users/apumu/AppData/Local/Temp/claude/C--Users-apumu-research-CHLabs/167d8677-872b-45ed-b144-905f7c6dd3bc/scratchpad
cmd=$1; shift
exec uv run --no-project --with torch --with numpy --with transformers==5.17.0 --with C:/Users/apumu/research/CHLabs/ssdm-data/decision-index \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  python -X utf8 -m train.proxy_loop $cmd --cache C:/Users/apumu/q9cache/stock_train/x/cache \
  --out C:/Users/apumu/q9cache/q9fcl_run "$@"

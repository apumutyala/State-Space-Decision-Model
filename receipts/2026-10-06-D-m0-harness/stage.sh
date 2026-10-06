#!/bin/bash
cd ~/ssdm-data
grep -A12 '"release_v1"' logs/rebuild.log | head -14
cp /mnt/c/Users/apumu/research/CHLabs/ssdm-data/scripts/cut_v2.py ~/ssdm-data/
.venv/bin/python -I ~/ssdm-data/cut_v2.py ~/ssdm-data/work /mnt/c/Users/apumu/research/CHLabs/ssdm-data/suite-0.2

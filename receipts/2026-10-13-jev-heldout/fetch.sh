set -u
# jev-heldout raw fetch (untrusted DATA; stdlib fetcher with a disk floor). Usage: bash receipts/2026-10-13-jev-heldout/fetch.sh
R=C:/Users/apumu/research/CHLabs/ssdm-data/raw/jevho
LOG=receipts/2026-10-13-jev-heldout/fetch-log.jsonl
P=../m2-heads/.venv/Scripts/python.exe
f() { $P -X utf8 -I scripts/jev3_fetch.py --url "$1" --out "$R/$2" --expect-mb "$3" --floor-mb 400 --log $LOG; }
HF=https://huggingface.co/datasets
GH=https://raw.githubusercontent.com
f $HF/nvidia/HelpSteer2/resolve/990b2711a36180dd19d9c94b8627844866f8982a/validation.jsonl.gz helpsteer2/validation.jsonl.gz 1
f $HF/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/resolve/d86bb8bedff51d25ac834ab7838f1cc61acb7a2c/validation.json aegis2/validation.json 2
f $HF/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/resolve/d86bb8bedff51d25ac834ab7838f1cc61acb7a2c/test.json aegis2/test.json 2
f https://media.githubusercontent.com/media/openai/prm800k/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/data/phase1_test.jsonl prm800k/phase1_test.jsonl 10
f https://media.githubusercontent.com/media/openai/prm800k/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/data/phase2_test.jsonl prm800k/phase2_test.jsonl 30
f https://media.githubusercontent.com/media/openai/prm800k/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits/test.jsonl prm800k/math_splits_test.jsonl 1
f https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip ambigqa/ambignq_light.zip 2
f $GH/NExTplusplus/TAT-QA/870accc41953dcde885aabeb963d94aabdc0fbc3/dataset_raw/tatqa_dataset_dev.json tatqa/tatqa_dataset_dev.json 2
f $GH/wenhuchen/Time-Sensitive-QA/38b05989070c1168b2bef3d5a2656afeeba763dc/dataset/human_annotated_dev.json timeqa/human_annotated_dev.json 1
f $GH/wenhuchen/Time-Sensitive-QA/38b05989070c1168b2bef3d5a2656afeeba763dc/dataset/human_annotated_test.json timeqa/human_annotated_test.json 1
f $GH/AbhilashaRavichander/PrivacyQA_EMNLP/5178ff3eaf61cbdaea03d85ff23b676b3fea451f/data/policy_test_data.csv privacyqa/policy_test_data.csv 30
f $HF/google/boolq/resolve/35b264d03638db9f4ce671b711558bf7ff0f80d5/data/validation-00000-of-00001.parquet boolq/validation-00000-of-00001.parquet 2
f $HF/bdsaglam/musique/resolve/22873a405dd809893b22ada0b499299fb612d2df/musique_ans_v1.0_dev.jsonl musique/musique_ans_v1.0_dev.jsonl 30
f $HF/SetFit/sst5/resolve/e51bdcd8cd3a30da231967c1a249ba59361279a3/test.jsonl sst5/test.jsonl 1
f $HF/SetFit/sst5/resolve/e51bdcd8cd3a30da231967c1a249ba59361279a3/dev.jsonl sst5/dev.jsonl 1

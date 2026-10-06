"""Fetch the pinned HLE parquet (needs accepted terms on the logged-in HF account)."""
import os, shutil
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download
p = hf_hub_download("cais/hle", "data/test-00000-of-00001.parquet", repo_type="dataset", revision="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29", local_dir=r"C:/Users/apumu/research/CHLabs/ssdm-data/hle-raw")
print(p)

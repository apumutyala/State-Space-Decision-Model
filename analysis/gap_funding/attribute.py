"""Attribute each proxy600 family gap (D13 - Clef) to one primary cause class; sum per class (X = judgement, evidence
column cites the measurement). Reads gap_funding.json only. Run: python -X utf8 -I analysis/gap_funding/attribute.py <gap_funding.json> <out.json>
Classes:
  frozen    - catalog in heldout-draw-v1: no data/generator/distillation may target it; only general capacity moves it
  data      - task family absent from D13's 24k draw (or its label prior is far off); allowed train-split / synthetic exists
  format    - family present but the eval SHAPE is never trained: K > train_max_k 16, K 10 vs K 4, docs > T 2,048
  context   - rows refused by the 16,384-token input cap (measured by transplanting Clef-flash's answers for those rows)
  readout   - parameter-free readout rules (PR #238 R1/R2) or polarity misread, measured on proxy600 (diagnostic)
  capacity  - family present in training, shape and prior matched, still wrong (or D13 >= Clef: a negative gap)
"""
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
# n -> (class, evidence). Readout / context carve-outs are subtracted from the family's gap first (see SPLIT).
CLS = {
    9: ("frozen", "Home appliances, 18 q/case, case-exact; R4 joint decoding falsified (DEAD_ENDS)"),
    32: ("frozen", "MuSR"), 44: ("frozen", "CLadder"), 33: ("frozen", "SATA (R2 top-m rule +0.16 split to readout)"),
    20: ("frozen", "BPoMP"), 22: ("frozen", "POP909 K 129"), 31: ("frozen", "ChessBench K 47"), 64: ("frozen", "New Yorker"),
    58: ("frozen", "BBH"),
    43: ("data", "CRUXEval: 0 rows in D1-24k (synthetic CRUXEval exists in D16/D17 draws); precedent: 877 GSM8K rows took D1 from 9.6 to 69.2 (subset)"),
    56: ("data", "PhishNChips: no training analog in any draw; D13 0.0 (answers 'legitimate' 17/19)"),
    48: ("data", "ForecastBench: absent; global yes/no train prior 14-19% yes vs 28% gold (leaves Index 0.3)"),
    61: ("data", "HoVer: absent from D1-24k (in 100k-v3 / v2e)"),
    37: ("data", "Amazon-ESCI: no 4-way E/S/C/I analog (nq binary only)"),
    41: ("data", "VAST: absent from D1-24k"), 50: ("data", "Habermas: absent from D1-24k"),
    38: ("data", "ACOS: absent; 58 q/row, 8k-token rows (also format)"),
    42: ("data", "NLI4CT: clinical domain absent (train split quarantined for premise overlap)"),
    59: ("data", "RAGTruth: absent from D1-24k; D13 AUROC 0.18 = polarity misread (flip +1.18 diag, unclaimed)"),
    23: ("capacity", "CFColor: 0 for every arm"), 40: ("capacity", "iSarcasm: 0 for every arm"),
    3: ("format", "API-Bank: eval K 53, 6.5k tok/row; D1-24k max K 16, T 2,048"),
    4: ("format", "BANKING77: eval K 77; train_max_k 16 dropped every K77 row"),
    5: ("format", "CLINC150: eval K 151 + 21% OOS; train K <= 16, OOS 0.8%"),
    57: ("format", "MMLU-Pro: eval K 10; mmlu_aux K 4 only"),
    11: ("format", "ContractNLI: 3.9k-token docs (100% > 2,048), NotMentioned 42%; anli short 3-way"),
    36: ("format", "BRIGHT: 9.2k tok, 32 docs/query; 5 queries refused by the 16k cap (split to context)"),
    2: ("format", "ToolRet: 7.5k tok, 32 docs/query; 1 query refused (split to context)"),
    62: ("capacity", "When2Call: in draw (895 rows); D13 regressed vs D1 55.6 -> 23.8 (tool_call AUROC 0.87 -> 0.59)"),
    28: ("capacity", "WinoGrande: in draw, matched prior"), 29: ("capacity", "HellaSwag: in draw"),
    39: ("format", "FinEntity: sst5 score regression vs 3-way choice"),
    1: ("capacity", "BFCL: in draw (glaive/toolace); R1 exactly-one rule +0.21 split to readout"),
    45: ("capacity", "HLE"), 21: ("capacity", "Humicroedit: in draw 50/50, neither arm discriminates"),
    25: ("capacity", "GPQA"), 12: ("capacity", "ANLI (D13 > Clef)"), 30: ("capacity", "GSM8K (D13 > Clef)"),
}
# measured carve-outs (Index pts on proxy600, diagnostic): readout rules (READOUT_LEVERS_D13), context transplant (M here)
ctx_all = d["context_transplant"]["gain_pts"]
SPLIT = {1: ("readout", 0.21), 33: ("readout", 0.16)}
out = {}
for ref in ("flash", "clef27"):
    tot = {}
    rows = []
    for f in d["families"]:
        g = f[f"gap_{ref}_pts"]
        cls, ev = CLS[f["n"]]
        parts = {}
        if f["n"] in SPLIT:
            k, v = SPLIT[f["n"]]
            v = min(v, max(0.0, -g))
            parts[k] = -v
            g += v
        parts[cls] = parts.get(cls, 0.0) + g
        for k, v in parts.items():
            tot[k] = tot.get(k, 0.0) + v
        rows.append({"n": f["n"], "family": f["family"], "area": f["area"], "gap": f[f"gap_{ref}_pts"], "class": cls,
                     "parts": parts, "evidence": ev})
    # context: transplant of that Clef's answers on D13's 6 refused rows (M); moved out of BRIGHT/ToolRet format
    ctx = ctx_all[ref]
    tot["context"] = -ctx
    tot["format"] += ctx
    out[ref] = {"by_class": {k: round(v, 2) for k, v in sorted(tot.items(), key=lambda kv: kv[1])},
                "total": round(sum(tot.values()), 2), "rows": rows}
    print(ref, out[ref]["total"], out[ref]["by_class"])
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1)

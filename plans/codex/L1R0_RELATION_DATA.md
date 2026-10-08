# L1r0 relation data preparation

Base: `agent-main@32d975159c64216193c47478e102d17c8d7e5658`.
Implements the data-only phase of `L1_RELATION_SLOT_HEAD.md`.

Hypothesis: the frozen 24 fit / 8 reserved calibration / 32 validation
worlds provide 576 / 192 / 768 relation examples with an exact earlier-choice
prefix and isolated relation supervision. This is a conditional diagnostic,
not free-running L1 capability or calibration evidence.

Add a separate adapter without changing the generator, training allowlist,
operation runner or free-running evaluation adapter. Verify complete pinned
partitions before selecting slots. Per coordinator clarification, R0 executes
no Item conversion because `heads.__init__` eagerly imports Torch. The
deferred fit Item method calls the existing whole-world consumer on all 3,072
fit rows before selecting relation Items; L1r1 executes that guarded proof.
Keep world identity and provenance outside requests. Validation requests
contain only the earlier operation/left/right choices; relation targets live
in a distinct evaluation sidecar and never in request or cache identity.

Exit proof: focused CPU tests reject source drift, empty/short worlds and
partition crossing; check exact prefixes, sidecar-only labels, normalized
content disjointness, and label-independent model inputs. Measure every
actual encoded row with the pinned RWKV vocabulary against 2,048 tokens,
including continuation lengths, refusing over-limit rows without truncation.
Use the exact serving `jrt-rwkv-g1-v1` template and ordered layout. R0 reads
only the pinned vocabulary and uses the stock prompt's segmented encoding;
exact runtime LayoutRow acceptance remains a guarded runner preflight.
Record exact source/partition/selected-row and tokenizer hashes and token
counts. No Torch, checkpoint/model load, inference, GPU, Index/heldout or spend.
Proceed to the relation-cache seam only after these proofs pass.

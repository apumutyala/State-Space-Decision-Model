"""Keel: an auditable judgment layer for AI workflows.

Subpackages:
  protocol  typed, trust-labelled decision requests; prompt rendering; replayable receipts
  runtime   Modal images, model cache, the vLLM judge (one-pass and think-first scoring)
  host      the host-side enforcement kernel: authority, deferral, review queue (bounded verification)
  gate      act/defer gate with cluster-valid risk certification
  card      the Decision Alignment Card: paired units, attacks, evidence retention, policy compliance, metrics
  distill   discernment distillation: triplet data, teacher labelling, one-pass student training

Design and claims: direction/MOONSHOT.md and PROPOSAL.md. Interface contract between subpackages: keel/CONTRACT.md.
"""

__version__ = "0.1.0"

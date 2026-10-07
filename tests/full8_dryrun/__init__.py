"""CPU stand-ins for the GPU / kit / private-data / weight-fetch steps of jobs/full8/full8_job.sh (FULL8_DRYRUN=1 only).

Everything else the runner does runs for real in the dry run: the preflight, the DEV-subset row filter, the REAL
convert.pipeline (stage 1, KL importance, 8 progressive steps with the enforced gate, rollback, resume from the
checkpoint dir, the PHASE_HOOK checkpoint POSTs), the REAL weight export (jobs.full8.export_gdn, incl. the strict HF
load), the REAL readout plugin loading the exported file (convert.readout.load_converted_plugin, inside the kit stub),
the REAL rate_guard wrappers, the REAL decide_full8 on the pipeline's own results.json, and finalize.py.
"""

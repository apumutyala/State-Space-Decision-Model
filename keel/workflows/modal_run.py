"""One H100: local vLLM agent; a 4B judge is loaded only for judgment arms."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import modal

from keel.paths import private_root
from keel.runtime.budget import approved_run
from keel.runtime.modal_app import HF_CACHE, MODELS, hf_cache, results_vol, vllm_image

APP = modal.App("keel-agentdojo-workflow")
IMAGE = vllm_image.pip_install("agentdojo==0.1.35").add_local_python_source("keel", "decide")


class LocalJudge:
    """Same single-label raw-logprob readout as keel.runtime.vllm_judge."""

    def __init__(self, max_think=1024):
        from vllm import LLM
        model, revision = MODELS["qwen3.5-4b"]
        self.llm = LLM(model=model, revision=revision, max_model_len=32768, gpu_memory_utilization=0.30,
                       max_num_seqs=4, max_num_batched_tokens=8192, enforce_eager=True,
                       enable_prefix_caching=True, max_logprobs=256, seed=0)
        self.tokenizer = self.llm.get_tokenizer()
        self.label_ids = {}
        self.max_think = max_think

    def __call__(self, rendered):
        from vllm import SamplingParams
        from keel.protocol.render import close_think, probs_from_logprobs
        from keel.protocol.receipts import PromptUse
        from .gate_element import JudgeEvaluation
        prompt = rendered.prompt
        reasoning, truncated = None, False
        if rendered.think:
            sample = self.llm.generate(
                [prompt], SamplingParams(max_tokens=self.max_think, temperature=.6, top_p=.95,
                                         top_k=20, seed=0, stop=["</think>"]), use_tqdm=False,
            )[0].outputs[0]
            reasoning = sample.text
            truncated = sample.finish_reason == "length"
            prompt = close_think(rendered, reasoning, truncated)
        base = self.tokenizer.encode(prompt, add_special_tokens=False)
        for label in rendered.labels:
            if label not in self.label_ids:
                extended = self.tokenizer.encode(prompt + " " + label, add_special_tokens=False)
                if extended[:-1] != base or len(extended) != len(base) + 1:
                    raise ValueError(f"label {label} is not one token at Answer:")
                self.label_ids[label] = extended[-1]
        output = self.llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0, logprobs=256),
                                  use_tqdm=False)[0]
        logprobs = output.outputs[0].logprobs[0]
        missing = [label for label in rendered.labels if self.label_ids[label] not in logprobs]
        if missing:
            raise RuntimeError(f"missing raw label logprobs: {missing}")
        values = [logprobs[self.label_ids[label]].logprob for label in rendered.labels]
        prompts = [PromptUse.from_rendered("q", rendered)]
        if rendered.think:
            prompts.append(PromptUse.from_rendered("q", rendered, reasoning=reasoning, truncated=truncated))
        return JudgeEvaluation(probs_from_logprobs(rendered, values), tuple(prompts))


@APP.function(image=IMAGE, gpu="H100", timeout=3 * 3600,
              volumes={HF_CACHE: hf_cache, "/results": results_vol})
def evaluate(config: dict, run_id: str):
    import importlib.metadata
    import urllib.request
    from .runner import EndpointClient, run_matrix

    start = time.monotonic()
    root = Path("/results/workflow") / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.json").write_text(json.dumps(config, indent=2))
    model, revision = MODELS["qwen3.5-9b"]
    command = ["vllm", "serve", model, "--revision", revision, "--host", "127.0.0.1", "--port", "8000",
               "--gpu-memory-utilization", "0.55", "--max-model-len", "32768", "--max-num-seqs", "8",
               "--max-num-batched-tokens", "8192", "--enforce-eager", "--enable-auto-tool-choice",
               "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3"]
    process = None
    meta = {"run_id": run_id, "config": config, "models": MODELS,
            "agentdojo": importlib.metadata.version("agentdojo"), "vllm": importlib.metadata.version("vllm"),
            "agent_server_command": command, "gpu": "H100", "hourly_usd_estimate": 3.95}
    try:
        with (root / "agent_server.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM agent exited {process.returncode}; see {root}/agent_server.log")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3):
                        break
                except (OSError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError("vLLM server startup exceeded 1200 seconds")
            meta["agent_load_s"] = time.monotonic() - start
            judge = (LocalJudge(config.get("judge_max_think", 1024))
                     if any(arm in ("keel", "keel_think") for arm in config["arms"]) else None)
            meta["judge_used"] = judge is not None
            meta["total_load_s"] = time.monotonic() - start
            client = EndpointClient("http://127.0.0.1:8000/v1", model, max_tokens=config.get("max_tokens", 1024))
            rows = run_matrix(client, judge, config, root)
            meta["episodes"] = len(rows)
            meta["errors"] = sum(row["error"] is not None for row in rows)
            meta["executed_calls"] = sum(row["executed_calls"] for row in rows)
            meta["dispatch_errors"] = sum(row["dispatch_errors"] for row in rows)
            if any(row["receipt_chain_verified_count"] is not None for row in rows):
                from .audit import verify_run
                meta["receipt_verification"] = verify_run(root, sample_size=20, seed=20261015)
            meta["completed"] = True
    except Exception as exc:
        meta["completed"] = False
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        meta["gpu_seconds_estimate"] = time.monotonic() - start
        meta["gpu_minutes_estimate"] = meta["gpu_seconds_estimate"] / 60
        meta["cost_usd_estimate"] = meta["gpu_seconds_estimate"] * 3.95 / 3600
        (root / "run.json").write_text(json.dumps(meta, indent=2))
        results_vol.commit()
        print(json.dumps(meta), flush=True)
    return meta


def download_run(run_id, destination):
    root = f"workflow/{run_id}"
    destination = Path(destination)
    for entry in results_vol.iterdir(root, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = Path(entry.path).relative_to(root)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            for chunk in results_vol.read_file(entry.path):
                stream.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default=str(private_root() / "workflow"))
    parser.add_argument("--download-only", action="store_true", help="download a finished run's logs (no GPU)")
    parser.add_argument("--max-usd", type=float, help="cap for an evaluation run, within the owner's approved budget")
    args = parser.parse_args()
    destination = Path(args.output) / args.run_id
    if args.download_only:
        download_run(args.run_id, destination)
        return
    if args.max_usd is None:
        parser.error("an evaluation run needs --max-usd within the owner's approved budget")
    config = json.loads(Path(args.config).read_text())
    with modal.enable_output(), approved_run(APP, label=f"workflows.{args.run_id}", cap_usd=args.max_usd):
        try:
            evaluate.with_options(timeout=config.get("function_timeout_seconds", 3 * 3600)).remote(config, args.run_id)
        finally:
            download_run(args.run_id, destination)


if __name__ == "__main__":
    main()

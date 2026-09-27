"""Run a bounded GPU compatibility matrix; never upload adapters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    """Run the bounded GPU compatibility matrix and return a failing status on errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--report", type=Path, default=Path(".runs/compatibility.json"))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    deadline = time.monotonic() + max(0, args.minutes) * 60
    root = Path(".runs") / ("smoke-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S"))
    root.mkdir(parents=True)
    os.environ["LORA_STUDIO_RUNS_ROOT"] = str(root.resolve())
    from lora_finetune_studio import jobs
    from lora_finetune_studio.models import (
        DatasetSpec,
        EvaluationConfig,
        JobState,
        PeftMode,
        TrainingApproach,
        TrainingConfig,
    )
    from lora_finetune_studio.provenance import prepare_run
    from lora_finetune_studio.sources import get_hf_token

    jobs.RUNS_ROOT = root
    sft = root / "sft.jsonl"
    preference = root / "preference.jsonl"
    sft.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"What is {i} plus one?"},
                        {"role": "assistant", "content": str(i + 1)},
                    ]
                }
            )
            for i in range(12)
        ),
        encoding="utf-8",
    )
    preference.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt": f"What is {i} plus one?",
                    "chosen": str(i + 1),
                    "rejected": str(i + 8),
                }
            )
            for i in range(12)
        ),
        encoding="utf-8",
    )
    cases = [(TrainingApproach.SFT, PeftMode.QLORA, False)]
    cases += [
        (a, m, False)
        for a in (
            TrainingApproach.REWARD,
            TrainingApproach.DPO,
            TrainingApproach.SFT,
            TrainingApproach.KTO,
            TrainingApproach.ORPO,
        )
        for m in PeftMode
        if (a, m, False) not in cases
    ]
    cases += [
        (a, m, True) for a in TrainingApproach for m in (PeftMode.LORA, PeftMode.QLORA)
    ]
    results = []

    def persist():
        jobs.write_json_atomic(
            args.report,
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "platform": sys.platform,
                "budget_minutes": args.minutes,
                "model": "Qwen/Qwen3-0.6B",
                "results": results,
            },
        )

    def wait(run_id, *, interrupt_for_resume=False):
        resumed = False
        while time.monotonic() < deadline:
            jobs.dispatch_next_run()
            status = jobs.read_status(run_id)
            config = jobs.read_config(run_id)
            checkpoints = list(
                Path(config.output_dir).glob("checkpoint-*/trainer_state.json")
            )
            if (
                interrupt_for_resume
                and not resumed
                and checkpoints
                and status.state is JobState.RUNNING
            ):
                jobs.cancel_run(run_id, dispatch_next=False)
                jobs.resume_run(run_id)
                resumed = True
            if (
                status.state
                in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
                and jobs.active_run() is None
            ):
                return status, resumed
            time.sleep(0.5)
        jobs.cancel_run(run_id, dispatch_next=False)
        raise TimeoutError("GPU validation budget exhausted.")

    for index, (approach, method, unsloth) in enumerate(cases):
        entry = {
            "approach": str(approach),
            "method": str(method),
            "backend": "unsloth" if unsloth else "standard",
            "status": "not run",
        }
        results.append(entry)
        if index >= args.limit or time.monotonic() >= deadline:
            entry["reason"] = "Case limit or time budget reached."
            persist()
            continue
        try:
            if unsloth and not Path(".venv-unsloth/Scripts/python.exe").exists():
                environment = os.environ.copy()
                environment["UV_PROJECT_ENVIRONMENT"] = str(
                    Path(".venv-unsloth").resolve()
                )
                with (root / "unsloth-setup.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        [
                            "uv",
                            "--system-certs",
                            "sync",
                            "--project",
                            "unsloth-runtime",
                            "--locked",
                            "--no-dev",
                            "--python",
                            "3.13.13",
                        ],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=max(1, deadline - time.monotonic()),
                        check=True,
                    )
            is_sft = approach is TrainingApproach.SFT
            spec = DatasetSpec(
                source="upload",
                local_path=str((sft if is_sft else preference).resolve()),
                format="messages" if is_sft else "preference",
                prompt_column="prompt",
                chosen_column="chosen",
                rejected_column="rejected",
            )
            config = TrainingConfig(
                model_id="Qwen/Qwen3-0.6B",
                datasets=[spec],
                approach=approach,
                peft_mode=method,
                use_unsloth=unsloth,
                max_length=128,
                max_steps=4 if index == 0 else 2,
                batch_size=2 if approach is TrainingApproach.KTO else 1,
                gradient_accumulation_steps=1,
                checkpoint_steps=1,
                eval_enabled=True,
                select_best_checkpoint=True,
            )
            prepare_run(config, get_hf_token())
            run_id = jobs.enqueue_run(config)
            entry["run_id"] = run_id
            status, resumed = wait(run_id, interrupt_for_resume=index == 0)
            entry["resume_verified"] = resumed and status.state is JobState.COMPLETED
            if status.state is not JobState.COMPLETED:
                raise RuntimeError(status.error or status.message)
            parent = jobs.read_config(run_id)
            evaluation = EvaluationConfig(
                rows=[
                    {"prompt": "What is 40 plus two?", "chosen": "42", "rejected": "7"}
                ],
                mode="reward" if approach is TrainingApproach.REWARD else "chat",
                sample_limit=1,
                max_new_tokens=16,
            )
            eval_id = jobs.enqueue_run(
                replace(
                    parent,
                    job_kind="evaluation",
                    parent_run_id=run_id,
                    evaluation=evaluation,
                    use_unsloth=False,
                    resume_from_checkpoint=None,
                    push_to_hub=False,
                )
            )
            eval_status, _ = wait(eval_id)
            if eval_status.state is not JobState.COMPLETED:
                raise RuntimeError(eval_status.error or eval_status.message)
            entry.update(
                status="GPU verified",
                evaluation_id=eval_id,
                metrics=status.metrics,
                manifest=str(Path(parent.output_dir).parent / "manifest.json"),
            )
        except Exception as error:  # noqa: BLE001
            message = str(error)
            for name in ("HF_TOKEN", "OPENAI_API_KEY"):
                if os.getenv(name):
                    message = message.replace(os.environ[name], "[REDACTED]")
            entry.update(status="failed", reason=message)
        persist()
        print(json.dumps(entry), flush=True)
    return int(any(entry["status"] == "failed" for entry in results))


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the Study-B solo runner ENTIRELY on Modal, detached from the laptop.

WHY THIS EXISTS. The normal invocation runs the orchestration loop locally: the
local process assembles prompts, calls the model APIs, translates each action
into a sandbox execution, scores, and writes results. That requires the laptop
to stay awake for hours. This driver moves the WHOLE loop into one Modal
container per model, so after `modal run --detach` the laptop can be shut down;
the three drivers keep running in the cloud, each spawning per-execution
sandboxes exactly as the local loop did (``modal.App.lookup`` resolves the same
way from inside Modal as outside).

WHAT LIVES WHERE
----------------
* code + frozen data  -> baked into the driver image at launch (add_local_dir);
  results/ and external/ and .git are EXCLUDED, so the upload is ~50 MB.
* API keys            -> a Modal Secret named ``pp-api-keys`` (created by
  ``--create-secret`` below; never baked into the image).
* results             -> a Modal Volume ``pp-results``, committed after every
  checkpoint file so a crash loses at most one checkpoint. Download later with
  ``modal volume get pp-results / results_remote``.

USAGE
-----
    # once: store the keys as a Modal secret (reads the scratchpad env file)
    python scripts/modal_driver.py --create-secret <path-to-api_keys.env>

    # launch, then the laptop may be closed
    modal run --detach scripts/modal_driver.py::run_all_models

    # or a single model
    modal run --detach scripts/modal_driver.py::run_model --model sol

    # watch / fetch afterwards
    modal app list
    modal volume get pp-results / results_remote
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/repo"

# results/, external/ and caches stay home: they are large and the driver does
# not read them. data/ and configs/ and the pinned swebench source are needed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyyaml>=6.0", "openai>=1.0", "anthropic>=0.40",
                 "google-genai>=0.1", "modal")
    .add_local_dir(str(LOCAL_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(str(LOCAL_ROOT / "studies"), remote_path=f"{REMOTE_ROOT}/studies")
    .add_local_dir(str(LOCAL_ROOT / "scripts"), remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_dir(str(LOCAL_ROOT / "configs"), remote_path=f"{REMOTE_ROOT}/configs")
    .add_local_dir(str(LOCAL_ROOT / "data"), remote_path=f"{REMOTE_ROOT}/data")
    .add_local_dir(str(LOCAL_ROOT / ".cache" / "swebench"),
                   remote_path=f"{REMOTE_ROOT}/.cache/swebench")
)

app = modal.App("pp-study-b-driver")
results_vol = modal.Volume.from_name("pp-results", create_if_missing=True)

def _runner_args(checkpoints: str, stimuli: str, out: str,
                 arm: str = "ours_prompt_open_tools") -> list:
    return [
        "--arm", arm,
        "--checkpoints", checkpoints,
        "--stimuli", stimuli,
        "--only-benchmark", "impossible",
        "--max-turns", "50",
        "--resume",
        "--out", out,
    ]


def _run(model: str,
         checkpoints: str = "data/checkpoints/checkpoints_v3.json",
         stimuli: str = "data/checkpoints/stimuli_v3.json",
         out: str = "results/B_v3_oursopen",
         arm: str = "ours_prompt_open_tools") -> int:
    """Invoke the unmodified runner inside the container."""
    cmd = [sys.executable, "scripts/run_study_b.py",
           "--model-config", f"configs/models/{model}.yaml",
           *_runner_args(checkpoints, stimuli, out, arm)]
    print(f"[driver] {' '.join(cmd)}", flush=True)
    # line-buffered passthrough so `modal app logs` shows live progress
    proc = subprocess.Popen(cmd, cwd=REMOTE_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        # persist results as they land, not only at the end
        if ".json" in line or "checkpoint" in line.lower():
            try:
                results_vol.commit()
            except Exception:                                  # noqa: BLE001
                pass
    proc.wait()
    results_vol.commit()
    print(f"[driver] {model} finished rc={proc.returncode}", flush=True)
    return int(proc.returncode or 0)


@app.function(image=image, volumes={f"{REMOTE_ROOT}/results": results_vol},
              secrets=[modal.Secret.from_name("pp-api-keys")],
              timeout=60 * 60 * 12, cpu=2.0, memory=4096)
def run_model(model: str,
              checkpoints: str = "data/checkpoints/checkpoints_v3.json",
              stimuli: str = "data/checkpoints/stimuli_v3.json",
              out: str = "results/B_v3_oursopen",
              arm: str = "ours_prompt_open_tools") -> int:
    """One model's episodes (4 conditions x N tasks), driven from the cloud."""
    return _run(model, checkpoints, stimuli, out, arm)


@app.function(image=image, volumes={f"{REMOTE_ROOT}/results": results_vol},
              secrets=[modal.Secret.from_name("pp-api-keys")],
              timeout=60 * 60 * 12, cpu=2.0, memory=4096)
def run_swarm(models: str = "sol,fable,gemini",
              arm: str = "official",
              prompt_arm: str = "ours",
              checkpoints: str = "data/checkpoints/checkpoints_v3.json",
              out: str = "results/StudyB_swarm_oursuntyped_mixed_t50",
              turns_per_agent: int = 50) -> int:
    """One swarm corpus (3 agents, shared fork + board), driven from the cloud.

    Defaults reproduce the ours-prompt + open-tools mixed swarm; pass
    models='sol,sol,sol' and the 3sol out dir for the homogeneous corpus.
    --resume is always on: completed checkpoint files are never re-run.
    """
    cmd = [sys.executable, "scripts/run_study_b_swarm.py",
           "--models", models, "--arm", arm, "--prompt-arm", prompt_arm,
           "--checkpoints", checkpoints, "--out", out,
           "--turns-per-agent", str(turns_per_agent), "--resume"]
    print(f"[driver] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=REMOTE_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if ".json" in line or "checkpoint" in line.lower():
            try:
                results_vol.commit()
            except Exception:                                  # noqa: BLE001
                pass
    proc.wait()
    results_vol.commit()
    print(f"[driver] swarm finished rc={proc.returncode}", flush=True)
    return int(proc.returncode or 0)


@app.function(image=image, volumes={f"{REMOTE_ROOT}/results": results_vol},
              secrets=[modal.Secret.from_name("pp-api-keys")],
              timeout=60 * 60 * 12, cpu=2.0, memory=4096)
def run_all_models(checkpoints: str = "data/checkpoints/checkpoints_v3.json",
                   stimuli: str = "data/checkpoints/stimuli_v3.json",
                   out: str = "results/B_v3_oursopen",
                   arm: str = "ours_prompt_open_tools") -> dict:
    """All three models, concurrently, from one detached launch."""
    handles = [run_model.spawn(m, checkpoints, stimuli, out, arm)
               for m in ("sol", "fable", "gemini")]
    return {m: h.get() for m, h in zip(("sol", "fable", "gemini"), handles)}


def _create_secret(env_file: str) -> None:
    """Store the three keys as the Modal secret ``pp-api-keys``."""
    kv = {}
    for line in Path(env_file).read_text().splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[7:].split("=", 1)
            kv[k] = v.strip("'\"")
    missing = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                           "GOOGLE_API_KEY") if k not in kv]
    if missing:
        raise SystemExit(f"env file lacks {missing}")
    subprocess.run(
        ["modal", "secret", "create", "pp-api-keys", "--force",
         *[f"{k}={v}" for k, v in kv.items()]],
        check=True)
    print("secret pp-api-keys created")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--create-secret":
        _create_secret(sys.argv[2])
    else:
        print(__doc__)

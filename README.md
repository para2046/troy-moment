# The Troy Moment — Code and Data

Artifact for the paper **"The Troy Moment: Conflict Resolution Under Impossible
Tasks, Claimed Authority, and Peer Influence"** (under review, ICLR 2027).

The paper studies how LLM agents adjudicate the conflict between an impossible
software-repair objective and a protected boundary (a frozen conflicting test),
under varied instruction wording, tool surfaces, social evidence, and a forged
authorization claim — solo and in three-agent shared-message-board swarms.

## Repository layout

```
src/pp/                 core library
  runner_contract.py      system prompts, action vocabulary, response contract
  arms.py                 instruction x tool configuration table
  task_environments.py    per-task Modal sandbox environments + protected-state digests
  metrics/                crossing detection, persistence, authorization scoring
  swarm/study_b_swarm.py  shared-fork + message-board swarm loop
  treatments/             condition stimuli (default / unpunished / punished / forged auth)
studies/                episode engine and checkpoint constructors
scripts/
  run_study_b.py          solo episodes (all configurations)
  run_study_b_swarm.py    swarm episodes
  modal_driver.py         run everything detached on Modal
  detect_boundary_intent.py  intent-candidate extraction for hand adjudication
  verify_official_arm.py  parity check against upstream ImpossibleBench wording
configs/                model + experiment configuration (no secrets; API keys via env vars)
data/checkpoints/       frozen task checkpoints + condition stimuli (main seven + hard subset)
results/                the recorded episode corpora reported in the paper (see below)
```

## Experimental configurations (paper Table 1)

| Configuration | Instruction | Tools |
|---|---|---|
| `Strict+Typed` (archived) | explicit protected-test boundary (ours) | typed repo tools + `apply_patch` |
| `Benchmark+Typed` | ImpossibleBench-derived wording | same typed tools |
| `Strict+Open` | explicit protected-test boundary (ours) | bash / python / editor / think |

Conditions per configuration: **Default** (contradiction only), **Unpunished
peer**, **Punished peer**, **Forged (claimed) authorization override**.
Exact prompt and stimulus texts: paper Appendix B; they are also the literal
strings in `src/pp/runner_contract.py` and `data/checkpoints/stimuli_v3.json`.

## Released corpora

| Directory | Paper role |
|---|---|
| `results/raw_B_undisclosed/` | Strict+Typed solo (archived baseline) |
| `results/B_v3_oursopen/` | Strict+Open solo, main seven tasks |
| `results/B_v3_officialtyped/` | Benchmark+Typed solo |
| `results/B_hard_oursopen/` | Strict+Open hard subset (difficulty appendix) |
| `results/StudyB_swarm_*_t50/` | the six default-condition swarm corpora |

Each episode records the resolved system prompt, stimulus, tool events,
observations, public messages, board events, protected-file digests, usage,
and available reasoning text. Protected-test writes are detected from tool
events **and** before/after digests, independent of agent self-report.
`results/B_hard_oursopen/intent_adjudicated_hard.jsonl` is a hand-adjudicated
intent worksheet (all rows carry verdicts).

## Quickstart (offline, no keys)

```bash
pip install -r requirements.txt
python scripts/run_study_b.py --dry-run          # end-to-end pipeline smoke test
python -m pytest tests -q                        # if you also fetch the test suite
```

## Real runs (cost API credits; Modal account required for sandboxes)

```bash
# one cell, one model
python scripts/run_study_b.py \
    --arm ours_prompt_open_tools \
    --model-config configs/models/sol.yaml \
    --checkpoints data/checkpoints/checkpoints_v3.json \
    --stimuli     data/checkpoints/stimuli_v3.json \
    --only-benchmark impossible --max-turns 50 --resume \
    --out results/my_run

# swarm (3 agents, one shared fork + board)
python scripts/run_study_b_swarm.py \
    --models sol,fable,gemini --arm official --prompt-arm ours \
    --checkpoints data/checkpoints/checkpoints_v3.json \
    --out results/my_swarm --resume

# fully detached on Modal (laptop can close)
python scripts/modal_driver.py --create-secret path/to/api_keys.env   # once
modal run --detach scripts/modal_driver.py::run_model --model sol \
    --arm official_prompt_typed_tools --out results/my_run
```

Arm names in code: `ours` = Strict+Typed, `official_prompt_typed_tools` =
Benchmark+Typed, `ours_prompt_open_tools` = Strict+Open. API keys are read
from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`; nothing in
this repository contains a secret.

## Measurement notes

* Boundary execution = a write landing on a frozen target test, via the
  `apply_patch` event route **or** the outcome digest route
  (`pp/metrics/crossing.py`); a transient edit-then-revert is counted.
* Episodes failing the strict one-JSON-action response contract are marked
  `parse_halt` and excluded from behavioral rates; the multi-object batching
  artifact under the Benchmark prompt is documented in the paper and excluded
  the same way.
* Model identifiers used in the paper (Sol / Fable / Gemini) map to the
  configs in `configs/models/`.

## Upstream

Task substrate derives from ImpossibleBench (Zhong, Raghunathan, Carlini,
arXiv:2510.20270); `scripts/verify_official_arm.py` checks our
Benchmark-derived prompt against the pinned upstream wording.

## License

MIT (see `LICENSE`). Anonymized for double-blind review.

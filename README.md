# FaultLine

**A fault-injection benchmark for LLM agent loops.** FaultLine measures what happens to an agent *after* something goes wrong: does it recover, retry sensibly, spiral into repeated tool calls, or crash? Results across models and failure modes are published as a public, inspectable leaderboard.

Most evals measure capability — can the model do the task under ideal conditions. FaultLine measures **robustness** — what the agent does when a tool times out, returns garbage, or fails transiently mid-task. In production, this is where agents actually die, and there is currently no public, apples-to-apples answer to "which models doom-loop when a tool fails."

## How it works

FaultLine wraps tools at the call boundary, so faults are injected without touching the agent framework's internals. Any agent loop that calls tools through the wrapper can be tested — the harness is framework-agnostic by design (the reference implementation runs on LangGraph, but nothing couples to it).

```
                        ┌─────────────────────┐
  agent loop ──────────▶│  FaultLine wrapper   │──────▶ real tool
  (LangGraph, custom)   │  (injects failures)  │
                        └─────────────────────┘
                                   │
                                   ▼
                     telemetry + verdict pipeline
              (step count, tool-call signatures, token
               cost, termination state vs. baseline)
```

### Failure modes (v1)

| Mode | Behavior |
|---|---|
| `hard_error` | Tool raises an unrecoverable exception |
| `transient` | Tool fails N times, then succeeds (retry-then-succeed) |
| `malformed` | Tool returns syntactically valid but semantically wrong output |
| `empty` | Tool returns an empty / null result |
| `timeout` | Tool hangs past the agent's timeout budget |

Each mode runs in two variants: **caught** (error surfaced to the model as a tool result) and **propagated** (exception raised through the loop). This separates *model-adaptation failures* — the model saw the error and still handled it badly — from *loop-crash failures*, where the framework never gave the model a chance.

### Verdicts

Every faulted run is scored against a fault-free baseline of the same task and classified by the telemetry pipeline:

| Verdict | Meaning |
|---|---|
| **Recovered** | Task completed correctly despite the fault, within a bounded overhead of baseline steps/tokens |
| **Degraded** | Task completed, but with materially worse output or excessive cost |
| **Spiraled** | Agent abandoned the task strategy and wandered (novel but unproductive tool calls) |
| **Doom-looped** | Agent repeated the same tool-call signature past a threshold |
| **Crashed** | Loop terminated abnormally |

Verdicts are derived from run telemetry (step counts, repeated tool-call signatures, token cost, termination state), not from an LLM judge — every classification is reproducible from the trace.

## The leaderboard

The headline artifact is a hosted results page: a **model × failure-mode matrix** of recovery rates, filterable by fault mode, caught/propagated variant, and task. Every cell links to the underlying run traces, so any number on the board can be inspected down to the individual tool calls that produced it.

Reported per model:

- **Recovery rate** per failure mode (with Wilson 95% CIs — cells are small, intervals matter)
- **Doom-loop rate** and median loop length before termination
- **Recovery overhead**: extra steps and tokens spent recovering vs. the fault-free baseline
- **Caught vs. propagated delta**: how much of the failure rate is the model vs. the loop

## v1 scope

Deliberately small and finishable:

- **Models:** 3–4 (one frontier, one mid-tier, one small/open-weight)
- **Failure modes:** the 5 above × 2 error-handling variants
- **Tasks:** ~20 multi-step tool-use tasks with deterministic success checks
- **Runs:** 3 seeds per cell → ~1,200–1,800 runs per full sweep
- **Site:** static results page generated from run artifacts; traces stored as JSON, viewable per run

Non-goals for v1: adversarial/prompt-injection faults, multi-agent setups, LLM-judged task success, more than one agent scaffold per model.

## Repository layout

```
faultline/
├── harness/          # tool wrapper, fault injectors, run orchestration
├── telemetry/        # trace capture, signature hashing, verdict pipeline
├── tasks/            # task definitions + deterministic checkers
├── runs/             # raw run artifacts (JSON traces)
├── analysis/         # aggregation, CIs, leaderboard data generation
└── site/             # static leaderboard + trace viewer
```

## Quickstart

```bash
pip install -e .

# run one task against one model with a timeout fault
faultline run --task tasks/flight_rebooking.yaml \
              --model claude-sonnet-4-6 \
              --fault timeout --variant caught --seed 0

# full sweep + regenerate leaderboard data
faultline sweep --config sweeps/v1.yaml
faultline report --out site/data/
```

## Roadmap

- [ ] v1 sweep + public leaderboard
- [ ] Compound faults (e.g., transient failure *followed by* malformed output)
- [ ] Fault scheduling by loop position (early / mid / late in task)
- [ ] Community-submitted tasks with deterministic checkers
- [ ] Cost-of-robustness view: recovery rate vs. tokens spent, per dollar

## Methodology notes

- Baselines are re-run per model per task; verdicts always compare like against like.
- Tool-call "signatures" hash tool name + normalized arguments, so retries with trivially different phrasing still count as repetition.
- All thresholds (doom-loop repetition count, overhead bounds) are declared in `analysis/config.yaml` and versioned with the results they produced.

## License

MIT

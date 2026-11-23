# Running the GuardLedger revision kit on Kaggle

`guardledger_revision.py` runs the extra analyses the ESWA reviewers asked for. It does not change the V2 pipeline. The rule code, the thresholds and the 720 benchmark tasks stay frozen (V2 SHA-256 `2df52e0a…96a9`). No number from this kit may go into the manuscript until you have run it and sent back the output folder.

## 1. What you need

| Input | Where it comes from |
|---|---|
| `guardledger_src.py` (V2, unchanged) | the GitHub repository, `src/` |
| `guardledger_revision.py` | this kit |
| BPIC 2019 CSV | the same Kaggle dataset you used for V2 |
| V2 `llm_cache.jsonl` (Qwen and Phi runs) | download from your V2 notebook outputs and attach it as a Kaggle dataset. The V2 pipeline finds it through `CACHE_INPUT_GLOB=/kaggle/input/**/llm_cache*.jsonl`, so the original agent and judge generations are reused, not regenerated. |
| (GPU sections only) a ≥14B instruct model | Kaggle Models, e.g. Qwen2.5-14B-Instruct-AWQ. Add it to the notebook through "Add Input". |

## 2. Notebook layout

```python
# Cell 1: the unchanged V2 pipeline
exec(open("/kaggle/input/<your-dataset>/guardledger_src.py").read(), globals())

# Cell 2 (optional): overrides BEFORE the kit runs
REV_OVERRIDE = dict(
    JUDGE_MODELS=[dict(name="qwen2.5-14b-instruct-awq",
                       kaggle_globs=["/kaggle/input/qwen2.5/transformers/14b-instruct-awq/*/config.json"],
                       hf_id="Qwen/Qwen2.5-14B-Instruct-AWQ")],
    EXTRA_AGENT_MODELS=[dict(name="qwen2.5-14b-instruct-awq",
                             kaggle_globs=["/kaggle/input/qwen2.5/transformers/14b-instruct-awq/*/config.json"],
                             hf_id="Qwen/Qwen2.5-14B-Instruct-AWQ")],
)

# Cell 3: the revision kit
exec(open("/kaggle/input/<your-dataset>/guardledger_revision.py").read(), globals())
```

Check the glob path against the real path Kaggle shows in the Input panel. If no model file matches, that section is skipped and logged, not failed.

## 3. Two passes (recommended)

| Pass | Accelerator | Set | Sections that run | Rough time |
|---|---|---|---|---|
| A | CPU | `os.environ["GL_REV_GPU"]="0"` before Cell 3 | R1–R10 | 1–3 h (R8 seeds and R9 rolling calibration dominate) |
| B | GPU T4×2 or P100 | `GL_REV_GPU` unset, `REV_OVERRIDE` as above | R11–R14 (R1–R10 run again from cache, quickly) | 4–9 h, depending on the model |

If a session will hit the 12 h limit, shrink the run with `REV_OVERRIDE`. For example: `SEEDS=(1,2,3)`, `CLOSED_LOOP_PER_CAT=5`, `ROLL_N=6000`.

## 4. What each section produces (`/kaggle/working/<OUT_DIR>/revision/`)

| Section | Reviewer request | Output files |
|---|---|---|
| R1 | per-step and per-task logs for every guard and stream | `per_step_all_guards.csv.gz`, `per_task_all_guards.csv.gz` |
| R2 | paired GL-vs-Isolation-Forest test by typology group; LOF, conformal-IF, fused score, GL+anomaly combination | `paired_tests_by_group.csv`, `stronger_baselines_summary.csv` |
| R3 | agent parse failures; fail-open vs fail-closed judge (false holds) | `agent_parse_failures.csv`, `judge_fail_open_vs_closed.csv` |
| R4 | per-invoice audit, R3 by mined vs default tier, hard-rule exception rate | `audit_per_invoice.csv.gz`, `audit_facts.json` |
| R5 | SPLIT_ADAPTIVE mechanism confirmation | `split_mechanism_per_task.csv`, `split_mechanism_counts.csv` |
| R6 | packet-construction latency | `latency_packet_construction.csv` |
| R7 | policy sensitivity (authority quantile, tier minimum, default tier, budget quantile) | `policy_sensitivity_audit.csv` |
| R8 | 5 extra benchmark seeds (approve-all stream) | `seeds_approve_all_summary.csv`, `seeds_approve_all_groups.csv` |
| R9 | rolling-origin calibration by month: static, rolling, adaptive conformal | `rolling_calibration_by_month.csv` |
| R10 | held-out ACFE-style typologies (designed without reference to the rules) | `heldout_approve_all_summary.csv`, `heldout_approve_all_per_category.csv`, `heldout_pooled_and_paired.csv` |
| R11 | larger, history-aware judge; fail-open and fail-closed | `independent_judges_summary.csv` |
| R12 | ≥14B agent model on the benchmark and the held-out set | `extra_agent_summary.csv`, `extra_agent_per_category.csv` |
| R13 | closed-loop retry pilot (agent receives the hold reason, then retries) | `closed_loop_pilot.csv` |
| R14 | agent system-prompt variants ("terse", "cautious") | `prompt_sensitivity_summary.csv` |
| all | status of every section | `revision_summary.md` |

## 5. What to send back

Zip the whole `revision/` folder together with the notebook log. Then:
1. Read `revision_summary.md` first. Any `FAILED` line needs its traceback from the log.
2. The manuscript will be updated **only** from these files, with every number traced to a CSV cell.
3. Negative findings will be reported as they are. Examples: held-out recall below Isolation Forest; a judge that holds legitimate tasks when run fail-closed.

## 6. What this kit does not do

- **BPIC 2020** (travel and expense logs) is not implemented. It needs its own policy miner and its own benchmark generator. The manuscript lists it as future work.
- **Published guard baselines** (AgentSpec, ShieldAgent, GuardAgent, Progent, FinHarness) are not re-implemented. Their code, tool schemas or models do not map onto this benchmark without substantial porting. The manuscript compares to them qualitatively only.
- The held-out typologies come from a published fraud taxonomy, but they are still synthetic injections into real BPIC 2019 cases. They are not labelled real fraud.

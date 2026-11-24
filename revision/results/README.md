# Supplementary-experiment results (revision runs)

These files hold the result tables of the revision runs. The manuscript's supplementary numbers (Section 6, Supplementary Experiments; Section 7; the held-out and supplementary-experiment tables) are taken from them, and `checks/make_revision_tables.py` builds `tables/tab_heldout.tex` and `tables/tab_extra.tex` from them.

| Folder or prefix | Kaggle notebook | Machine | What it ran |
|---|---|---|---|
| `passA_gpu/` | guardledger-revision-pass-a-cpu, version 2 (the notebook name dates from version 1) | GPU T4×2 machine type, no model loaded | R0–R10 and R1: LLM-stream rebuild, fail-closed judges, paired tests, stronger baselines, audit facts, SPLIT mechanism, packet latency, policy sensitivity, seeds, rolling calibration, held-out typologies (approve-all) |
| `passB_*` | guardledger-revision-pass-b-gpu, version 1 | GPU T4×2 | R10–R13: held-out typologies, independent Qwen2.5-14B NF4 judge, Qwen2.5-14B agent, closed-loop pilot |

## How the files were made

The notebooks print every result file into their logs, between `<<<FILE name>>>` and `<<<END name>>>` markers (`revision/run_pass_a.py`). We copied the tables from the logs.

Files marked `_compact` keep the original rows and values, but with fewer columns:

- The `usd_*` and latency columns are dropped, except where a section uses them.
- Model names are written out in full.

## Not usable

Pass A version 1 and the export notebook ran on CPU sessions. Those sessions produced a different benchmark instance, so their results are not used anywhere; see `passB_v2_reproduction_check.txt`.

## Check against the main run (V2)

On the GPU machine type, the rebuilt LLM streams match V2 for every guard: harmful proposals, prevention, false-hold rate and executed amounts are all identical.

The two runs differ in one place. 13 of the 878 steps per stream had no cached generation, and their tasks were excluded. As a result, the HER denominators are 467 instead of 480.

# Confirmatory pass: pre-specified protocol

Written and committed **before** the confirmatory run. The commit hash of this file is cited in the manuscript.

## Why this pass exists

The supplementary experiments (passes A and B) showed two things:

- GuardLedger's price layer S1 did no better than a conformally thresholded Isolation Forest (IF) placed in the same position, and on the held-out typologies it did worse.
- The static split-conformal threshold went over its 5% target in four of five later months, while adaptive conformal inference (ACI) did not.

As a result, the residual layer is re-specified **after** those results as:

> a pluggable split-conformal residual detector at α = 0.05, with the Isolation Forest as the default detector, recalibrated online by ACI (γ = 0.005, window 2,000).

The deterministic rules R1–R8 are unchanged. This pass tests the re-specification on data it was **not** chosen on.

## Frozen inputs

- **Source and kit:** the V2 source (SHA-256 2df52e0a…96a9) and the kit, unchanged. They supply the rules, the mined policy and the IF model, IF features and IF calibration set from V2.
- **Configurations compared:**
  - GL-S1: hard rules + S1 with the deployed λ
  - GL-IF: hard rules + conformal IF
  - IF: conformal IF alone
  - GL-hard: hard rules only
  - plus the other kit baselines
- **No tuning:** no threshold, feature or hyper-parameter is tuned in this pass.

## Fresh held-out tasks (primary data)

- **Typologies and generator:** the same five held-out typologies and the same generator (`build_heldout`) as R10.
- **Seeds:** new seeds **2027, 4049 and 6067**, with 40 tasks per typology per seed. That gives 600 tasks, none of them from seed 314, which was used before.
- **Agent stream:** approve-all (worst-case compliant agent).

## Hypotheses and tests

All tests are exact two-sided McNemar tests on task-level outcomes over the tasks with a harmful proposal, at α = 0.05.

**H1 (primary).** On the fresh held-out tasks, GL-IF prevents more harmful proposals than GL-S1.
- Confirmed if GL-IF > GL-S1 and p < 0.05.

**H2.** On the fresh held-out tasks, GL-IF prevents at least as much as IF alone.
- Reported as the GL-IF − IF difference with its McNemar p.
- Confirmed if GL-IF ≥ IF, or if the difference is not significant at 0.05.

**H3.** On the frozen V2 benchmark instance (approve-all and the four cached LLM streams), GL-IF prevents at least as much as GL-S1, at the same false-hold rate on legitimate tasks.
- Confirmed if GL-IF ≥ GL-S1 in every stream, or if no stream shows a significant deficit.

**H4.** Applied to the IF score of oracle-compliant invoices in the test months, ACI keeps the overall false-escalation rate within 5% ± 1 percentage point.
- Also reported: the static conformal IF threshold and monthly rolling recalibration.
- Sample: the same 12,000-invoice sample as R9 (`random_state=21`).

## Decision rule

| Outcome | Consequence for the manuscript |
|---|---|
| H1 and H3 confirmed | The manuscript adopts GL-IF as the proposed residual layer and reports S1 as the ablation that motivated the change. |
| H1 not confirmed | The manuscript keeps S1 as the proposed layer and reports GL-IF as an alternative whose advantage did not replicate. |
| Any result | All results are reported whatever their direction. |

## Secondary, descriptive analyses (no hypothesis)

1. **Determinism diagnostics:**
   - PYTHONHASHSEED and CPU SIMD flags;
   - ties in invoice time;
   - fingerprints of the benchmark built by the V2 code, by the V2 code with `sorted()` set iteration, and by the V2 code with `sorted()` plus stable sorting.
2. **Restoration of the V2 instance:** for each agent step whose cached generation is missing, the V2 prompt is searched for among all vendor-master substitutions of the injected vendor name. A SHA-1 key match identifies the V2 vendor exactly.
3. **Export of the frozen task files**, with SHA-256:
   - the restored V2 benchmark (720 tasks);
   - the R10 held-out tasks (seed 314, 200 tasks);
   - the fresh held-out tasks (600 tasks).

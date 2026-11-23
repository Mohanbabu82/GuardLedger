# GuardLedger on Kaggle: run guide

## 1. Setup (one time)
1. Download the BPI Challenge 2019 CSV (about 38 MB) from 4TU.ResearchData, DOI `10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1` (CC BY 4.0).
   On Kaggle: **Datasets → New Dataset → upload the CSV**. Keep `BPI` and `2019` in the file name.
2. Upload `GuardLedger_Kaggle.ipynb` (**Code → New Notebook → File → Import**).
3. In the notebook's right panel:
   - **Add Input → Datasets:** your BPI 2019 dataset
   - **Add Input → Models:** *Qwen2.5 → Transformers → 7b-instruct*
   - **Accelerator:** GPU T4 ×2
   - **Internet:** on only if you want the Hugging Face fallback or vLLM
4. Run all.

## 2. Session plan (fits about 30–45 min per session)
| Session | What runs | Approx. time |
|---|---|---|
| 1 | Data, policy mining, benchmark, Experiments A–C (CPU), then LLM generation until `LLM_TIME_BUDGET_MIN=35` | ~40 min |
| 2+ | **Save Version**, attach the output (`llm_cache.jsonl`) as an input dataset, run again. Finished generations are skipped automatically. | ~35 min each |

- **Faster option:** set `PIP_INSTALL_VLLM=True` (needs Internet). vLLM makes the LLM stage roughly 5–10× faster. If it fails to load, the notebook falls back to transformers by itself.
- **Cross-model study:** uncomment the Llama-3.1-8B entry in `LLM_MODELS`. Reviewers like seeing two model families.
- **Quick test first:** set `MAX_EVENTS=600_000`, `N_LEGIT=120`, `N_PER_CAT=20`.

## 3. Outputs (`/kaggle/working/guardledger_out`)
| File | What it is |
|---|---|
| `data_summary.json`, `policy_mined.json` | Dataset and mined-policy description (paper §Data, §Method) |
| `expA_summary.csv`, `expA_per_category.csv`, `table_expA.tex` | Main guard comparison and ablations (approve-all stream) |
| `expB_crc_validity.csv`, `table_crc.tex`, fig3 | Conformal Risk Control guarantee on future payments |
| `expC_natural_audit.csv`, `table_audit.tex`, fig5 | Real control exceptions found in the company log |
| `expD_llm_summary.csv`, `table_expD.tex`, fig1 per model | Real LLM agents + guards (incl. LLM-as-judge) |
| `agent_behaviour.csv`, fig6 | How often LLM agents propose harmful payments |
| `mcnemar_tests.csv` | Paired significance tests |
| `decision_log.csv.gz`, `llm_raw_outputs.json` | Per-decision logs for error analysis and qualitative examples |

## 4. Contributions to claim (only after you have the real numbers)
1. **GuardLedger-Bench.** Built from a real P2P log. Policies are mined on a temporal split. 12 control-failure and fraud typologies, plus an in-the-wild audit track.
2. **Commitment-aware look-ahead.** Budget checks use actuals + open-PO encumbrances + the payment.
3. **Trajectory-level split-purchase detection.**
4. **Provenance taint.** Deterministic defence against invoice-borne prompt injection.
5. **Hard/soft hybrid.** Conformal Risk Control bounds the dollar-weighted rate of wrongly blocked legitimate payments.
6. **Empirical study with real open-weight LLM agents** against prompt-only, LLM-as-judge, AgentSpec-style and Isolation-Forest baselines.

The validity threats are listed in the last notebook cell; state them in the paper.

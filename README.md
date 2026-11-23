# GuardLedger

Deterministic, ledger-aware pre-execution verification of an LLM agent's invoice-approval tool calls.

GuardLedger checks each proposed payment against purchase-to-pay ledger state reconstructed as of decision time. That state comprises month-to-date actuals, open purchase-order commitments, the vendor master, mined approval authority and vendor history. On top of the ledger checks, GuardLedger adds:
- trajectory checks for duplicate and split purchases;
- equality of payment arguments with the system of record;
- a robust price score with a split-conformal (conformal risk control) escalation threshold.

The benchmark and natural audit are built from the public **BPI Challenge 2019** event log.

> This repository accompanies the manuscript *"GuardLedger: Commitment-Aware and Provenance-Grounded Runtime Verification of LLM Agents in Financial Workflows"* (submitted to IEEE Access). [Add the citation or DOI once it is available.]

## Repository layout
| Path | Contents |
|---|---|
| `src/guardledger_src.py` | Full pipeline: data loading, ledger, policy mining, benchmark, GuardLedger, baselines, CRC, natural audit, LLM agents and judges. Version used for the reported (V2) results: SHA-256 `2df52e0a5a96776cacca126dcfaead3e1b3a2be257f4f794d752d996313496a9`. |
| `notebooks/` | Kaggle notebooks for Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct. They differ only in `LLM_MODELS`. |
| `docs/RUN_ON_KAGGLE.md` | Step-by-step guide to running the pipeline on Kaggle. |
| `results/v2_qwen`, `results/v2_phi` | V2 result files (CSV/JSON) behind every number in the paper. |
| `analysis/` | Scripts that regenerate the paper's tables (`build_tables.py`, which self-checks 89 values), the rule-matched versus rule-independent table, and Figures 1–8. |
| `figures/` | Figures as PDF and PNG. |
| `tools/mock_bpi.py` | Generator of a small synthetic log for smoke tests. It is **not** used for any reported result. |

## Reproduce the tables and figures (no GPU, no dataset needed)
```bash
pip install -r requirements.txt
python analysis/build_tables.py           # writes analysis/tables_out/, prints "89 / 89 checks passed"
python analysis/make_rulesplit_table.py
python analysis/make_results_figs.py      # Figures 3-8 into figures/
python analysis/make_diagrams.py          # Figures 1-2 (diagrams)
```

## Re-run the full pipeline
1. Download the BPI Challenge 2019 log from 4TU.ResearchData. DOI: [10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1) (CC BY 4.0). The pipeline reads the CSV version. The data are **not** included in this repository.
2. Follow `docs/RUN_ON_KAGGLE.md`. All stages except LLM generation are deterministic given the seeds (42, 0, 1, 2, 7, 11).
3. LLM generation uses greedy decoding with FP16 weights and is cached. Re-generating without the cache may differ slightly, depending on the inference library and the GPU.

## Data and licence
- **Dataset:** B. van Dongen, *BPI Challenge 2019*, 4TU.ResearchData, 2019, CC BY 4.0. Amounts are anonymised by the publisher.
- **Code licence:** MIT. See `LICENSE`.

## Citation
[Add the BibTeX entry after publication.]

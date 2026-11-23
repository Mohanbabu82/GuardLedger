"""Generate the data-driven LaTeX tables of the GuardLedger manuscript directly from the V2 result files,
and verify every reported value.  Usage: python build_tables.py <res_v2_qwen> <res_v2_phi> <outdir>"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys, os, json, math
import numpy as np, pandas as pd
from scipy.stats import beta
QW, PH, OUT = (sys.argv[1:4] + [None] * 3)[:3]
QW = QW or str(ROOT/"results/v2_qwen"); PH = PH or str(ROOT/"results/v2_phi"); OUT = OUT or str(ROOT/"analysis/tables_out")
os.makedirs(OUT, exist_ok=True)
SA = pd.read_csv(f"{QW}/expA_summary.csv"); SA["stream"] = "approve_all"
SD = pd.concat([pd.read_csv(f"{d}/expD_llm_summary.csv") for d in (QW, PH)], ignore_index=True)
S = pd.concat([SA, SD], ignore_index=True)
CA = pd.read_csv(f"{QW}/expA_per_category.csv")
CD = pd.concat([pd.read_csv(f"{d}/expD_llm_per_category.csv") for d in (QW, PH)], ignore_index=True)
CAT = pd.concat([CA, CD], ignore_index=True)
BEH = pd.concat([pd.read_csv(f"{d}/agent_behaviour.csv") for d in (QW, PH)], ignore_index=True).drop_duplicates()
MC = pd.concat([pd.read_csv(f"{d}/mcnemar_tests.csv") for d in (QW, PH)], ignore_index=True).drop_duplicates()
CRC = pd.read_csv(f"{QW}/expB_crc_validity.csv"); AUD = pd.read_csv(f"{QW}/expC_natural_audit.csv")
MAN = json.load(open(f"{QW}/run_manifest.json")); POL = json.load(open(f"{QW}/policy_mined.json"))
JUDGE_PARSE = {"qwen2.5-7b-instruct|plain": (0, 771), "phi-3.5-mini-instruct|plain": (141, 848)}   # from Kaggle run logs
STREAMS = [("approve_all", "Approve-all"), ("qwen2.5-7b-instruct|plain", "Qwen2.5-7B, plain"),
           ("qwen2.5-7b-instruct|policy_prompt", "Qwen2.5-7B, policy"), ("phi-3.5-mini-instruct|plain", "Phi-3.5-mini, plain"),
           ("phi-3.5-mini-instruct|policy_prompt", "Phi-3.5-mini, policy")]
GN = {"NoGuard": "No guard", "AgentSpec-style": "Stateless rules", "IsolationForest": "Isolation Forest", "GL-hard (no soft layer)": "GL-hard",
      "GuardLedger (full)": "\\textbf{GuardLedger}", "GL-hard + fixed z>3 (no CRC)": "GL-hard + $z>3$"}
def judge(st): return [g for g in S[S.stream == st].guard if g.startswith("LLM-as-judge")]
def row(st, g): r = S[(S.stream == st) & (S.guard == g)]; assert len(r) == 1, (st, g); return r.iloc[0]
def cp(k, n): return (beta.ppf(.025, k, n - k + 1) if k else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)
from decimal import Decimal, ROUND_HALF_UP
def P(x, d=2):   # percentage with round-half-up (avoids 8.125 -> 8.12 float artefacts)
    return str(Decimal(repr(round(float(100 * x), 9))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
E = lambda x: f"{x:,.2f}" if abs(x - round(x)) > 1e-9 else f"{x:,.0f}"
CHECKS = []
def check(name, got, exp, tol=1e-9): CHECKS.append((name, got, exp, abs(got - exp) <= tol))
LEG_N = {"approve_all": 240, "qwen2.5-7b-instruct|plain": 239, "qwen2.5-7b-instruct|policy_prompt": 193,
         "phi-3.5-mini-instruct|plain": 240, "phi-3.5-mini-instruct|policy_prompt": 240}
def tab(name, caption, label, colspec, header, body, star=True, notes=None, size="\\footnotesize"):
    env = "table*" if star else "table"
    s = [f"\\begin{{{env}}}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
         "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\hline", header + " \\\\", "\\hline"] + body + ["\\hline", "\\end{tabular}"]
    if notes: s += ["\\par\\vspace{2pt}\\parbox{\\linewidth}{\\scriptsize " + notes + "}"]
    s += [f"\\end{{{env}}}"]
    open(f"{OUT}/{name}.tex", "w").write("\n".join(s) + "\n"); return "\n".join(s)

# ---------- Table VI: dataset ----------
D = MAN["data"]; L = POL["limits"]
body = [f"Source & BPI Challenge 2019 \\cite{{dongen2019bpi}}, doi:10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1, CC BY 4.0 \\\\",
        f"Event rows in file / retained & {D['n_lines']:,} / {D['n_loaded']:,} ({D['n_out_of_date_range']} outside 2017--2019 removed) \\\\",
        f"PO items retained / invoiced items & {D['n_po_items']:,} / {D['n_invoiced']:,} \\\\",
        f"Vendors / users / spend areas & {D['n_vendors']:,} / {D['n_users']:,} / {D['n_spend_areas']} \\\\",
        f"Total first-invoice amount & {D['total_eur']:,.0f} anonymised EUR \\\\",
        "\\hline",
        f"Training period & invoices before {D['cal_start']} (first 50\\%): tier limits, user authority, budgets, price-score statistics, Isolation Forest fit \\\\",
        f"Calibration period & {D['cal_start']} to {D['test_start']} (next 20\\%): CRC thresholds, Isolation Forest threshold, vendor master (with training) \\\\",
        f"Test period & from {D['test_start']} (last 30\\%): benchmark (720 tasks, 878 steps), CRC evaluation, natural audit \\\\",
        "\\hline",
        f"Mined limits T1/T2/T3 & {float(L['1']):,.0f} / {float(L['2']):,.0f} / {float(L['3']):,.0f} anonymised EUR (CFO unlimited) \\\\",
        f"Users with a mined tier & {sum(POL['n_users_by_tier'].values())} ({POL['n_users_by_tier'].get('3',0)} T3, {POL['n_users_by_tier'].get('2',0)} T2); all others T1 \\\\",
        f"Vendor master / area budgets & {POL['n_vendor_master']:,} vendors / {len(POL['budgets'])} budgets (Table A-III) \\\\"]
tab("table06_dataset", "Dataset, preprocessing, temporal partition and mined policy (V2 run).", "tab:dataset", "p{0.24\\linewidth}p{0.70\\linewidth}",
    "Quantity & Value", body, star=False)

# ---------- Table XI: main results ----------
body = []
for st, lab in STREAMS:
    gl = ["NoGuard", "AgentSpec-style", "IsolationForest"] + judge(st) + ["GL-hard (no soft layer)", "GuardLedger (full)"]
    body.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{lab}}}}} \\\\")
    for g in gl:
        r = row(st, g); name = "LLM judge" if g.startswith("LLM") else GN[g]
        prev = "0.00" if r.prevention == 0 else f"{P(r.prevention)} [{P(r.prev_lo)}, {P(r.prev_hi)}]"
        fhr = "0.00" if r.fpr_legit == 0 else f"{P(r.fpr_legit)} [{P(r.fpr_lo)}, {P(r.fpr_hi)}]"
        k_f = round(r.fpr_legit * LEG_N[st]); check(f"FHR count {st}/{g}", k_f / LEG_N[st], r.fpr_legit, 1e-6)
        k_h = round(r.harm_exec_rate_all * 480); check(f"HER count {st}/{g}", k_h / 480, r.harm_exec_rate_all, 1e-6)
        bold = g == "GuardLedger (full)"
        cells = [name, f"{int(r.n_harm_proposed)}", prev, fhr, P(r.harm_exec_rate_all), E(r.usd_leaked), E(r.usd_held)]
        if bold: cells[2:] = ["\\textbf{" + c + "}" for c in cells[2:]]
        body.append(" & ".join(cells[:2]) + " & " + " & ".join(cells[2:]) + " & " + (f"{k_f}/{LEG_N[st]}") + " \\\\")
tab("table11_main", "Main comparison on the synthetic benchmark. $n_{HP}$: tasks with a harmful proposal. Prevention and false-hold rate (FHR) with 95\\% bootstrap intervals (1,000 task resamples). HER: share of the 480 violation tasks with harmful execution. Amounts in anonymised EUR. The Phi-3.5-mini judge maps 141/848 unparseable outputs to ALLOW.",
    "tab:main", "llrrrrrr", "Guard & $n_{HP}$ & Prevention \\% [CI] & FHR \\% [CI] & HER \\% & Harmful EUR executed & Legit EUR held & FH count", body)

# ---------- Table XII: cross-model ----------
body = []
for st, lab in STREAMS[1:]:
    gl = row(st, "GuardLedger (full)"); ng = row(st, "NoGuard"); j = judge(st)
    jtxt = "--" if not j else f"{P(row(st, j[0]).prevention)} ({JUDGE_PARSE[st][0]}/{JUDGE_PARSE[st][1]})"
    nleg = LEG_N[st]
    body.append(f"{lab} & {int(ng.n_harm_proposed)} & {P(ng.harm_exec_rate_all)} & {nleg}/240 & {P(gl.prevention)} [{P(gl.prev_lo)}, {P(gl.prev_hi)}] & {P(gl.harm_exec_rate_all)} & {P(gl.fpr_legit)} & {jtxt} \\\\")
tab("table12_crossmodel", "Cross-model summary of the four LLM streams (720 tasks each). Unguarded HER and GuardLedger HER are shares of the 480 violation tasks; correct approvals are LEGIT tasks with a correct approval proposal. Per-typology proposal rates are shown in Fig.~\\ref{fig:behaviour}.",
    "tab:crossmodel", "lrrrrrrr", "Stream & $n_{HP}$ & Unguarded HER\\% & Correct appr. & GL prev.\\% [CI] & GL HER\\% & GL FHR\\% & Judge prev.\\% (unparseable)", body, size="\\scriptsize")

# ---------- Table XIII: ablation ----------
VAR = [("GuardLedger (full)", "GuardLedger (full)"), ("GL-hard + fixed z>3 (no CRC)", "GL-hard + fixed $z>3$"), ("GL-hard (no soft layer)", "GL-hard"),
       ("GL-hard - encumbrance", "GL-hard $-$ encumbrance"), ("GL-hard - trajectory", "GL-hard $-$ trajectory"),
       ("GL-hard - provenance", "GL-hard $-$ provenance"), ("GL-hard - quantitative", "GL-hard $-$ quantitative")]
body = []
for g, lab in VAR:
    cells = [lab]
    for st, _ in STREAMS:
        v = P(row(st, g).prevention)
        if g != "GuardLedger (full)":
            m = MC[(MC.stream == st) & (MC.baseline == g)].iloc[0]; assert m.baseline_only_prevented == 0
            v += f" ({int(m.gl_only_prevented)})"
        cells.append(v)
    body.append(" & ".join(cells) + " \\\\")
tab("table13_ablation", "Ablation: prevention (\\%) by guard variant and stream. In parentheses: tasks prevented by the full guard but not by the variant (exact McNemar discordant count); the reverse count is 0 in every cell.",
    "tab:ablation", "lrrrrr", "Variant & Approve-all & Qwen plain & Qwen policy & Phi plain & Phi policy", body)

# ---------- Table XIV: latency ----------
LG = [("AgentSpec-style", "Stateless rules"), ("GL-hard (no soft layer)", "GL-hard"), ("GuardLedger (full)", "GuardLedger"),
      ("GL-hard + fixed z>3 (no CRC)", "GL-hard + $z>3$"), ("IsolationForest", "Isolation Forest")]
def fmt_t(us): return f"{us:,.1f}~\\textmu s" if us < 1e3 else (f"{us/1e3:,.1f}~ms" if us < 1e6 else f"{us/1e6:,.2f}~s")
body = []
for g, lab in LG:
    body.append(lab + " & " + " & ".join(f"{fmt_t(row(st, g).lat_p50_us)} / {fmt_t(row(st, g).lat_p99_us)}" for st, _ in STREAMS) + " \\\\")
body.append("LLM judge (amortised) & -- & " + fmt_t(row(STREAMS[1][0], judge(STREAMS[1][0])[0]).lat_p50_us) + " & -- & " +
            fmt_t(row(STREAMS[3][0], judge(STREAMS[3][0])[0]).lat_p50_us) + " & -- \\\\")
tab("table14_latency", "Time per guard call, median / 99th percentile, over approval steps. Rule-evaluation times exclude ledger queries (packet construction) and LLM inference. Isolation Forest times include per-call library overhead. The judge value is batched generation time divided by the number of judged approvals and is not comparable with a single rule evaluation.",
    "tab:latency", "lrrrrr", "Guard & Approve-all & Qwen plain & Qwen policy & Phi plain & Phi policy", body)

# ---------- Table XV: McNemar ----------
NONNESTED = ["NoGuard", "AgentSpec-style", "IsolationForest", "GL-hard + fixed z>3 (no CRC)"]
body = []; ps = []
for st, lab in STREAMS:
    gl = NONNESTED[:3] + judge(st) + NONNESTED[3:]
    for i, g in enumerate(gl):
        m = MC[(MC.stream == st) & (MC.baseline == g)].iloc[0]; ps.append(m.p_value)
        name = "LLM judge" if g.startswith("LLM") else GN[g].replace("\\textbf{", "").replace("}", "")
        short = lab.replace("Qwen2.5-7B", "Qwen").replace("Phi-3.5-mini", "Phi").replace(", ", " ")
        body.append(f"{short if i == 0 else ''} & {name} & {int(m.gl_only_prevented)} & {int(m.baseline_only_prevented)} & {m.p_value:.2e} \\\\")
check("n non-nested tests", len(ps), 22); check("max p < 0.05/22", float(max(ps) < 0.05 / 22), 1.0)
tab("table15_mcnemar", "Exact McNemar tests, GuardLedger versus non-nested alternatives (22 tests; Bonferroni threshold $0.05/22=2.27\\times10^{-3}$). GL only / Alt.\\ only: tasks with a harmful proposal prevented by one guard and not the other.",
    "tab:mcnemar", "llrrr", "Stream & Alternative & GL only & Alt.\\ only & $p$", body, star=False)

# ---------- Table XVI: CRC ----------
n_cal, n_te, n_ov = 2054, 2470, 40
rng = np.random.default_rng(0); body = []
for _, r in CRC.iterrows():
    a = r.alpha; lossn = {"count": "count", "dollar": "weighted"}[r.loss] if a != "fixed z>3" else "count"
    k_rec = round(r.overpricing_recall * n_ov); check(f"recall {a}/{r.loss}", k_rec / n_ov, r.overpricing_recall, 1e-9)
    if lossn == "count":
        k = round(r.test_risk * n_te); check(f"risk {a}", k / n_te, r.test_risk, 1e-6); lo, hi = cp(k, n_te)
        if a != "fixed z>3":
            al = float(a); ka = int(math.floor(al * (n_cal + 1) - 1 + 1e-12))
            q = beta.rvs(ka + 1, n_cal - ka, size=200000, random_state=rng); rr = rng.binomial(n_te, q) / n_te
            pl, ph = np.quantile(rr, [.025, .975]); pred = f"[{100*pl:.2f}, {100*ph:.2f}]"
        else: pred = "--"
        risk = f"{P(r.test_risk)}\\% ({k})"; cpt = f"[{100*lo:.2f}, {100*hi:.2f}]"
    else:
        risk, cpt, pred = f"{r.test_risk:.4f}", "--", "--"
    alab = "fixed" if a == "fixed z>3" else f"{float(a):.2f}"
    body.append(f"{alab} & {lossn if a != 'fixed z>3' else '$z>3$'} & {float(r.lam):.3f} & {risk} & {cpt} & {pred} & {k_rec} ({100*r.overpricing_recall:.1f}\\%) \\\\")
tab("table16_crc", "Conformal risk control on the test period: $n=2{,}470$ hard-rule-compliant legitimate invoices and 40 collusive over-pricing tasks; calibration $n=2{,}054$. CP: Clopper--Pearson 95\\% interval. Predictive range: 95\\% range of one realisation under exchangeability (Monte Carlo). Weighted-loss risk is a normalised capped exposure, not a share of value.",
    "tab:crc", "llrrrrr", "$\\alpha$ & Loss & $\\hat\\lambda$ & Realised risk & CP & Predictive range & Recall (of 40)", body, star=True)

# ---------- Table XVII: audit ----------
NAMES = {"ANY_EXCEPTION": "Any exception", "R3_authority": "R3 authority", "S1_soft_price": "S1 price ($\\alpha=0.05$)",
         "R7_split": "R7 split pattern", "R6_budget": "R6 budget", "R1_vendor": "R1 vendor master",
         "R4_match": "R4 match", "R5_dup_fuzzy": "R5 duplicate"}
body = []
for _, r in AUD.iterrows():
    lo, hi = cp(int(r.n), 20000)
    body.append(f"{NAMES[r.rule]} & {int(r.n):,} & {P(r.rate)} [{100*lo:.2f}, {100*hi:.2f}] & {r.eur:,.0f} \\\\")
check("audit total firings", float(AUD[AUD.rule != 'ANY_EXCEPTION'].n.sum()), 7032.0)
tab("table17_audit", "Natural audit of 20,000 real test-period invoices replayed as recorded. Counts are exceptions against the \\emph{mined} policy, not fraud labels; no exception was verified with the company. Rules overlap (7,032 firings on 6,079 invoices).",
    "tab:audit", "lrrr", "Rule & Invoices & Rate \\% [CP 95\\%] & Anon.\\ EUR", body, star=False)

# ---------- Appendix A-II: per-typology prevention, all streams ----------
CATS = ["UNAPPROVED_VENDOR", "SOD", "NO_GOODS_RECEIPT", "AUTHORITY", "INFLATED_MATCH", "DUPLICATE", "BUDGET_PLAIN", "BUDGET_ENCUMBRANCE",
        "SPLIT", "PROMPT_INJECTION", "OVERPRICED_COLLUSIVE", "SPLIT_ADAPTIVE"]
body = []
for c in CATS:
    cells = [c.replace("_", "\\_")]
    for st, _ in STREAMS:
        rate = BEH[(BEH.stream == st) & (BEH.category == c)].harmful_proposal_rate.iloc[0]; n = round(rate * 40)
        x = CAT[(CAT.stream == st) & (CAT.category == c) & (CAT.guard == "GuardLedger (full)")]
        cells.append("--" if not len(x) else f"{100*x.prevention.iloc[0]:.1f} ({round(x.prevention.iloc[0]*n)}/{n})")
    body.append(" & ".join(cells) + " \\\\")
tab("tableA2_typology", "GuardLedger prevention by typology and stream, \\% (prevented/harmful proposals). --: no harmful proposal in that stream.",
    "tab:typology", "lrrrrr", "Typology & Approve-all & Qwen plain & Qwen policy & Phi plain & Phi policy", body)

# ---------- Appendix A-III: budgets ----------
body = [f"{k.replace('&', '\\&') if k!='nan' else '(missing value)'} & {v:,.2f} \\\\" for k, v in sorted(POL["budgets"].items(), key=lambda x: x[0].lower())]
tab("tableA3_budgets", "Mined monthly spend-area budgets (anonymised EUR), Eq.~(5).", "tab:budgets", "lr", "Spend area & Budget", body, star=False)

# ---------- consistency checks against stated text values ----------
TXT = {"GL approve-all prevention": (row("approve_all", "GuardLedger (full)").prevention, 0.9125),
       "GL Qwen plain prevention": (row(STREAMS[1][0], "GuardLedger (full)").prevention, 338 / 380),
       "GL Qwen policy prevention": (row(STREAMS[2][0], "GuardLedger (full)").prevention, 185 / 224),
       "GL Phi plain prevention": (row(STREAMS[3][0], "GuardLedger (full)").prevention, 384 / 426),
       "GL Phi policy prevention": (row(STREAMS[4][0], "GuardLedger (full)").prevention, 362 / 404),
       "judge Qwen": (row(STREAMS[1][0], judge(STREAMS[1][0])[0]).prevention, 89 / 380),
       "judge Phi": (row(STREAMS[3][0], judge(STREAMS[3][0])[0]).prevention, 42 / 426),
       "EUR reduction Qwen plain": (1 - row(STREAMS[1][0], "GuardLedger (full)").usd_leaked / row(STREAMS[1][0], "NoGuard").usd_leaked, 0.957584),
       "EUR reduction Phi plain": (1 - row(STREAMS[3][0], "GuardLedger (full)").usd_leaked / row(STREAMS[3][0], "NoGuard").usd_leaked, 0.962730),
       "Unguarded HER Qwen": (row(STREAMS[1][0], "NoGuard").harm_exec_rate_all, 380 / 480),
       "Unguarded HER Phi": (row(STREAMS[3][0], "NoGuard").harm_exec_rate_all, 426 / 480),
       "lambda*": (MAN["lambda_star"], 1.5892759073403064), "amt_mean_cal": (MAN["amt_mean_cal"], 2402.1168451801364),
       "CRC count risk 0.05": (float(CRC[(CRC.alpha == '0.05') & (CRC.loss == 'count')].test_risk.iloc[0]), 119 / 2470),
       "audit any": (float(AUD[AUD.rule == 'ANY_EXCEPTION'].rate.iloc[0]), 6079 / 20000)}
for k, (g, e) in TXT.items(): check(k, g, e, 5e-6)
rep = ["| Check | Value | Expected | OK |", "|---|---|---|---|"] + [f"| {n} | {g:.6g} | {e:.6g} | {'PASS' if ok else 'FAIL'} |" for n, g, e, ok in CHECKS]
open(f"{OUT}/verification_report.md", "w").write(f"# Table value verification\n\n{sum(c[3] for c in CHECKS)}/{len(CHECKS)} checks passed.\n\n" + "\n".join(rep) + "\n")
print(sum(c[3] for c in CHECKS), "/", len(CHECKS), "checks passed;", sorted(os.listdir(OUT)))

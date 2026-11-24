"""Build tables/tab_heldout.tex and tables/tab_extra.tex from checks/rev_results (no number typed by hand)."""
import csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "rev_results"); A = os.path.join(R, "passA_gpu"); T = os.path.join(HERE, "..", "tables")
rd = lambda p: list(csv.DictReader(open(p)))
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import binomtest
def pc(x, d=1):
    q = Decimal(1).scaleb(-d)
    return str((Decimal(str(x)) * 100).quantize(q, rounding=ROUND_HALF_UP))

def pval(p, a=None, b=None):
    p = float(p)
    if a is not None:   # recompute the exact two-sided McNemar p from the discordant counts and check it against the file
        q = binomtest(a, a + b, 0.5).pvalue if a + b else 1.0
        assert abs(round(q, 5) - p) < 1e-9, (a, b, q, p); p = q
    if p < 1e-6: m, e = f"{p:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
    if p >= 0.01: return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e"); return f"${int(m)}\\times10^{{{int(e)}}}$"

# ---------------- held-out typologies ----------------
CATS = ["PASS_THROUGH_MARKUP", "CROSS_VENDOR_DOUBLE_BILLING", "PERSONAL_PURCHASE", "SELF_RECEIPTED_PURCHASE", "CROSS_REQUESTER_SPLIT"]
SHORT = {"PASS_THROUGH_MARKUP": "Mark-up", "CROSS_VENDOR_DOUBLE_BILLING": "Double bill", "PERSONAL_PURCHASE": "Personal",
         "SELF_RECEIPTED_PURCHASE": "Self-receipt", "CROSS_REQUESTER_SPLIT": "Cross-req.\\ split"}
percat = {(r["category"], r["guard"]): r["prevention"] for r in rd(os.path.join(R, "passB_heldout_per_category.csv"))}
pooled_a = rd(os.path.join(A, "heldout_pooled_and_paired.csv"))
pooled = {r["guard"]: r for r in pooled_a if r["prevented"]}
paired = {r["guard"]: r for r in pooled_a if r["n_tasks"]}
# pass A (GPU) and pass B pooled results must agree
pb = {r["guard"]: r for r in rd(os.path.join(R, "passB_heldout_pooled_and_paired.csv"))}
for g, r in pooled.items():
    assert int(float(r["prevented"])) == int(pb[g]["prevented"]), g
judge = [r for r in rd(os.path.join(R, "passB_independent_judges_summary.csv")) if r["stream"] == "heldout|approve_all"][0]

ROWS = [("GuardLedger (full)", "GuardLedger"), ("GL-hard (no soft layer)", "GL-hard"), ("GL-hard + fixed z>3 (no CRC)", "GL-hard + $z>3$"),
        ("AgentSpec-style", "Stateless rules"), ("IsolationForest", "Isolation Forest"), ("IsolationForest (conformal)", "IF (conformal)"),
        ("LOF (conformal)", "LOF (conformal)"), ("Fused S1+IF (conformal)", "Fused (conformal)"), ("GL-hard + IF (conformal)", "GL-hard + IF (conf.)"),
        ("GL-hard + fused (conformal)", "GL-hard + fused (conf.)")]
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Held-out typologies (approve-all stream, 40 tasks each, 200 in all), defined after the rules were frozen. Per-typology prevention (\%) is given where the per-typology file was retained. Pooled prevention carries Clopper--Pearson 95\% intervals; the paired columns count tasks prevented only by GuardLedger and only by the other guard, with the exact McNemar $p$-value.}",
     r"\label{tab:heldout}", r"\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{adjustbox}{max width=\textwidth}",
     r"\begin{tabular}{lccccccc}", r"\hline",
     "Guard & " + " & ".join(SHORT[c] for c in CATS) + r" & Pooled \% [CP 95\%] (n/200) & GL only / other only, $p$ \\", r"\hline"]
for g, lab in ROWS:
    cells = [pc(percat[(c, g)]) if (c, g) in percat else "--" for c in CATS]
    r = pooled[g]
    pool = f"{pc(r['prevention'])} [{pc(r['cp_lo'])}, {pc(r['cp_hi'])}] ({int(float(r['prevented']))})"
    pr = "--" if g == "GuardLedger (full)" else f"{int(float(paired[g]['first_only']))} / {int(float(paired[g]['second_only']))}, {pval(paired[g]['p_value'], int(float(paired[g]['first_only'])), int(float(paired[g]['second_only'])))}"
    L.append(f"{lab} & " + " & ".join(cells) + f" & {pool} & {pr} \\\\")
L.append(f"Independent 14B judge & -- & -- & -- & -- & -- & {pc(judge['prevention'])} [{pc(judge['prev_lo'])}, {pc(judge['prev_hi'])}]$^\\dagger$ & -- \\\\")
L += [r"\hline", r"\end{tabular}", r"\end{adjustbox}",
      r"\par\smallskip\footnotesize $^\dagger$95\% bootstrap interval; the other pooled intervals are Clopper--Pearson.",
      r"\end{table*}"]
open(os.path.join(T, "tab_heldout.tex"), "w").write("\n".join(L) + "\n")

# ---------------- independent judge, 14B agent, seeds ----------------
SN = {"approve_all": "Approve-all", "qwen2.5-7b-instruct|plain": "Qwen2.5-7B plain", "phi-3.5-mini-instruct|plain": "Phi-3.5-mini plain"}
J = rd(os.path.join(R, "passB_independent_judges_summary.csv"))
Ag = rd(os.path.join(R, "passB_extra_agent_q14_summary.csv"))
S = rd(os.path.join(A, "seeds_approve_all_summary_compact.csv"))
RI = rd(os.path.join(A, "seeds_approve_all_rule_independent_compact.csv"))
L = [r"\begin{table*}[!t]", r"\centering",
     r"\caption{Supplementary experiments. (a) Independent judge (Qwen2.5-14B-Instruct, 4-bit NF4, shown the approvals already proposed in the episode). (b) Qwen2.5-14B-Instruct as agent. (c) Approve-all stream over five further benchmark seeds. Prevention (\%) with 95\% bootstrap intervals, and in (a) also the false-hold rate (FHR); HER with all violation tasks as denominator.}",
     r"\label{tab:extra}", r"\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{adjustbox}{max width=\textwidth}",
     r"\begin{tabular}{llcccc}", r"\hline",
     r"\multicolumn{6}{l}{\emph{(a) Independent judge}} \\",
     r"Stream & Mode & Prevention & FHR & HER & Unparseable / judged \\", r"\hline"]
for r in J:
    if r["stream"] not in SN: continue
    mode = "fail-open" if "fail-open" in r["guard"] else "fail-closed"
    L.append(f"{SN[r['stream']]} & {mode} & {pc(r['prevention'],2)} [{pc(r['prev_lo'],2)}, {pc(r['prev_hi'],2)}] & {pc(r['fpr_legit'],2)} [{pc(r['fpr_lo'],2)}, {pc(r['fpr_hi'],2)}] & {pc(r['harm_exec_rate_all'],2)} & {r['unparseable']} / {r['judged']} \\\\")
L += [r"\hline", r"\multicolumn{6}{l}{\emph{(b) Qwen2.5-14B-Instruct as agent}} \\",
      r"Stream (harmful proposals) & Guard & Prevention & FHR & HER & \\", r"\hline"]
AN = {"bench|q14|plain": "Benchmark plain", "bench|q14|policy_prompt": "Benchmark policy", "heldout|q14|plain": "Held-out plain"}
GN = {"NoGuard": "No guard", "GuardLedger (full)": "GuardLedger", "GL-hard (no soft layer)": "GL-hard", "IsolationForest": "Isolation Forest",
      "GL-hard + IF (conformal)": "GL-hard + IF (conf.)", "AgentSpec-style": "Stateless rules"}
for r in Ag:
    if r["guard"] not in GN: continue
    fhr = "--" if r["fpr_legit"] == "" else f"{pc(r['fpr_legit'],2)}"
    ci = "" if r["guard"] == "NoGuard" else f" [{pc(r['prev_lo'],2)}, {pc(r['prev_hi'],2)}]"
    L.append(f"{AN[r['stream']]} ({r['n_harm_proposed']}) & {GN[r['guard']]} & {pc(r['prevention'],2)}{ci} & {fhr} & {pc(r['harm_exec_rate_all'],2)} & \\\\")
L += [r"\hline", r"\multicolumn{6}{l}{\emph{(c) Five further benchmark seeds, approve-all stream}} \\",
      r"Guard & & Prevention: range (mean) & FHR range & Rule-indep.\ prevented (of 80 per seed) & Pooled \\", r"\hline"]
SG = [("GuardLedger (full)", "GuardLedger"), ("GL-hard (no soft layer)", "GL-hard"), ("GL-hard + IF (conformal)", "GL-hard + IF (conf.)"),
      ("IsolationForest", "Isolation Forest"), ("AgentSpec-style", "Stateless rules")]
for g, lab in SG:
    p = [float(r["prevention"]) for r in S if r["guard"] == g]; f = [float(r["fpr_legit"]) for r in S if r["guard"] == g]
    ri = [int(r["prevented"]) for r in RI if r["guard"] == g]
    L.append(f"{lab} & & {pc(min(p),2)}--{pc(max(p),2)} ({pc(st.mean(p),2)}) & {pc(min(f),2)}--{pc(max(f),2)} & {min(ri)}--{max(ri)} & {sum(ri)}/400 \\\\")
L += [r"\hline", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
open(os.path.join(T, "tab_extra.tex"), "w").write("\n".join(L) + "\n")
print("wrote tab_heldout.tex and tab_extra.tex")

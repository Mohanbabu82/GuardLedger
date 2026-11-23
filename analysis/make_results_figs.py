"""Final result figures (Figs. 3-8) for the GuardLedger IEEE Access manuscript.
Reads only the V2 result CSVs.  Usage: python make_results_figs_final.py <res_v2_qwen> <res_v2_phi> <outdir>"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta, betabinom

QW, PH, OUT = (sys.argv[1:4] + [None] * 3)[:3]
QW = QW or str(ROOT/"results/v2_qwen"); PH = PH or str(ROOT/"results/v2_phi"); OUT = OUT or str(ROOT/"figures")
os.makedirs(OUT, exist_ok=True)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", violet="#6250d6",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", muted="#9a9893")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)

SUM_A = pd.read_csv(f"{QW}/expA_summary.csv"); CAT_A = pd.read_csv(f"{QW}/expA_per_category.csv")
SUM_D = pd.concat([pd.read_csv(f"{d}/expD_llm_summary.csv") for d in (QW, PH)], ignore_index=True)
CAT_D = pd.concat([pd.read_csv(f"{d}/expD_llm_per_category.csv") for d in (QW, PH)], ignore_index=True)
BEH = pd.concat([pd.read_csv(f"{d}/agent_behaviour.csv") for d in (QW, PH)], ignore_index=True).drop_duplicates()
CRC = pd.read_csv(f"{QW}/expB_crc_validity.csv")

CAT_ORDER = ["UNAPPROVED_VENDOR", "SOD", "NO_GOODS_RECEIPT", "AUTHORITY", "INFLATED_MATCH", "DUPLICATE", "BUDGET_PLAIN",
             "BUDGET_ENCUMBRANCE", "SPLIT", "PROMPT_INJECTION", "OVERPRICED_COLLUSIVE", "SPLIT_ADAPTIVE"]
CAT_LABEL = {"UNAPPROVED_VENDOR": "Unapproved vendor", "SOD": "Segregation of duties", "NO_GOODS_RECEIPT": "No goods receipt",
             "AUTHORITY": "Approval authority", "INFLATED_MATCH": "Inflated match", "DUPLICATE": "Duplicate invoice",
             "BUDGET_PLAIN": "Budget overrun", "BUDGET_ENCUMBRANCE": "Encumbrance overrun", "SPLIT": "Split purchase",
             "PROMPT_INJECTION": "Prompt injection", "OVERPRICED_COLLUSIVE": "Collusive over-pricing†",
             "SPLIT_ADAPTIVE": "Adaptive split†"}
GUARDS = [("NoGuard", "No guard"), ("AgentSpec-style", "Stateless rules"), ("IsolationForest", "Isolation Forest"),
          ("JUDGE", "LLM judge"), ("GL-hard - quantitative", "GL-hard − quant."), ("GL-hard - encumbrance", "GL-hard − encumb."),
          ("GL-hard - trajectory", "GL-hard − traj."), ("GL-hard - provenance", "GL-hard − prov."),
          ("GL-hard (no soft layer)", "GL-hard"), ("GL-hard + fixed z>3 (no CRC)", "GL-hard + z>3"), ("GuardLedger (full)", "GuardLedger")]

def pct(v):
    p = 100 * v
    return f"{p:.0f}" if abs(p - round(p)) < 1e-6 else f"{p:.1f}"

def heatmap(d, name, title, note=None):
    d = d.copy(); d.loc[d["guard"].str.startswith("LLM-as-judge"), "guard"] = "JUDGE"
    g = [(k, l) for k, l in GUARDS if k in set(d["guard"])]
    pv = d.pivot_table(index="category", columns="guard", values="prevention").reindex(CAT_ORDER)[[k for k, _ in g]]
    dropped = [c for c in pv.index if pv.loc[c].isna().all()]; pv = pv.drop(index=dropped)
    fig, ax = plt.subplots(figsize=(7.16, 0.25 * len(pv) + 1.25))
    im = ax.imshow(pv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(g))); ax.set_xticklabels([l for _, l in g], rotation=35, ha="right")
    ax.set_yticks(range(len(pv))); ax.set_yticklabels([CAT_LABEL[c] for c in pv.index])
    ax.set_xticks(np.arange(-.5, len(g)), minor=True); ax.set_yticks(np.arange(-.5, len(pv)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="both", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, "–" if np.isnan(v) else pct(v), ha="center", va="center", fontsize=6.3,
                    color=C["muted"] if np.isnan(v) else ("white" if v > 0.6 else C["ink"]))
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01); cb.outline.set_visible(False)
    cb.set_ticks([0, .25, .5, .75, 1]); cb.set_ticklabels(["0", "25", "50", "75", "100"]); cb.set_label("Prevention (%)")
    ax.set_title(title, fontsize=8, loc="left")
    # footnotes (dagger meaning, omitted rows, judge parse failures) are given in the caption
    save(fig, name)

# Fig 3-5 heatmaps
heatmap(CAT_A, "fig3_heatmap_approve_all", "Prevention by typology and guard: approve-all stream (40 tasks per typology)")
heatmap(CAT_D[CAT_D["stream"] == "qwen2.5-7b-instruct|plain"], "fig4_heatmap_qwen_plain",
        "Prevention by typology and guard: Qwen2.5-7B-Instruct, plain prompt")
heatmap(CAT_D[CAT_D["stream"] == "phi-3.5-mini-instruct|plain"], "fig5_heatmap_phi_plain",
        "Prevention by typology and guard: Phi-3.5-mini-instruct, plain prompt",
        note="LLM judge: 141/848 unparseable outputs mapped to ALLOW (fail-open).")

# Fig 6 agent behaviour (typology x stream matrix)
streams = [("qwen2.5-7b-instruct|plain", "Qwen\nplain"), ("qwen2.5-7b-instruct|policy_prompt", "Qwen\npolicy"),
           ("phi-3.5-mini-instruct|plain", "Phi\nplain"), ("phi-3.5-mini-instruct|policy_prompt", "Phi\npolicy")]
pv = BEH.pivot_table(index="category", columns="stream", values="harmful_proposal_rate").reindex(CAT_ORDER)[[s for s, _ in streams]]
fig, ax = plt.subplots(figsize=(3.5, 3.55))
im = ax.imshow(pv.values, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in streams]); ax.xaxis.tick_top()
ax.set_yticks(range(len(pv))); ax.set_yticklabels([CAT_LABEL[c] for c in pv.index])
ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, len(pv)), minor=True)
ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="both", length=0)
for s in ax.spines.values(): s.set_visible(False)
for i in range(pv.shape[0]):
    for j in range(4):
        v = pv.values[i, j]; ax.text(j, i, pct(v), ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else C["ink"])
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.outline.set_visible(False)
cb.set_ticks([0, .5, 1]); cb.set_ticklabels(["0", "50", "100"]); cb.set_label("Harmful-proposal rate (%)")
save(fig, "fig6_agent_behaviour")

# Fig 7 trade-offs
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.65), gridspec_kw=dict(wspace=0.42))
ax = axs[0]; a = SUM_A.set_index("guard")
pts = [("NoGuard", "No guard", C["muted"], "o"), ("AgentSpec-style", "Stateless rules", C["yellow"], "s"),
       ("IsolationForest", "Isolation Forest", C["violet"], "D"), ("GL-hard (no soft layer)", "GL-hard", C["aqua"], "^"),
       ("GL-hard + fixed z>3 (no CRC)", "GL-hard + z>3", C["orange"], "v"), ("GuardLedger (full)", "GuardLedger", C["blue"], "o")]
off = {"No guard": (5, 3), "Stateless rules": (5, -9), "Isolation Forest": (-4, 6), "GL-hard": (5, 2), "GL-hard + z>3": (5, 3), "GuardLedger": (5, 4)}
for g, lab, col, mk in pts:
    x, y = a.loc[g, "usd_held"] / 1e3, a.loc[g, "usd_leaked"] / 1e6
    ax.scatter(x, y, s=34, color=col, marker=mk, edgecolor="white", lw=0.8, zorder=3)
    dx, dy = off[lab]; ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.6,
                                   ha="right" if lab == "Isolation Forest" else "left")
ax.set_xlabel("Legitimate amount held (thousand anonymised €)"); ax.set_ylabel("Harmful amount executed\n(million anonymised €)")
ax.set_xlim(-10, 175); ax.set_ylim(-0.3, 6.6); ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.set_title("(a) Exposure trade-off, approve-all stream", fontsize=8, loc="left")
ax = axs[1]
for st, lab_m, mk in (("qwen2.5-7b-instruct|plain", "Qwen", "o"), ("phi-3.5-mini-instruct|plain", "Phi", "s")):
    d = SUM_D[SUM_D["stream"] == st].set_index("guard")
    for g, lab, col in (("AgentSpec-style", "Stateless rules", C["yellow"]), ("GL-hard (no soft layer)", "GL-hard", C["aqua"]),
                        ("GuardLedger (full)", "GuardLedger", C["blue"]), ("IsolationForest", "Isolation Forest", C["violet"]),
                        ([k for k in d.index if k.startswith("LLM-as-judge")][0], "LLM judge (amortised)", C["orange"])):
        ax.scatter(d.loc[g, "lat_p50_us"], 100 * d.loc[g, "prevention"], s=30, color=col, marker=mk, edgecolor="white", lw=0.8, zorder=3,
                   label=f"{lab}" if st.startswith("qwen") else None)
ax.set_xscale("log"); ax.set_xlim(20, 1e7); ax.set_ylim(0, 100)
ax.set_xlabel("Median time per check (µs, log scale)"); ax.set_ylabel("Prevention (%)")
ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.set_title("(b) Prevention vs time per check (○ Qwen, □ Phi; plain)", fontsize=8, loc="left")
ax.legend(frameon=False, fontsize=6.3, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "fig7_tradeoffs")

# Fig 8 CRC with exact intervals and exchangeable predictive band
def cp(k, n): return (beta.ppf(.025, k, n - k + 1) if k else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0)
n_cal, n_te, n_ov = 2054, 2470, 40
c = CRC[CRC["alpha"] != "fixed z>3"].copy(); c["alpha"] = c["alpha"].astype(float)
cnt = c[c["loss"] == "count"].sort_values("alpha"); wt = c[c["loss"] == "dollar"].sort_values("alpha")
rng = np.random.default_rng(0)
band = []
for al in cnt["alpha"]:
    k = int(np.floor(al * (n_cal + 1) - 1 + 1e-12))  # exact beta-binomial predictive range (revision R2-3)
    band.append(np.array(betabinom(n_te, k + 1, n_cal - k).ppf([.025, .975])) / n_te)
band = np.array(band)
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw=dict(wspace=0.3))
ax = axs[0]
ax.fill_between(cnt["alpha"], 100 * band[:, 0], 100 * band[:, 1], color=C["blue"], alpha=0.13, lw=0, label="Exchangeable 95% range (count)")
ax.plot([0, .21], [0, 21], "--", color=C["muted"], lw=0.9, label="Target (risk = α)")
k_c = np.rint(cnt["test_risk"] * n_te).astype(int)
ci = np.array([cp(k, n_te) for k in k_c])
ax.errorbar(cnt["alpha"], 100 * cnt["test_risk"], yerr=[100 * (cnt["test_risk"] - ci[:, 0]), 100 * (ci[:, 1] - cnt["test_risk"])],
            fmt="o-", color=C["blue"], ms=3.5, lw=1.4, capsize=2, label="Count loss (deployed), CP 95%")
ax.plot(wt["alpha"], 100 * wt["test_risk"], "s--", color=C["orange"], ms=3.2, lw=1.2, label="Weighted loss (normalised exposure ×100)")
ax.set_xlabel("Target α"); ax.set_ylabel("Realised risk on test period (%)"); ax.set_xlim(0, 0.21); ax.set_ylim(0, 23)
ax.set_xticks([0.01, 0.05, 0.10, 0.20]); ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.0, loc="upper left"); ax.set_title("(a) S1 false-escalation risk, n = 2,470", fontsize=8, loc="left")
ax = axs[1]
for d, col, mk, lab in ((cnt, C["blue"], "o-", "Count-loss threshold"), (wt, C["orange"], "s--", "Weighted-loss threshold")):
    k = np.rint(d["overpricing_recall"] * n_ov).astype(int); ci = np.array([cp(x, n_ov) for x in k])
    ax.errorbar(d["alpha"], 100 * d["overpricing_recall"], yerr=[100 * (d["overpricing_recall"] - ci[:, 0]), 100 * (ci[:, 1] - d["overpricing_recall"])],
                fmt=mk, color=col, ms=3.3, lw=1.2, capsize=2, label=lab)
ax.set_xlabel("Target α"); ax.set_ylabel("Over-pricing recall (%), n = 40"); ax.set_xlim(0, 0.21); ax.set_ylim(0, 80)
ax.set_xticks([0.01, 0.05, 0.10, 0.20]); ax.yaxis.grid(True, color=C["grid"], lw=0.5); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.0, loc="upper left"); ax.set_title("(b) Detection of collusive over-pricing (CP 95%)", fontsize=8, loc="left")
save(fig, "fig8_crc")
print(sorted(os.listdir(OUT)))

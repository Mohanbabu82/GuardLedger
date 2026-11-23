"""Figures 1-2 (system model, architecture) of the GuardLedger manuscript; writes PDF and PNG into figures/."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"pdf.fonttype":42})
INK="#222222"; MUTED="#555555"
F_TR="#e8eef6"; F_UN="#f7e9e4"; F_GD="#eef3ea"; F_OUT=["#e7f2e9","#f8f0dc","#f6e3e3"]; F_OFF="#f2f2f2"
def box(ax,x,y,w,h,txt,fc="white",ec=INK,lw=0.8,fs=7,bold=False,ls="-",ha="center"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.008,rounding_size=0.012",fc=fc,ec=ec,lw=lw,ls=ls))
    ax.text(x+(w/2 if ha=="center" else 0.008),y+h/2,txt,ha=ha,va="center",fontsize=fs,color=INK,weight="bold" if bold else "normal",linespacing=1.15)
def arr(ax,p,q,ls="-",col=INK,cs="arc3",lw=0.8,ms=7):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=col,ls=ls,connectionstyle=cs,shrinkA=1,shrinkB=1))
def save(fig,name):
    fig.savefig(str(ROOT/"figures"/name)+".pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(str(ROOT/"figures"/name)+".png",dpi=300,bbox_inches="tight",pad_inches=0.02); plt.close(fig)

# ---------- Fig. 1 system model and trust boundary (double column)
fig,ax=plt.subplots(figsize=(7.16,2.6)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.add_patch(Rectangle((0.005,0.52),0.99,0.46,fc=F_TR,ec=MUTED,lw=0.6,ls="--"))
ax.text(0.012,0.965,"TRUSTED: system of record, mined policy, guard",fontsize=6.2,color=MUTED,va="top",weight="bold")
ax.add_patch(Rectangle((0.005,0.02),0.44,0.45,fc=F_UN,ec=MUTED,lw=0.6,ls="--"))
ax.text(0.012,0.455,"UNTRUSTED: counterparty text, agent output",fontsize=6.2,color=MUTED,va="top",weight="bold")
xs=[0.02,0.175,0.33]; labs=["Create PO item\n$m^{po}$, $u^{po}$","Goods receipt\n$m^{gr}$, $u^{gr}$","Invoice receipt\n$m$, $v$"]
for x,l in zip(xs,labs): box(ax,x,0.62,0.135,0.17,l,fs=5.9)
for a,b in [(0.155,0.175),(0.31,0.33)]: arr(ax,(a,0.705),(b,0.705))
box(ax,0.52,0.585,0.35,0.24,"Ledger state as of time $t$\nactuals $A_a(t)$; open commitments $C_a(t;c)$;\nvendor history $H_v(t)$; vendor master $\\mathcal{V}$;\nmined authority $\\ell(u)$; budgets $\\beta_a$",fs=5.9)
arr(ax,(0.465,0.705),(0.52,0.705)); ax.text(0.492,0.66,"event log",fontsize=5.5,color=MUTED,ha="center")
box(ax,0.02,0.10,0.14,0.20,"Invoice document\nfree text (memo)",fs=6.0)
box(ax,0.22,0.10,0.19,0.20,"LLM agent proposes\napprove_invoice($\\hat m,\\hat v,\\hat u$)",fs=5.9)
arr(ax,(0.16,0.20),(0.22,0.20))
arr(ax,(0.40,0.62),(0.33,0.30),ls=(0,(3,2)),col=MUTED)
ax.text(0.385,0.49,"packet shown to agent",fontsize=5.5,color=MUTED,ha="left")
box(ax,0.50,0.08,0.22,0.24,"GuardLedger check$(p,a,E)$\nhard rules R1–R8, soft S1",fc=F_GD,lw=1.0,fs=6.2)
arr(ax,(0.41,0.20),(0.50,0.20))
arr(ax,(0.68,0.585),(0.64,0.32))
ax.text(0.585,0.45,"trusted packet fields",fontsize=5.5,color=MUTED,ha="right")
outs=[("ALLOW: execute;\nappend to episode $E$",0.35),("ESCALATE: hold for\nhuman review",0.195),("BLOCK: refuse",0.04)]
for (t,y),c in zip(outs,F_OUT):
    box(ax,0.80,y,0.185,0.125,t,fc=c,fs=5.9); arr(ax,(0.72,0.20),(0.80,y+0.062))
save(fig,"fig1_system_model")

# ---------- Fig. 2 architecture (double column)
fig,ax=plt.subplots(figsize=(7.16,3.5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
ax.add_patch(Rectangle((0.005,0.72),0.99,0.27,fc=F_OFF,ec=MUTED,lw=0.6,ls="--"))
ax.text(0.012,0.98,"OFFLINE (once, before deployment)",fontsize=6.2,color=MUTED,va="top",weight="bold")
box(ax,0.02,0.75,0.13,0.17,"Event log,\ntraining split",fs=6.0)
box(ax,0.20,0.75,0.33,0.17,"Policy mining (Alg. 3): tier limits $L_k$,\nuser authority $\\ell(u)$, budgets $\\beta_a$,\nprice-score statistics; vendor master $\\mathcal{V}$",fs=5.9)
box(ax,0.58,0.75,0.17,0.17,"Calibration split:\nhard-rule-compliant\nlegitimate packets",fs=5.9)
box(ax,0.80,0.75,0.185,0.17,"CRC calibration (Alg. 4):\nthreshold $\\hat\\lambda$ for target $\\alpha$",fs=5.9)
arr(ax,(0.15,0.835),(0.20,0.835)); arr(ax,(0.53,0.835),(0.58,0.835)); arr(ax,(0.75,0.835),(0.80,0.835))
ax.add_patch(Rectangle((0.005,0.01),0.99,0.68,fc="white",ec=MUTED,lw=0.6,ls="--"))
ax.text(0.012,0.675,"ONLINE (every proposed tool call)",fontsize=6.2,color=MUTED,va="top",weight="bold")
box(ax,0.02,0.52,0.12,0.08,"Agent proposal $a$",fs=5.8)
box(ax,0.02,0.39,0.12,0.08,"Normalise\narguments",fs=5.8)
box(ax,0.02,0.22,0.12,0.12,"Packet builder\n(Alg. 2): ledger\nqueries as of $t$",fs=5.6)
box(ax,0.02,0.04,0.12,0.12,"Episode state $E$\n(approvals already\nexecuted)",fs=5.6)
arr(ax,(0.08,0.52),(0.08,0.47)); arr(ax,(0.08,0.39),(0.08,0.34))
gx=0.17; gw=0.47
rules=[("Provenance — R8: amount, vendor and approver equal\nthe ledger record (otherwise BLOCK)",0.53,F_GD),
("Master data — R1 vendor master (BLOCK); R2 segregation\nof duties (ESCALATE); R4 goods receipt present (BLOCK)",0.42,F_GD),
("Quantitative — R3 authority (ESCALATE); R4 2/3-way\nmatch (BLOCK); R6 commitment-aware budget (ESCALATE)",0.31,F_GD),
("Trajectory — R5 duplicate, exact or fuzzy (BLOCK);\nR7 split purchase, 14-day window (ESCALATE)",0.20,F_GD),
("Soft layer — S1 price score $s>\\hat\\lambda$, threshold\nfrom Alg. 4 (ESCALATE)",0.09,"#f3eedf")]
for t,y,c in rules:
    box(ax,gx,y-0.042,gw,0.088,t,fc=c,fs=5.5,ha="left"); arr(ax,(0.14,0.28),(gx,y),lw=0.5,col=MUTED)
arr(ax,(0.365,0.75),(0.40,0.578),lw=0.6,col=MUTED,ls=(0,(3,2))); ax.text(0.385,0.655,"frozen policy",fontsize=5.5,color=MUTED)
box(ax,0.67,0.29,0.11,0.13,"Aggregate:\nmaximum\nseverity",fs=5.8)
for t,y,c in rules: arr(ax,(gx+gw,y),(0.67,0.355),lw=0.5,col=MUTED)
outs=[("ALLOW → execute;\nupdate episode $E$",0.47),("ESCALATE → review",0.31),("BLOCK → refuse",0.15)]
for (t,y),c in zip(outs,F_OUT):
    box(ax,0.83,y,0.155,0.11,t,fc=c,fs=5.7); arr(ax,(0.78,0.355),(0.83,y+0.055))
box(ax,0.67,0.04,0.315,0.09,"Decision record: verdict, fired rules, latency",fs=5.6)
arr(ax,(0.725,0.29),(0.725,0.13),lw=0.5,col=MUTED)
arr(ax,(0.14,0.11),(gx,0.20),lw=0.5,col=MUTED,ls=(0,(3,2)))
arr(ax,(0.14,0.13),(gx,0.285),lw=0.5,col=MUTED,ls=(0,(3,2)))
save(fig,"fig2_architecture")

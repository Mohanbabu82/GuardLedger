"""Rule-matched vs rule-independent prevention, recomputed from the V2 per-typology files (Reviewer 1, comment 1)."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
from scipy.stats import beta
from decimal import Decimal, ROUND_HALF_UP
Q, P = str(ROOT/'results/v2_qwen'), str(ROOT/'results/v2_phi')
def R(x, d=1): return str(Decimal(repr(round(float(100 * x), 9))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
pc = pd.concat([pd.read_csv(f'{Q}/expA_per_category.csv'), pd.read_csv(f'{Q}/expD_llm_per_category.csv'), pd.read_csv(f'{P}/expD_llm_per_category.csv')]).drop_duplicates()
ab = pd.concat([pd.read_csv(f'{Q}/agent_behaviour.csv'), pd.read_csv(f'{P}/agent_behaviour.csv')]).drop_duplicates()
ab['nhp'] = (ab.harmful_proposal_rate * ab.n).round().astype(int)
m = pc.merge(ab[['stream', 'category', 'nhp']], on=['stream', 'category'])
m['k'] = (m.prevention * m.nhp).round().astype(int)
m['grp'] = m.category.isin(['OVERPRICED_COLLUSIVE', 'SPLIT_ADAPTIVE']).map({True: 'ind', False: 'rule'})
g = m.groupby(['stream', 'guard', 'grp'])[['k', 'nhp']].sum().reset_index()
streams = [('approve_all', 'Approve-all'), ('qwen2.5-7b-instruct|plain', 'Qwen plain'), ('qwen2.5-7b-instruct|policy_prompt', 'Qwen policy'), ('phi-3.5-mini-instruct|plain', 'Phi plain'), ('phi-3.5-mini-instruct|policy_prompt', 'Phi policy')]
guards = [('GuardLedger (full)', 'GuardLedger'), ('GL-hard (no soft layer)', 'GL-hard'), ('GL-hard + fixed z>3 (no CRC)', 'GL-hard + $z>3$'), ('IsolationForest', 'Isolation Forest'), ('AgentSpec-style', 'Stateless rules'), ('LLM', 'LLM judge')]
def cell(st, gd, grp):
    r = g[(g.stream == st) & (g.grp == grp) & (g.guard.str.startswith(gd) if gd == 'LLM' else g.guard == gd)]
    if r.empty: return '--'
    k, n = int(r.k.iloc[0]), int(r.nhp.iloc[0])
    if grp == 'rule': return f'{R(k / n)} ({k}/{n})'
    lo = 0 if k == 0 else beta.ppf(.025, k, n - k + 1); hi = 1 if k == n else beta.ppf(.975, k + 1, n - k)
    return f'{R(k / n)} [{R(lo)}, {R(hi)}] ({k}/{n})'
L = [r'\begin{table*}[!t]', r'\centering',
     r'\caption{Prevention (\%) split by typology group. Rule-independent: OVERPRICED\_\allowbreak{}COLLUSIVE and SPLIT\_\allowbreak{}ADAPTIVE, which no hard rule targets, with Clopper--Pearson 95\% intervals; rule-matched: the other ten typologies. Counts are prevented/harmful-proposal tasks, recomputed from the per-typology result files.}',
     r'\label{tab:rulesplit}', r'\scriptsize\setlength{\tabcolsep}{2.5pt}', r'\begin{tabular}{llccccc}', r'\hline',
     'Group & Guard & ' + ' & '.join(s for _, s in streams) + r' \\', r'\hline']
for grp, gl in (('ind', 'Rule-independent'), ('rule', 'Rule-matched')):
    for i, (gd, gn) in enumerate(guards):
        L.append(f'{gl if i == 0 else ""} & {gn} & ' + ' & '.join(cell(s, gd, grp) for s, _ in streams) + r' \\')
    L.append(r'\hline')
L += [r'\end{tabular}', r'\end{table*}']
(ROOT/'analysis/tables_out').mkdir(exist_ok=True); open(ROOT/'analysis/tables_out/tab_rulesplit.tex', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))

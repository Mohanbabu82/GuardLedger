# =====================================================================================
# GuardLedger -- CONFIRMATORY PASS (pre-specified in revision/CONFIRMATORY_PROTOCOL.md)
# Run after guardledger_src.py (RUN_LLM=False) and guardledger_revision.py (all R sections skipped),
# in the same namespace. Writes <OUT_DIR>/confirmatory/ and prints compact result tables into the log.
#   C0 determinism diagnostics      C1 restore the V2 instance from the V2 caches
#   C2 GL-IF on the frozen V2 benchmark (approve-all + 4 cached LLM streams)
#   C3 fresh held-out tasks (seeds 2027, 4049, 6067)       C4 ACI on the IF score
#   C5 export of the frozen task files with SHA-256
# CONF["DIAG_ONLY"]=True runs C0 and the C1 match count only (for a CPU session).
# =====================================================================================
import os, re, gc, sys, json, gzip, hashlib, platform, time, io
import numpy as np, pandas as pd
from collections import defaultdict

CONF = dict(OUT=os.path.join(CFG["OUT_DIR"], "confirmatory"), HELDOUT_SEEDS=(2027, 4049, 6067), DIAG_ONLY=False,
            SRC_PATH=None, PREFIX_VARIANTS=True)
CONF.update(globals().get("CONF_OVERRIDE", {}))
os.makedirs(CONF["OUT"], exist_ok=True)
GLIF, GLS1, IFC, IFP = "GL-hard + IF (conformal)", "GuardLedger (full)", "IsolationForest (conformal)", "IsolationForest"
RESULTS = {}

def csave(df, name):
    df.to_csv(os.path.join(CONF["OUT"], name), index=False); RESULTS[name] = df; log(f"[confirm] wrote {name} ({len(df)} rows)")

def _conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (set, frozenset)): return sorted(o, key=str)
    return str(o)

def canon(tasks):
    return "\n".join(json.dumps(t, sort_keys=True, default=_conv) for t in tasks) + "\n"

def fp(tasks): return hashlib.sha256(canon(tasks).encode()).hexdigest()

def inv_order_fp(inv): return hashlib.sha256("|".join(map(str, inv["case_id"])).encode()).hexdigest()

# ---------------- C0: determinism diagnostics ----------------
def cpu_flags():
    try:
        txt = open("/proc/cpuinfo").read()
        model = re.search(r"model name\s*:\s*(.*)", txt); flags = re.search(r"flags\s*:\s*(.*)", txt)
        fl = set(flags.group(1).split()) if flags else set()
        return dict(cpu_model=model.group(1) if model else "?", avx2="avx2" in fl, avx512f="avx512f" in fl,
                    avx512_vbmi2="avx512_vbmi2" in fl, n_cpu=os.cpu_count())
    except Exception as e:
        return dict(cpu_error=str(e))

diag = dict(PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED"), hash_randomization=int(sys.flags.hash_randomization),
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, gpu=bool(globals().get("HAS_GPU")),
            **cpu_flags())
_ties = int(INV["inv_t"].duplicated(keep=False).sum())
_stable_inv = ITEMS[ITEMS["inv_t"].notna() & ITEMS["inv_amount"].gt(0)].sort_values("inv_t", kind="stable").reset_index(drop=True)
diag.update(inv_rows=int(len(INV)), inv_rows_with_tied_time=_ties, inv_order_fp=inv_order_fp(INV),
            inv_order_equals_stable=bool((INV["case_id"].to_numpy() == _stable_inv["case_id"].to_numpy()).all()),
            bench_fp_session=fp(BENCH), vendor_master_first5_in_set_order=list(POLICY.vendor_master)[:5])
del _stable_inv

def run_prefix_variant(tag, src_text, patches):
    """Execute the V2 source up to the benchmark, in a fresh namespace, with textual patches; return fingerprints."""
    cut = src_text.index("bench_counts = Counter(t[\"category\"] for t in BENCH)")
    txt = src_text[:cut]
    txt = txt.replace('"/kaggle/working/guardledger_out" if ON_KAGGLE', f'"/kaggle/working/gl_variant_{tag}" if ON_KAGGLE', 1)
    txt = txt.replace('os.environ.get("GL_OUT", "./guardledger_out")', f'os.environ.get("GL_OUT", "./guardledger_out") + "_variant_{tag}"', 1)
    for a, b in patches:
        n = txt.count(a); assert n >= 1, f"patch target not found: {a}"; txt = txt.replace(a, b)
    ns = {"__name__": f"variant_{tag}"}
    t0 = time.time(); exec(compile(txt, f"variant_{tag}", "exec"), ns)
    out = dict(variant=tag, inv_order_fp=inv_order_fp(ns["INV"]), bench_fp=fp(ns["BENCH"]), seconds=round(time.time() - t0, 1),
               n_tasks=len(ns["BENCH"]))
    del ns; gc.collect(); return out

SORTED_SET = [('vens = [v for v in POLICY.vendor_master if v != p0["vendor"]]',
               'vens = [v for v in sorted(POLICY.vendor_master) if v != p0["vendor"]]')]
STABLE = [('.sort_values("inv_t")', '.sort_values("inv_t", kind="stable")'), ('.sort_values("po_t")', '.sort_values("po_t", kind="stable")')]
variants = []
if CONF["PREFIX_VARIANTS"] and CONF["SRC_PATH"]:
    _src = open(CONF["SRC_PATH"]).read().replace("RUN_LLM=True,", "RUN_LLM=False,", 1)
    for tag, patches in (("v2_code", []), ("sorted_set", SORTED_SET), ("sorted_set_stable_sort", SORTED_SET + STABLE)):
        try:
            variants.append(run_prefix_variant(tag, _src, patches)); log("[confirm] variant", variants[-1])
        except Exception as e:
            variants.append(dict(variant=tag, error=f"{type(e).__name__}: {e}")); log("[confirm] variant failed", tag, e)
diag["variants"] = variants
json.dump(diag, open(os.path.join(CONF["OUT"], "determinism.json"), "w"), indent=1, default=_conv)
log("[confirm] determinism:", json.dumps(diag, default=_conv))

# ---------------- C1: restore the V2 instance from the V2 caches ----------------
C = GenCache(os.path.join(CFG["OUT_DIR"], "llm_cache_confirm.jsonl"))
MODELS = [s["name"] for s in CFG.get("LLM_MODELS", [])]
STREAMS = [(m, wp) for m in MODELS for wp in (False, True)]
def step_missing(p):
    return [(m, wp) for m, wp in STREAMS if C.key(m, agent_messages(p, wp)) not in C.d]
SUFFIX = lambda v: INJ_TEMPLATES[2].format(amt=0.0, usr="", ven=v)
rest_rows, n_missing_before = [], 0
for t in BENCH:
    for j, s in enumerate(t["steps"]):
        p = s["packet"]; miss = step_missing(p)
        if not miss: continue
        n_missing_before += 1
        inj = s.get("injected_action") or {}
        ven_cur = (inj.get("args") or {}).get("vendor_id")
        row = dict(task_id=t["task_id"], step=j, category=t["category"], streams_missing=len(miss), restored=False)
        if CONF["DIAG_ONLY"] or t["category"] != "PROMPT_INJECTION" or not ven_cur or not p["memo"].endswith(SUFFIX(ven_cur)):
            rest_rows.append(row); continue
        base = p["memo"][: len(p["memo"]) - len(SUFFIX(ven_cur))]
        m0, wp0 = miss[0]; hits = []
        for v in sorted(POLICY.vendor_master):
            if v == p["vendor"]: continue
            q = dict(p); q["memo"] = base + SUFFIX(v)
            if C.key(m0, agent_messages(q, wp0)) in C.d: hits.append(v)
        row["candidates_matched"] = len(hits)
        if len(hits) == 1:
            p["memo"] = base + SUFFIX(hits[0]); inj["args"]["vendor_id"] = hits[0]
            row.update(restored=not step_missing(p), v2_vendor_rank_in_sorted=sorted(POLICY.vendor_master).index(hits[0]))
        rest_rows.append(row)
n_missing_after = sum(1 for t in BENCH for s in t["steps"] if step_missing(s["packet"]))
csave(pd.DataFrame(rest_rows) if rest_rows else pd.DataFrame([dict(note="no missing steps")]), "restoration_log.csv")
RESTORE = dict(steps_missing_before=n_missing_before, steps_missing_after=n_missing_after, bench_fp_restored=fp(BENCH),
               total_steps=sum(len(t["steps"]) for t in BENCH))
log("[confirm] restoration:", RESTORE)
json.dump(RESTORE, open(os.path.join(CONF["OUT"], "restoration_summary.json"), "w"), indent=1)

if not CONF["DIAG_ONLY"]:
    # ---------------- C2: GL-IF on the frozen V2 benchmark ----------------
    def all_guards(prop, bench):
        return make_guards([iforest_guard_on(prop, bench)]) + extra_guards(prop, bench)
    sums, cats, pairs = [], [], []
    streams = [("approve_all", None)] + [(f"{m}|{'policy_prompt' if wp else 'plain'}", (m, wp)) for m, wp in STREAMS]
    for name, mw in streams:
        if mw is None:
            prop = approve_all_proposals(); bench = BENCH
        else:
            prop, missing, _ = _stream_props(mw[0], mw[1], C)
            bench = [t for t in BENCH if t["task_id"] not in missing]
            if not bench: continue
        r, sm, ct = run_guards_on(bench, prop, all_guards(prop, bench), name)
        sm["tasks_evaluated"] = len(bench); sums.append(sm); cats.append(ct)
        for a_, b_ in ((GLIF, GLS1), (GLIF, IFC), (GLS1, IFP), (GLIF, "GL-hard (no soft layer)")):
            for grp, cs in (("all", None), ("rule-independent", NONRULE), ("rule-matched", set(CATS) - NONRULE)):
                pairs.append(dict(stream=name, first=a_, second=b_, group=grp, **paired(r[a_], r[b_], cs)))
    S2 = pd.concat(sums); csave(S2, "benchmark_all_guards_summary.csv"); csave(pd.concat(cats), "benchmark_all_guards_per_category.csv")
    csave(pd.DataFrame(pairs), "benchmark_paired.csv")
    # agreement with the V2 result files (identical streams must give identical numbers)
    v2 = []
    for f in sorted(glob.glob("/kaggle/input/**/expD_llm_summary.csv", recursive=True)) + sorted(glob.glob("/kaggle/input/**/expA_summary.csv", recursive=True)):
        try: v2.append(pd.read_csv(f))
        except Exception: pass
    if v2:
        V2 = pd.concat(v2).drop_duplicates(["stream", "guard"])
        M = S2.merge(V2, on=["stream", "guard"], suffixes=("", "_v2"))
        rows = []
        for c in ("n_harm_proposed", "prevention", "fpr_legit", "harm_exec_rate_all", "usd_leaked"):
            if c + "_v2" in M: rows.append(dict(metric=c, pairs_compared=len(M), max_abs_diff=float((M[c] - M[c + "_v2"]).abs().max())))
        csave(pd.DataFrame(rows), "agreement_with_v2.csv")

    # ---------------- C3: fresh held-out tasks ----------------
    H = []
    for sd in CONF["HELDOUT_SEEDS"]:
        hs = build_heldout(seed=sd)
        for t in hs:
            old = t["task_id"]; t["task_id"] = f"S{sd}-{old}"; t["seed"] = sd
            for s in t["steps"]: s["packet"]["invoice_id"] = f"S{sd}-" + s["packet"]["invoice_id"]
        H += hs
    H314 = build_heldout(seed=314)          # the R10 tasks, for the frozen-file export and as a check against R10
    old = BENCH
    try:
        globals()["BENCH"] = H; propH = approve_all_proposals()
    finally:
        globals()["BENCH"] = old
    rH, smH, ctH = run_guards_on(H, propH, all_guards(propH, H), "heldout_fresh|approve_all")
    csave(smH, "heldout_fresh_summary.csv"); csave(ctH, "heldout_fresh_per_category.csv")
    rows = []
    for g, d in rH.items():
        rows.append(dict(guard=g, seed="pooled", **pooled(d, set(HELDOUT_CATS))))
        for sd in CONF["HELDOUT_SEEDS"]:
            dd = d[d["task_id"].str.startswith(f"S{sd}-")]
            rows.append(dict(guard=g, seed=sd, **pooled(dd, set(HELDOUT_CATS))))
    csave(pd.DataFrame(rows), "heldout_fresh_pooled.csv")
    hp = [dict(hypothesis=h, first=a_, second=b_, **paired(rH[a_], rH[b_])) for h, a_, b_ in
          (("H1", GLIF, GLS1), ("H2", GLIF, IFC), ("H2b", GLIF, IFP), ("ref", GLS1, IFP), ("ref", GLIF, "GL-hard + fused (conformal)"))]
    csave(pd.DataFrame(hp), "heldout_fresh_paired.csv")

    # ---------------- C4: ACI on the IF score ----------------
    df = INV[INV["split"].isin(["cal", "test"])]
    df = df.sample(n=min(REV["ROLL_N"], len(df)), random_state=21).sort_values("inv_t")
    PK = []
    for _, r_ in df.iterrows():
        p = make_packet(r_)
        if oracle_ok(p)[0]: PK.append(p)
    T = np.array([p["t"] for p in PK])
    X = np.array([feats(p, truth_action(p)) for p in PK])
    SCORES = {"IF": -IF_MODEL.score_samples(X),
              "S1": np.array([POLICY.soft_score(p["inv_amount"], p["vendor"], p["spend_area"], p["item_type"]) for p in PK])}
    STATIC = {"IF": IF_THR_CONF, "S1": LAM_STAR}
    a = CFG["ALPHA"]; idx = np.flatnonzero(T >= T_TEST0); month = np.array([fmt_day(t)[:7] for t in T])
    rows = []
    for nm, S in SCORES.items():
        roll = {}
        for m in sorted(set(month[idx])):
            start = T[month == m].min(); roll[m] = conf_thr(S[T < start][-REV["ROLL_WIN"]:], a)
        alpha_t, err, thr_path = a, [], []
        for i in idx:
            win = S[:i][-REV["ROLL_WIN"]:]
            at = min(max(alpha_t, 1.0 / (len(win) + 1)), 0.999)
            thr = conf_thr(win, at); e = float(S[i] > thr); err.append(e); thr_path.append(thr)
            alpha_t = alpha_t + REV["ACI_GAMMA"] * (a - e)
        err = np.array(err); thr_path = np.array(thr_path)
        for m in sorted(set(month[idx])):
            mm = month[idx] == m; si = S[idx][mm]
            rows.append(dict(score=nm, month=m, n=int(mm.sum()), static=float(np.mean(si > STATIC[nm])),
                             rolling=float(np.mean(si > roll[m])), aci=float(err[mm].mean()),
                             aci_thr_median=float(np.median(thr_path[mm])), static_thr=float(STATIC[nm])))
        R = pd.DataFrame([x for x in rows if x["score"] == nm])
        rows.append(dict(score=nm, month="ALL", n=int(len(idx)), static=float(np.mean(S[idx] > STATIC[nm])),
                         rolling=float(np.average(R["rolling"], weights=R["n"])), aci=float(err.mean()),
                         aci_thr_median=float(np.median(thr_path)), static_thr=float(STATIC[nm])))
    csave(pd.DataFrame(rows), "aci_by_month.csv")

    # ---------------- C5: frozen task files ----------------
    man = {}
    for nm, tasks in (("benchmark_v2_720.jsonl.gz", BENCH), ("heldout_seed314_200.jsonl.gz", H314), ("heldout_fresh_600.jsonl.gz", H)):
        body = canon(tasks).encode()
        with open(os.path.join(CONF["OUT"], nm), "wb") as fh:
            with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz: gz.write(body)
        man[nm] = dict(tasks=len(tasks), sha256_uncompressed_jsonl=hashlib.sha256(body).hexdigest(),
                       sha256_gz=hashlib.sha256(open(os.path.join(CONF["OUT"], nm), "rb").read()).hexdigest())
    man["note"] = "One canonical JSON object per task (sort_keys); numpy scalars converted to Python numbers."
    json.dump(man, open(os.path.join(CONF["OUT"], "frozen_tasks_manifest.json"), "w"), indent=1)
    RESTORE["heldout_seed314_fp"] = fp(H314)

# ---------------- compact print-out ----------------
def show(name, df, cols=None):
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    print(f"\n<<<FILE {name}>>>\n" + d.round(5).to_csv(index=False) + f"<<<END {name}>>>")
print("\n<<<FILE determinism.json>>>\n" + json.dumps(diag, indent=1, default=_conv) + "\n<<<END determinism.json>>>")
print("\n<<<FILE restoration_summary.json>>>\n" + json.dumps(RESTORE, indent=1) + "\n<<<END restoration_summary.json>>>")
if "restoration_log.csv" in RESULTS: show("restoration_log.csv", RESULTS["restoration_log.csv"])
if not CONF["DIAG_ONLY"]:
    KEY = [GLS1, GLIF, IFC, IFP, "GL-hard (no soft layer)", "AgentSpec-style", "LOF (conformal)", "Fused S1+IF (conformal)",
           "GL-hard + fused (conformal)", "GL-hard + fixed z>3 (no CRC)", "NoGuard"]
    S = RESULTS["benchmark_all_guards_summary.csv"]; S = S[S["guard"].isin(KEY)]
    show("benchmark_summary_key.csv", S, ["stream", "guard", "tasks_evaluated", "n_harm_proposed", "prevention", "prev_lo", "prev_hi",
                                          "fpr_legit", "fpr_lo", "fpr_hi", "usd_leaked", "usd_held", "harm_exec_rate_all", "lat_p50_us", "lat_p99_us"])
    if "agreement_with_v2.csv" in RESULTS: show("agreement_with_v2.csv", RESULTS["agreement_with_v2.csv"])
    Cc = RESULTS["benchmark_all_guards_per_category.csv"]
    W = Cc[Cc["guard"].isin([GLS1, GLIF, IFC, "GL-hard (no soft layer)"])].pivot_table(index=["stream", "category"], columns="guard", values="prevention").reset_index()
    show("benchmark_per_category_key.csv", W)
    show("benchmark_paired.csv", RESULTS["benchmark_paired.csv"])
    show("heldout_fresh_pooled.csv", RESULTS["heldout_fresh_pooled.csv"])
    show("heldout_fresh_paired.csv", RESULTS["heldout_fresh_paired.csv"])
    Hc = RESULTS["heldout_fresh_per_category.csv"]
    show("heldout_fresh_per_category_key.csv", Hc[Hc["guard"].isin(KEY)].pivot_table(index="category", columns="guard", values="prevention").reset_index())
    show("heldout_fresh_summary.csv", RESULTS["heldout_fresh_summary.csv"], ["guard", "n_harm_proposed", "prevention", "prev_lo", "prev_hi", "usd_leaked", "lat_p50_us"])
    show("aci_by_month.csv", RESULTS["aci_by_month.csv"])
    print("\n<<<FILE frozen_tasks_manifest.json>>>\n" + json.dumps(man, indent=1) + "\n<<<END frozen_tasks_manifest.json>>>")
log("[confirm] finished")

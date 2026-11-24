"""Pass A (CPU) of the GuardLedger revision kit on Kaggle.
Inputs attached to the notebook: the outputs of the V2 notebooks guardledger-v2-realbudget-qwen and -phi
(BPIC 2019 CSV and the V2 generation caches llm_cache.jsonl). Internet on (downloads the code from GitHub)."""
import os, subprocess, hashlib, glob
os.environ["GL_REV_GPU"] = "0"
W = "/kaggle/working/gl_code"; os.makedirs(W, exist_ok=True)
BASE = "https://raw.githubusercontent.com/Mohanbabu82/GuardLedger/main/"
for rel in ("src/guardledger_src.py", "revision/guardledger_revision.py"):
    subprocess.run(["wget", "-q", "-O", os.path.join(W, os.path.basename(rel)), BASE + rel], check=True)
raw = open(os.path.join(W, "guardledger_src.py"), "rb").read()
h = hashlib.sha256(raw).hexdigest(); print("V2 source SHA-256:", h)
assert h.startswith("2df52e0a") and h.endswith("96a9"), "not the frozen V2 source"
print("V2 caches:", sorted(glob.glob("/kaggle/input/**/llm_cache*.jsonl", recursive=True)))
print("BPIC CSV:", sorted(glob.glob("/kaggle/input/**/*Challenge_2019*.csv", recursive=True)))
# The V2 LLM stage is not re-executed: agent and judge streams are rebuilt from the V2 caches by kit section R3.
src = raw.decode().replace("RUN_LLM=True,", "RUN_LLM=False,", 1)
exec(compile(src, "guardledger_src.py", "exec"), globals())
CFG["LLM_MODELS"] = [
    dict(name="qwen2.5-7b-instruct", kaggle_globs=["/kaggle/input/qwen2.5/transformers/7b-instruct/*/config.json"], hf_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(name="phi-3.5-mini-instruct", kaggle_globs=[], hf_id="microsoft/Phi-3.5-mini-instruct")]
REV_OVERRIDE = dict()
exec(compile(open(os.path.join(W, "guardledger_revision.py")).read(), "guardledger_revision.py", "exec"), globals())
print(open(os.path.join(REV["OUT"], "revision_summary.md")).read())
import pandas as _pd   # print every result table into the log (readable without downloading the output)
for _p in sorted(glob.glob(os.path.join(REV["OUT"], "*"))):
    if _p.endswith(".gz"): continue
    _n = os.path.basename(_p); print(f"\n<<<FILE {_n}>>>")
    print(open(_p).read() if _p.endswith((".json", ".md")) else _pd.read_csv(_p).round(5).to_csv(index=False))
    print(f"<<<END {_n}>>>")

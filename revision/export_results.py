"""Print the revision results (pass A and, if attached, pass B) as plain text in the notebook log, and re-run the
fail-open / fail-closed judge comparison (kit section R3) which pass A skipped. CPU only, a few minutes."""
import os, glob, json, subprocess, hashlib
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
W = "/kaggle/working/gl_code"; os.makedirs(W, exist_ok=True)
BASE = "https://raw.githubusercontent.com/Mohanbabu82/GuardLedger/main/"
for rel in ("src/guardledger_src.py", "revision/guardledger_revision.py"):
    subprocess.run(["wget", "-q", "-O", os.path.join(W, os.path.basename(rel)), BASE + rel], check=True)
raw = open(os.path.join(W, "guardledger_src.py"), "rb").read()
assert hashlib.sha256(raw).hexdigest().startswith("2df52e0a")
os.environ["GL_REV_GPU"] = "0"
exec(compile(raw.decode().replace("RUN_LLM=True,", "RUN_LLM=False,", 1), "guardledger_src.py", "exec"), globals())
CFG["LLM_MODELS"] = [dict(name="qwen2.5-7b-instruct", kaggle_globs=[], hf_id="Qwen/Qwen2.5-7B-Instruct"),
                     dict(name="phi-3.5-mini-instruct", kaggle_globs=[], hf_id="microsoft/Phi-3.5-mini-instruct")]
REV_OVERRIDE = dict(SKIP=tuple(f"R{i}" for i in (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)))
exec(compile(open(os.path.join(W, "guardledger_revision.py")).read(), "guardledger_revision.py", "exec"), globals())
def show(path):
    name = os.path.basename(path)
    print(f"\n<<<FILE {name}>>>")
    if name.endswith(".json") or name.endswith(".md"): print(open(path).read())
    else:
        d = pd.read_csv(path)
        print(d.round(5).to_csv(index=False))
    print(f"<<<END {name}>>>")
for root in sorted(set(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/revision/revision_summary.md", recursive=True))) + [REV["OUT"]]:
    print(f"\n######## RESULTS FROM {root} ########")
    for p in sorted(glob.glob(os.path.join(root, "*"))):
        if p.endswith(".gz"): continue
        show(p)

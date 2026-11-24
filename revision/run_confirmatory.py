"""Confirmatory pass (protocol: revision/CONFIRMATORY_PROTOCOL.md) on Kaggle.
Same inputs as pass A: the outputs of the V2 notebooks (BPIC 2019 CSV, V2 llm_cache.jsonl, V2 result CSVs).
No model is loaded. Set GL_DIAG_ONLY=1 for the determinism diagnostics only (e.g. on a CPU session)."""
import os, subprocess, hashlib, glob
os.environ["GL_REV_GPU"] = "0"
W = "/kaggle/working/gl_code"; os.makedirs(W, exist_ok=True)
BASE = "https://raw.githubusercontent.com/Mohanbabu82/GuardLedger/main/"
for rel in ("src/guardledger_src.py", "revision/guardledger_revision.py", "revision/confirmatory.py"):
    subprocess.run(["wget", "-q", "-O", os.path.join(W, os.path.basename(rel)), BASE + rel], check=True)
raw = open(os.path.join(W, "guardledger_src.py"), "rb").read()
h = hashlib.sha256(raw).hexdigest(); print("V2 source SHA-256:", h)
assert h.startswith("2df52e0a") and h.endswith("96a9"), "not the frozen V2 source"
print("PYTHONHASHSEED at start:", os.environ.get("PYTHONHASHSEED"))
src = raw.decode().replace("RUN_LLM=True,", "RUN_LLM=False,", 1)
exec(compile(src, "guardledger_src.py", "exec"), globals())
CFG["LLM_MODELS"] = [
    dict(name="qwen2.5-7b-instruct", kaggle_globs=[], hf_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(name="phi-3.5-mini-instruct", kaggle_globs=[], hf_id="microsoft/Phi-3.5-mini-instruct")]
REV_OVERRIDE = dict(SKIP=tuple(f"R{i}" for i in range(15)))
exec(compile(open(os.path.join(W, "guardledger_revision.py")).read(), "guardledger_revision.py", "exec"), globals())
CONF_OVERRIDE = dict(SRC_PATH=os.path.join(W, "guardledger_src.py"), DIAG_ONLY=os.environ.get("GL_DIAG_ONLY") == "1")
exec(compile(open(os.path.join(W, "confirmatory.py")).read(), "confirmatory.py", "exec"), globals())

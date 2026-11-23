"""Pass B (GPU) of the GuardLedger revision kit on Kaggle: larger history-aware judge (R11), >=14B agent (R12),
closed-loop pilot (R13) and prompt variants (R14). R3 and R10 run again because R11/R12 use their streams.
Inputs: V2 notebook outputs (qwen, phi), Kaggle Model Qwen2.5 7B-instruct and 14B-instruct (transformers). Internet on."""
import os, subprocess, hashlib, glob, importlib.util
W = "/kaggle/working/gl_code"; os.makedirs(W, exist_ok=True)
BASE = "https://raw.githubusercontent.com/Mohanbabu82/GuardLedger/main/"
for rel in ("src/guardledger_src.py", "revision/guardledger_revision.py"):
    subprocess.run(["wget", "-q", "-O", os.path.join(W, os.path.basename(rel)), BASE + rel], check=True)
if importlib.util.find_spec("bitsandbytes") is None:
    subprocess.run(["pip", "install", "-q", "bitsandbytes"], check=False)
raw = open(os.path.join(W, "guardledger_src.py"), "rb").read()
h = hashlib.sha256(raw).hexdigest(); print("V2 source SHA-256:", h)
assert h.startswith("2df52e0a") and h.endswith("96a9"), "not the frozen V2 source"
print("V2 caches:", sorted(glob.glob("/kaggle/input/**/llm_cache*.jsonl", recursive=True)))
print("models:", sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))[:20])
src = raw.decode().replace("RUN_LLM=True,", "RUN_LLM=False,", 1)
exec(compile(src, "guardledger_src.py", "exec"), globals())
CFG["LLM_MODELS"] = [
    dict(name="qwen2.5-7b-instruct", kaggle_globs=["/kaggle/input/qwen2.5/transformers/7b-instruct/*/config.json",
                                                   "/kaggle/input/**/*7b-instruct*/**/config.json"], hf_id="Qwen/Qwen2.5-7B-Instruct"),
    dict(name="phi-3.5-mini-instruct", kaggle_globs=["/kaggle/input/**/*phi-3.5-mini*/**/config.json"], hf_id="microsoft/Phi-3.5-mini-instruct")]
CFG["LLM_TIME_BUDGET_MIN"] = 150  # per GPU section; bounds total GPU use
Q14 = dict(name="qwen2.5-14b-instruct-nf4", quant="4bit",
           kaggle_globs=["/kaggle/input/qwen2.5/transformers/14b-instruct/*/config.json", "/kaggle/input/**/*14b-instruct*/**/config.json"],
           hf_id="Qwen/Qwen2.5-14B-Instruct")
REV_OVERRIDE = dict(SKIP=("R2", "R4", "R5", "R6", "R7", "R8", "R9", "R14"), JUDGE_MODELS=[Q14], EXTRA_AGENT_MODELS=[Q14], CLOSED_LOOP_PER_CAT=10)
exec(compile(open(os.path.join(W, "guardledger_revision.py")).read(), "guardledger_revision.py", "exec"), globals())
print(open(os.path.join(REV["OUT"], "revision_summary.md")).read())

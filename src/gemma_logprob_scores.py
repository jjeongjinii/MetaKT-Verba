"""W3: continuous Gemma score from Yes/No next-token probabilities.

The verbalized confidence used so far takes only ~8-11 distinct values (multiples of five),
so most sentence pairs are tied. Here Gemma answers a one-word Yes/No question and the score
is P(Yes) / (P(Yes) + P(No)) from the first generated token's top log-probabilities.
Same zero-shot instruction under both premises; only the premise text differs.

Needs the vLLM OpenAI-compatible server already used for Gemma (no other dependencies).
  python gemma_logprob_scores.py \
      --cases outputs/verifier_family/cases.pkl \
      --scores outputs/verifier_family/scores.pkl \
      --api-base http://localhost:8000/v1 --model <served Gemma name>

Writes:
  <out-dir>/scores_gemma_logprob.csv     new rows only (verifier = Gemma-3-27B-it-logprob)
  <out-dir>/scores_with_logprob.pkl      --scores plus the new rows (feed to cluster_bootstrap_ci.py)
"""

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

NAME = "Gemma-3-27B-it-logprob"
TITLES = {"fact": ("symbolic fact(s) produced by a learner model", "Fact(s)"),
          "record": ("student's behavior record that the teacher can see", "Record")}


def build_prompt(premise_text: str, sentence: str, mode: str) -> str:
    what, label = TITLES[mode]
    return "\n".join([
        f"You check whether a sentence written for a teacher is supported by the {what}.",
        "Judge ONLY from the text below. Do not use outside knowledge about the student.",
        "",
        f"{label}:\n{premise_text}",
        f"Sentence: {sentence}",
        "",
        "Is the sentence supported? Answer with exactly one word: Yes or No.",
    ])


def p_yes(prompt: str, model: str, api_base: str, top_k: int, timeout: int = 120):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": top_k,
    }).encode("utf-8")
    req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            tops = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            py = sum(math.exp(t["logprob"]) for t in tops if t["token"].strip().lower() == "yes")
            pn = sum(math.exp(t["logprob"]) for t in tops if t["token"].strip().lower() == "no")
            first = data["choices"][0]["message"]["content"]
            if py + pn == 0:
                return np.nan, py, pn, first
            return py / (py + pn), py, pn, first
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError,
                TypeError, json.JSONDecodeError) as e:
            err = e
    raise RuntimeError(f"request failed 3 times: {err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="/home/elicer/project/metaKT-verba/outputs/verifier_family/cases.pkl")
    ap.add_argument("--scores", default="/home/elicer/project/metaKT-verba/outputs/verifier_family/scores.pkl")
    ap.add_argument("--out-dir", default="/home/elicer/project/metaKT-verba/outputs/verifier_family")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default='google/Gemma-3-27B-it', help="model name as served by vLLM")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows per premise (smoke test)")
    args = ap.parse_args()

    cases = pd.read_pickle(args.cases)
    jobs = []
    for mode in ("fact", "record"):
        d = cases[cases[f"{mode}_premise"].notna()]
        if args.limit:
            d = d.head(args.limit)
        for idx, r in d.iterrows():
            jobs.append((idx, mode, build_prompt(r[f"{mode}_premise"], r["sentence_text"], mode)))
    print(f"[{NAME}] {len(jobs)} prompts (fact + record)")

    def run(job):
        idx, mode, prompt = job
        return (idx, mode, *p_yes(prompt, args.model, args.api_base, args.top_logprobs))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run, jobs))

    rows = []
    for idx, mode, score, py, pn, first in res:
        rows.append({**cases.loc[idx].to_dict(), "score": score, "verifier": NAME, "premise": mode,
                     "judge_label": first, "p_yes_raw": py, "p_no_raw": pn})
    out = pd.DataFrame(rows).drop(columns=["steps"], errors="ignore")
    os.makedirs(args.out_dir, exist_ok=True)
    out.to_csv(os.path.join(args.out_dir, "scores_gemma_logprob.csv"), index=False)

    for mode, g in out.groupby("premise"):
        mass = (g["p_yes_raw"] + g["p_no_raw"])
        print(f"[{NAME}/{mode}] n={len(g)}  missing={int(g['score'].isna().sum())}  "
              f"distinct scores={g['score'].round(6).nunique()}  "
              f"median Yes+No mass={mass.median():.3f}  pass(>=.5)={(g['score'] >= .5).mean():.3f}")

    if args.scores and os.path.exists(args.scores) and not args.limit:
        base = pd.read_pickle(args.scores)
        base = base[base["verifier"] != NAME]
        merged = pd.concat([base, out], ignore_index=True)
        merged.to_pickle(os.path.join(args.out_dir, "scores_with_logprob.pkl"))
        print(f"merged -> {os.path.join(args.out_dir, 'scores_with_logprob.pkl')} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
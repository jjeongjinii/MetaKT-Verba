import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
from openpyxl import load_workbook

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

LEVEL1_CATEGORIES = ["supported", "partial", "unsupported", "unknown"]
LEVEL2_DIMS = ["faithfulness", "actionability", "clarity"]
NEGATIVE_CONTROL_GATE = 0.70

TAG_NORMALIZE = {
    '\uc9c0\uc9c0\ub428': 'supported', '\ubd80\ubd84\uc9c0\uc9c0': 'partial', '\uadfc\uac70\uc5c6\uc74c': 'unsupported', '\ud310\ub2e8\ubd88\uac00': 'unknown',
    'supported': 'supported', 'partial': 'partial', 'unsupported': 'unsupported', 'unknown': 'unknown',
}

def _normalize_tag(raw):
    if raw is None:
        return None
    raw = str(raw).strip()
    return TAG_NORMALIZE.get(raw, raw if raw else None)

def _parse_level1_tags_field(tag_str: str) -> dict:
    result = {}
    if not isinstance(tag_str, str) or not tag_str.strip():
        return result
    for part in tag_str.split(';'):
        part = part.strip()
        if not part or ':' not in part:
            continue
        sid, tag = part.split(':', 1)
        result[sid.strip()] = _normalize_tag(tag)
    return result

def load_csv_response(path: str):
    df = pd.read_csv(path, dtype=str)
    required = {'item_id', 'rater_id', 'level1_tags', 'faithfulness', 'actionability', 'clarity'}
    missing = required - set(df.columns)
    if missing:
        print(f"[warning] Skipping {path}: missing required columns {sorted(missing)}")
        return None

    rows = []
    for _, r in df.iterrows():
        tags = _parse_level1_tags_field(r.get('level1_tags', ''))
        base = {
            'item_id': r['item_id'], 'rater_id': r['rater_id'],
            'faithfulness': r.get('faithfulness'), 'actionability': r.get('actionability'),
            'clarity': r.get('clarity'), 'free_text': r.get('free_text', ''),
            'duration_sec': r.get('duration_sec'), '__source_file': os.path.basename(path),
        }
        if tags:
            for sid, tag in tags.items():
                rows.append({**base, 'sentence_id': sid, 'level1_tag': tag})
        else:

            rows.append({**base, 'sentence_id': None, 'level1_tag': None})
    return pd.DataFrame(rows)

def load_xlsx_response(path: str):
    wb = load_workbook(path, data_only=True)
    if 'Level1_\ubb38\uc7a5\ud3c9\uc815' not in wb.sheetnames or 'Level2_\ubb38\ud56d\ud3c9\uc815' not in wb.sheetnames:
        print(f"[warning] Skipping {path}: required rating sheets are missing")
        return None

    fname = os.path.basename(path)
    m = re.match(r'([^_]+)_\ud3c9\uc815\uc591\uc2dd', fname)
    rater_id = m.group(1) if m else None
    if not rater_id and '\uc548\ub0b4' in wb.sheetnames:
        for row in wb['\uc548\ub0b4'].iter_rows(min_row=1, max_row=10, max_col=1):
            v = row[0].value
            if isinstance(v, str) and '\ud3c9\uc815\uc790 ID' in v:
                rater_id = v.split(':', 1)[-1].strip()
                break
    if not rater_id:
        print(f"[warning] Skipping {path}: rater ID could not be determined")
        return None

    ws1 = wb['Level1_\ubb38\uc7a5\ud3c9\uc815']
    headers1 = [c.value for c in next(ws1.iter_rows(min_row=1, max_row=1))]
    rows_l1 = []
    for row in ws1.iter_rows(min_row=2, values_only=True):
        d = dict(zip(headers1, row))
        item_id = d.get('\ubb38\ud56dID')
        if not item_id or str(item_id).startswith('(\uc608\uc2dc)'):
            continue
        tag_raw = d.get('\ud3c9\uc815(\ub4dc\ub86d\ub2e4\uc6b4 \uc120\ud0dd)')
        if tag_raw is None or str(tag_raw).strip() == '':
            continue
        rows_l1.append({'item_id': item_id, 'sentence_id': d.get('\ubb38\uc7a5ID'),
                         'level1_tag': _normalize_tag(tag_raw)})
    l1_df = pd.DataFrame(rows_l1)

    ws2 = wb['Level2_\ubb38\ud56d\ud3c9\uc815']
    headers2 = [c.value for c in next(ws2.iter_rows(min_row=1, max_row=1))]
    rows_l2 = []
    for row in ws2.iter_rows(min_row=2, values_only=True):
        d = dict(zip(headers2, row))
        item_id = d.get('\ubb38\ud56dID')
        if not item_id or str(item_id).startswith('(\uc608\uc2dc)'):
            continue
        rows_l2.append({
            'item_id': item_id,
            'faithfulness': d.get('\ucda9\uc2e4\ub3c4(1-5)'), 'actionability': d.get('\uc2e4\ud589\uac00\ub2a5\uc131(1-5)'),
            'clarity': d.get('\uba85\ub8cc\uc131(1-5)'), 'free_text': d.get('\uc790\uc720\uc11c\uc220(\uc120\ud0dd)') or '',
        })
    l2_df = pd.DataFrame(rows_l2)

    if l1_df.empty and l2_df.empty:
        print(f"[warning] Skipping {path}: no completed responses found")
        return None
    if l1_df.empty:
        merged = l2_df.copy()
        merged['sentence_id'] = None
        merged['level1_tag'] = None
    elif l2_df.empty:
        merged = l1_df.copy()
        for c in LEVEL2_DIMS:
            merged[c] = None
        merged['free_text'] = ''
    else:
        merged = l1_df.merge(l2_df, on='item_id', how='outer')

    merged['rater_id'] = rater_id
    merged['duration_sec'] = None
    merged['__source_file'] = fname
    return merged

def load_responses(responses_dir: str) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(os.path.join(responses_dir, "*.csv")))
    xlsx_paths = sorted(glob.glob(os.path.join(responses_dir, "*.xlsx")))
    if not csv_paths and not xlsx_paths:
        raise FileNotFoundError(f"No CSV/XLSX response files found in {responses_dir!r}.")

    frames = []
    for p in csv_paths:
        df = load_csv_response(p)
        if df is not None and not df.empty:
            frames.append(df)
    for p in xlsx_paths:
        df = load_xlsx_response(p)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise ValueError("No valid response files could be loaded.")

    out = pd.concat(frames, ignore_index=True)
    for col in LEVEL2_DIMS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    n_raters = out["rater_id"].nunique()
    print(f"[responses] files={len(frames)}, raters={n_raters}, rows={len(out)}")
    return out

def load_answer_key(path: str) -> pd.DataFrame:
    key_df = pd.read_csv(path, dtype={"display_id": str})
    key_df["is_negative_control"] = key_df["is_negative_control"].astype(str).str.lower() == "true"
    key_df["in_core_set"] = key_df["in_core_set"].astype(str).str.lower() == "true"
    return key_df.set_index("display_id")

def load_sentence_answer_key(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"display_id": str, "sentence_id": str})
    return df

def fleiss_kappa(subject_category_counts: np.ndarray) -> dict:
    N, k = subject_category_counts.shape
    n = subject_category_counts.sum(axis=1)
    if not np.all(n == n[0]):
        raise ValueError("Fleiss kappa requires the same number of ratings per subject.")
    n = int(n[0])
    if n < 2:
        raise ValueError("Fleiss kappa requires at least two raters per subject.")

    p_j = subject_category_counts.sum(axis=0) / (N * n)
    P_i = (np.sum(subject_category_counts ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else float("nan")
    return {"kappa": kappa, "n_subjects": N, "n_raters_per_subject": n, "P_bar": P_bar, "P_e": P_e}

_ORDINAL_VALUE = {'unsupported': 0, 'partial': 1, 'supported': 2}

def _ordinal_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a == 'unknown' or b == 'unknown':
        return 2.0
    return abs(_ORDINAL_VALUE[a] - _ORDINAL_VALUE[b])

def weighted_pairwise_kappa(wide: pd.DataFrame) -> dict:
    raters = list(wide.columns)
    max_dist = 2.0
    pair_kappas = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            pair = wide[[raters[i], raters[j]]].dropna()
            if len(pair) < 2:
                continue
            a_vals, b_vals = pair.iloc[:, 0].tolist(), pair.iloc[:, 1].tolist()
            categories = sorted(set(a_vals) | set(b_vals))
            if len(categories) < 2:
                continue
            n = len(a_vals)

            obs_weights = [1 - (_ordinal_distance(a, b) / max_dist) ** 2 for a, b in zip(a_vals, b_vals)]
            po_w = np.mean(obs_weights)

            pa = {c: a_vals.count(c) / n for c in categories}
            pb = {c: b_vals.count(c) / n for c in categories}
            pe_w = sum(pa[c1] * pb[c2] * (1 - (_ordinal_distance(c1, c2) / max_dist) ** 2)
                       for c1 in categories for c2 in categories)
            if pe_w == 1:
                continue
            pair_kappas.append((po_w - pe_w) / (1 - pe_w))
    if not pair_kappas:
        return {'weighted_kappa': None, 'n_rater_pairs': 0}
    return {'weighted_kappa': float(np.mean(pair_kappas)), 'n_rater_pairs': len(pair_kappas)}

def icc_2_1(ratings: np.ndarray) -> dict:
    n, k = ratings.shape
    if n < 2 or k < 2:
        raise ValueError("ICC(2,1) requires at least two items and two raters.")

    mean_targets = ratings.mean(axis=1)
    mean_raters = ratings.mean(axis=0)
    grand_mean = ratings.mean()

    SSR = k * np.sum((mean_targets - grand_mean) ** 2)
    SSC = n * np.sum((mean_raters - grand_mean) ** 2)
    SST = np.sum((ratings - grand_mean) ** 2)
    SSE = SST - SSR - SSC

    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    icc = (MSR - MSE) / denom if denom != 0 else float("nan")
    return {"icc": icc, "n_targets": n, "k_raters": k, "MSR": MSR, "MSC": MSC, "MSE": MSE}

def pearson_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    result = {"n": int(mask.sum())}
    if result["n"] < 3:
        result.update({"pearson_r": float("nan"), "pearson_p": float("nan"),
                        "spearman_rho": float("nan"), "spearman_p": float("nan")})
        return result
    if _HAS_SCIPY:
        r, p = scipy_stats.pearsonr(x, y)
        rho, sp = scipy_stats.spearmanr(x, y)
    else:
        r = float(np.corrcoef(x, y)[0, 1])
        p = float("nan")
        rank_x, rank_y = pd.Series(x).rank(), pd.Series(y).rank()
        rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
        sp = float("nan")
    result.update({"pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": sp})
    return result

def build_item_level_summary(resp_df: pd.DataFrame, key_df: pd.DataFrame) -> pd.DataFrame:
    level2_first_rows = resp_df.groupby(["item_id", "rater_id"], as_index=False).first()
    agg = level2_first_rows.groupby("item_id").agg(
        n_raters=("rater_id", "nunique"),
        mean_faithfulness=("faithfulness", "mean"),
        mean_actionability=("actionability", "mean"),
        mean_clarity=("clarity", "mean"),
    ).reset_index()

    joined = agg.join(key_df, on="item_id", how="left")
    unmatched = joined[joined["category"].isna()]["item_id"].tolist()
    if unmatched:
        print(f"[warning] {len(unmatched)} response item ID(s) are missing from the answer key")

    no_response_ids = set(key_df.index) - set(agg["item_id"])
    if no_response_ids:
        print(f"[warning] {len(no_response_ids)} answer-key item(s) have no responses")
    return joined

def _read_score_table(path: str) -> pd.DataFrame:
    suffix = str(path).lower()
    if suffix.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    if suffix.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported verifier-score format: {path}")

def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

def attach_reconstructed_item_scores(
    item_summary: pd.DataFrame,
    verifier_scores_path: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    scores = _read_score_table(verifier_scores_path)

    d = scores[
        (scores["verifier"] == "mDeBERTa-v3-base-mnli-xnli")
        & (scores["premise"] == "fact")
        & (~_as_bool_series(scores["is_negative_control"]))
        & scores["score"].notna()
    ].copy()

    if d.duplicated(["item_id", "sentence_id"]).any():
        raise RuntimeError(
            "Duplicate mDeBERTa/fact sentence scores found."
        )

    item_auto = (
        d.groupby("item_id")
        .agg(
            reconstructed_mean_entailment=("score", "mean"),
            reconstructed_faithfulness_rate=(
                "score",
                lambda x: float((x >= threshold).mean()),
            ),
            n_reconstructed_sentences=("sentence_id", "nunique"),
        )
        .reset_index()
    )

    out = item_summary.merge(
        item_auto,
        on="item_id",
        how="left",
        validate="one_to_one",
    )

    noncontrol = out[
        ~out["is_negative_control"].fillna(False).astype(bool)
    ]

    n_with_score = int(
        noncontrol["reconstructed_mean_entailment"].notna().sum()
    )

    print(
        f"[item reconstructed scores] "
        f"{n_with_score}/{len(noncontrol)} non-control items have scores"
    )

    if len(noncontrol) != 120:
        print(
            f"[warning] Expected 120 non-control items, "
            f"found {len(noncontrol)}."
        )

    return out

def build_sentence_level_summary(resp_df: pd.DataFrame, sentence_key_df: pd.DataFrame) -> pd.DataFrame:
    l1 = resp_df.dropna(subset=["sentence_id", "level1_tag"])
    if l1.empty:
        return pd.DataFrame()

    def _mode_or_none(s):
        m = s.mode()
        return m.iloc[0] if not m.empty else None

    def _counts_str(s):
        return ';'.join(f"{k}={v}" for k, v in s.value_counts().items())

    agg = l1.groupby(["item_id", "sentence_id"]).agg(
        n_raters=("rater_id", "nunique"),
        majority_tag=("level1_tag", _mode_or_none),
        tag_distribution=("level1_tag", _counts_str),
    ).reset_index()

    if sentence_key_df is not None and not sentence_key_df.empty:
        key_small = sentence_key_df.rename(columns={"display_id": "item_id"})[
            ["item_id", "sentence_id", "sentence_en", "auto_status", "auto_entailment", "cited_indicators"]
        ]
        agg = agg.merge(key_small, on=["item_id", "sentence_id"], how="left")
    return agg

def compute_core_set_reliability(resp_df: pd.DataFrame, key_df: pd.DataFrame) -> dict:
    core_ids = set(key_df[key_df["in_core_set"]].index)
    core_resp = resp_df[resp_df["item_id"].isin(core_ids)]
    if core_resp.empty:
        return {"error": "No core-set responses are available."}

    result = {}

    l1 = core_resp.dropna(subset=["level1_tag"])
    counts = l1.groupby(["item_id", "sentence_id", "level1_tag"]).size().unstack(fill_value=0)
    for cat in LEVEL1_CATEGORIES:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[LEVEL1_CATEGORIES]
    n_per_subject = counts.sum(axis=1)
    modal_n = n_per_subject.mode().iloc[0] if not n_per_subject.empty else 0
    complete = counts[n_per_subject == modal_n]
    dropped = len(counts) - len(complete)
    if len(complete) >= 2 and modal_n >= 2:
        fk = fleiss_kappa(complete.to_numpy())
        fk["n_subjects_dropped_incomplete"] = int(dropped)
        result["level1_fleiss_kappa"] = fk
    else:
        result["level1_fleiss_kappa"] = {"error": "Too few complete core-set sentences for Fleiss kappa."}

    l1_wide = l1.pivot_table(index=["item_id", "sentence_id"], columns="rater_id",
                              values="level1_tag", aggfunc="first")
    wk = weighted_pairwise_kappa(l1_wide)
    result["level1_weighted_kappa"] = wk

    l2 = core_resp.groupby(["item_id", "rater_id"], as_index=False).first()
    icc_results = {}
    for dim in LEVEL2_DIMS:
        pivot = l2.pivot(index="item_id", columns="rater_id", values=dim)
        complete_rows = pivot.dropna(axis=0, how="any")
        if complete_rows.shape[0] >= 2 and complete_rows.shape[1] >= 2:
            icc_results[dim] = icc_2_1(complete_rows.to_numpy(dtype=float))
            icc_results[dim]["n_items_dropped_incomplete"] = int(pivot.shape[0] - complete_rows.shape[0])
        else:
            icc_results[dim] = {"error": "Too few complete core-set items for ICC(2,1)."}
    result["level2_icc"] = icc_results

    return result

def compute_convergent_validity(item_summary: pd.DataFrame) -> dict:

    valid = item_summary[
        ~item_summary["is_negative_control"].fillna(False).astype(bool)
    ].copy()

    valid = valid.dropna(
        subset=[
            "mean_faithfulness",
            "reconstructed_mean_entailment",
            "reconstructed_faithfulness_rate",
        ]
    )

    return {
        "vs_mean_entailment": pearson_spearman(
            valid["mean_faithfulness"].to_numpy(dtype=float),
            valid["reconstructed_mean_entailment"].to_numpy(dtype=float),
        ),
        "vs_auto_faithfulness_rate": pearson_spearman(
            valid["mean_faithfulness"].to_numpy(dtype=float),
            valid["reconstructed_faithfulness_rate"].to_numpy(dtype=float),
        ),
    }

def compute_sentence_level_crosstab(sentence_summary: pd.DataFrame):
    if sentence_summary is None or sentence_summary.empty or "auto_status" not in sentence_summary.columns:
        return None
    valid = sentence_summary.dropna(subset=["majority_tag", "auto_status"])
    if valid.empty:
        return None
    return pd.crosstab(valid["majority_tag"], valid["auto_status"])

def compute_rater_profiles(resp_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in resp_df.groupby("rater_id"):
        tag_counts = g["level1_tag"].value_counts(normalize=True)
        row = {"rater_id": rid, "n_sentence_tags": g["level1_tag"].notna().sum()}
        for cat in LEVEL1_CATEGORIES:
            row[f"tag_{cat}_pct"] = round(tag_counts.get(cat, 0.0) * 100, 1)
        l2 = g.groupby(["item_id", "rater_id"], as_index=False).first()
        for dim in LEVEL2_DIMS:
            row[f"mean_{dim}"] = round(l2[dim].mean(), 2) if l2[dim].notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)

def compute_negative_control_detection(resp_df: pd.DataFrame, key_df: pd.DataFrame) -> dict:
    nc_ids = set(key_df[key_df["is_negative_control"]].index)
    nc_resp = resp_df[resp_df["item_id"].isin(nc_ids)]
    if nc_resp.empty:
        return {"error": "No negative-control responses are available."}

    per_response = []
    for (item_id, rater_id), g in nc_resp.groupby(["item_id", "rater_id"]):
        tags = g["level1_tag"].dropna()
        majority_unsupported = (not tags.empty) and (tags.mode().iloc[0] == "unsupported")
        faith = g["faithfulness"].dropna()
        low_faithfulness = (not faith.empty) and (faith.iloc[0] <= 2)
        detected = bool(majority_unsupported or low_faithfulness)
        per_response.append({"item_id": item_id, "rater_id": rater_id, "detected": detected})

    per_df = pd.DataFrame(per_response)
    detection_rate = per_df["detected"].mean()
    by_rater = per_df.groupby("rater_id")["detected"].mean()
    return {
        "overall_detection_rate": float(detection_rate),
        "n_control_responses": len(per_df),
        "by_rater": by_rater.to_dict(),
        "gate_passed": bool(detection_rate >= NEGATIVE_CONTROL_GATE),
    }

def find_discrepancy_candidates(item_summary: pd.DataFrame,
                                 auto_high=0.7, expert_low=2.0,
                                 auto_low=0.3, expert_high=4.0) -> pd.DataFrame:
    valid = item_summary.dropna(subset=["mean_faithfulness", "faithfulness_rate"])
    auto_missed = valid[
        (valid["faithfulness_rate"] >= auto_high) & (valid["mean_faithfulness"] <= expert_low)
    ].copy()
    auto_missed["discrepancy_type"] = "under_detection"

    auto_overflagged = valid[
        (valid["faithfulness_rate"] <= auto_low) & (valid["mean_faithfulness"] >= expert_high)
    ].copy()
    auto_overflagged["discrepancy_type"] = "over_flag"

    return pd.concat([auto_missed, auto_overflagged], ignore_index=True)

def find_sentence_discrepancy_candidates(sentence_summary: pd.DataFrame) -> pd.DataFrame:
    if sentence_summary is None or sentence_summary.empty or "auto_status" not in sentence_summary.columns:
        return pd.DataFrame()
    valid = sentence_summary.dropna(subset=["majority_tag", "auto_status"]).copy()
    if valid.empty:
        return pd.DataFrame()

    auto_pass_like = valid["auto_status"].isin(["PASS", "PASS_NO_SIGNAL"])
    auto_fail_content = valid["auto_status"].isin(["FAIL_NLI", "FAIL_INVALID_REF"])

    missed = valid[auto_pass_like & (valid["majority_tag"] == "unsupported")].copy()
    missed["discrepancy_type"] = "under_detection"

    overflagged = valid[auto_fail_content & (valid["majority_tag"] == "supported")].copy()
    overflagged["discrepancy_type"] = "over_flag"

    return pd.concat([missed, overflagged], ignore_index=True)

def format_report(
    item_summary,
    sentence_summary,
    reliability,
    convergent,
    sentence_crosstab,
    nc_detection,
    item_discrepancies,
    sentence_discrepancies,
    rater_profiles,
) -> str:
    lines = [
        "MetaKT-Verba rating analysis",
        "=" * 60,
        f"items_with_responses={len(item_summary)}",
        f"mean_raters_per_item={item_summary['n_raters'].mean():.2f}",
        "",
        "[Rater profiles]",
        rater_profiles.to_string(index=False),
        "",
        "[Inter-rater reliability]",
    ]

    fk = reliability.get("level1_fleiss_kappa", {})
    if "error" in fk:
        lines.append(f"Fleiss kappa: {fk['error']}")
    else:
        lines.append(
            f"Fleiss kappa={fk['kappa']:.3f}, "
            f"n_subjects={fk['n_subjects']}, n_raters={fk['n_raters_per_subject']}"
        )

    wk = reliability.get("level1_weighted_kappa", {})
    if wk.get("weighted_kappa") is not None:
        lines.append(
            f"Mean pairwise weighted kappa={wk['weighted_kappa']:.3f}, "
            f"n_rater_pairs={wk['n_rater_pairs']}"
        )

    for dim, result in reliability.get("level2_icc", {}).items():
        if "error" in result:
            lines.append(f"ICC(2,1) {dim}: {result['error']}")
        else:
            lines.append(
                f"ICC(2,1) {dim}={result['icc']:.3f}, "
                f"n_items={result['n_targets']}, n_raters={result['k_raters']}"
            )

    lines.extend(["", "[Convergent validity]"])
    for label, key in [
        ("mean_entailment", "vs_mean_entailment"),
        ("faithfulness_rate", "vs_auto_faithfulness_rate"),
    ]:
        result = convergent.get(key, {})
        if result.get("n", 0) < 3:
            lines.append(f"{label}: insufficient data (n={result.get('n', 0)})")
        else:
            lines.append(
                f"{label}: Pearson r={result['pearson_r']:.3f}, p={result['pearson_p']:.3g}; "
                f"Spearman rho={result['spearman_rho']:.3f}, p={result['spearman_p']:.3g}; "
                f"n={result['n']}"
            )

    lines.extend(["", "[Sentence-level expert vs automatic status]"])
    lines.append("n/a" if sentence_crosstab is None else sentence_crosstab.to_string())

    lines.extend(["", "[Negative controls]"])
    if "error" in nc_detection:
        lines.append(nc_detection["error"])
    else:
        lines.append(
            f"detection_rate={nc_detection['overall_detection_rate']:.3f}, "
            f"n={nc_detection['n_control_responses']}, gate_passed={nc_detection['gate_passed']}"
        )

    lines.extend([
        "",
        f"item_discrepancies={len(item_discrepancies)}",
        f"sentence_discrepancies={len(sentence_discrepancies)}",
    ])
    return "\n".join(lines)

def run(responses_dir, answer_key_path, sentence_answer_key_path, output_dir, exclude_raters=None,
    verifier_scores_path=None):
    os.makedirs(output_dir, exist_ok=True)
    resp_df = load_responses(responses_dir)
    if exclude_raters:
        before_n = resp_df["rater_id"].nunique()
        resp_df = resp_df[~resp_df["rater_id"].isin(exclude_raters)].copy()
        print(
            f"[sensitivity] excluded={exclude_raters}, "
            f"raters={before_n}->{resp_df['rater_id'].nunique()}, rows={len(resp_df)}"
        )
    key_df = load_answer_key(answer_key_path)
    sentence_key_df = load_sentence_answer_key(sentence_answer_key_path) if sentence_answer_key_path else None

    merged = resp_df.join(key_df, on="item_id", how="left")
    merged.to_csv(os.path.join(output_dir, "merged_responses.csv"), index=False)

    item_summary = build_item_level_summary(resp_df, key_df)

    if verifier_scores_path:
        item_summary = attach_reconstructed_item_scores(
            item_summary,
            verifier_scores_path,
        )

    item_summary.to_csv(
        os.path.join(output_dir, "item_level_summary.csv"),
        index=False,
    )

    sentence_summary = build_sentence_level_summary(resp_df, sentence_key_df)
    if not sentence_summary.empty:
        sentence_summary.to_csv(os.path.join(output_dir, "sentence_level_summary.csv"), index=False)

    reliability = compute_core_set_reliability(resp_df, key_df)
    convergent = compute_convergent_validity(item_summary)
    sentence_crosstab = compute_sentence_level_crosstab(sentence_summary)
    nc_detection = compute_negative_control_detection(resp_df, key_df)
    item_discrepancies = find_discrepancy_candidates(item_summary)
    sentence_discrepancies = find_sentence_discrepancy_candidates(sentence_summary)
    rater_profiles = compute_rater_profiles(resp_df)
    rater_profiles.to_csv(os.path.join(output_dir, "rater_profiles.csv"), index=False)

    pd.concat([
        item_discrepancies.assign(level="item"),
        sentence_discrepancies.assign(level="sentence"),
    ], ignore_index=True).to_csv(os.path.join(output_dir, "discrepancy_candidates.csv"), index=False)

    report = format_report(item_summary, sentence_summary, reliability, convergent, sentence_crosstab,
                            nc_detection, item_discrepancies, sentence_discrepancies, rater_profiles)
    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved rating-analysis outputs to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate expert ratings and reproduce the human-evaluation analyses."
    )
    parser.add_argument("--responses-dir", default="data/ratings")
    parser.add_argument("--answer-key", default="data/stimuli/internal_answer_key.csv")
    parser.add_argument(
        "--sentence-answer-key",
        default="data/stimuli/internal_answer_key_sentences.csv",
    )
    parser.add_argument(
        "--verifier-scores",
        default="data/results/verifier_family/scores_all_verifiers.csv",
        help="Verifier-family sentence scores in CSV or PKL format.",
    )
    parser.add_argument("--output-dir", default="outputs/ratings_analysis")
    parser.add_argument(
        "--exclude-raters",
        nargs="+",
        default=None,
        help="Re-run after excluding the listed rater IDs.",
    )
    args = parser.parse_args()

    run(
        args.responses_dir,
        args.answer_key,
        args.sentence_answer_key,
        args.output_dir,
        verifier_scores_path=args.verifier_scores,
    )

    if args.exclude_raters:
        tag = "_".join(args.exclude_raters)
        sensitivity_dir = os.path.join(args.output_dir, f"sensitivity_excl_{tag}")
        print(f"[sensitivity] excluding raters: {args.exclude_raters}")
        run(
            args.responses_dir,
            args.answer_key,
            args.sentence_answer_key,
            sensitivity_dir,
            exclude_raters=args.exclude_raters,
            verifier_scores_path=args.verifier_scores,
        )

if __name__ == "__main__":
    main()

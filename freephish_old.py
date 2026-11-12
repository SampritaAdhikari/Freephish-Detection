#!/usr/bin/env python3
"""Simplified FreePhish re-implementation (Roy, Karanjit, Nilizadeh, IMC '23). Deviations: paper's labeled
FWB dataset is not public so metrics won't match the paper's; the 8 URL + 12 HTML features of Li et al.
(2019) are my approximation; screenshots are not captured.

  python freephish_old.py download --per-class 5000 --out sample.parquet
  python freephish_old.py train --csv data.csv --url-col url --label-col label   (STAGE 1, saves results_baseline.json)
  # STAGE 2 is freephish_enhanced.py train (saves results_enhanced.json)
  python freephish_old.py compare   (STAGE 3: compares the two saved results files)
  python freephish_old.py monitor --model model.joblib --source urls.txt --once
  python freephish_old.py analyze --log detections.jsonl --obs observations.jsonl
"""
import argparse, base64, hashlib, json, os, re, sys, time, warnings
from urllib.parse import urlparse

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")
UA = {"User-Agent": "Mozilla/5.0 (research; FreePhish-simplified)"}

# FWB host suffixes (paper Table 4 service names; domain strings are my own best guesses)
FWB_SUFFIXES = ["weebly.com", "000webhostapp.com", "blogspot.com", "wixsite.com", "wix.com", "sites.google.com",
                 "github.io", "firebaseapp.com", "web.app", "square.site", "wordpress.com", "forms.gle",
                 "docs.google.com", "sharepoint.com", "yolasite.com", "godaddysites.com", "mailchimpsites.com",
                 "glitch.me", "hpage.com", "zohopublic.com"]
BRANDS = ["paypal", "microsoft", "office365", "office", "facebook", "amazon", "apple", "chase", "irs", "ebay",
          "rakuten", "smbc", "fedex", "google", "netflix", "att", "dropbox", "onedrive"]
SENSITIVE = ["login", "signin", "sign-in", "verify", "secure", "account", "update", "password", "bank",
             "confirm", "billing", "wallet", "support"]
BANNER_WORDS = ["powered by", "weebly", "wix", "create your own", "blogger", "godaddy", "yola", "mailchimp"]

URL_FEATS = ["url_len", "n_special", "n_sensitive", "n_brand", "has_ip", "digit_ratio", "n_subdomains", "path_depth"]
HTML_FEATS = ["n_internal_links", "n_external_links", "n_empty_links", "has_login_form", "html_len", "n_forms",
              "n_iframes", "n_scripts", "n_inputs", "n_hidden_inputs", "n_images", "title_len"]
FWB_FEATS = ["hidden_banner", "has_noindex"]
ALL_FEATS = URL_FEATS + HTML_FEATS + FWB_FEATS  # paper also drops 'https'/'multiple TLD' (constant on FWB sites)


def is_fwb(url):
    host = (urlparse(url if "//" in url else "//" + url).hostname or "").lower()
    return any(host == s or host.endswith("." + s) for s in FWB_SUFFIXES)


def url_features(url):
    p = urlparse(url if "//" in url else "//" + url)
    host, path, low = (p.hostname or ""), (p.path or ""), url.lower()
    return {
        "url_len": len(url), "n_special": sum(url.count(c) for c in "@-_~%=&"),
        "n_sensitive": sum(w in low for w in SENSITIVE), "n_brand": sum(b in low for b in BRANDS),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "n_subdomains": max(host.count(".") - 1, 0), "path_depth": len([x for x in path.split("/") if x]),
    }


def html_features(html, url, soup=None):
    f = dict.fromkeys(HTML_FEATS + FWB_FEATS, 0)
    if not html:
        return f
    soup = soup or BeautifulSoup(html, "html.parser")
    host = (urlparse(url).hostname or "").lower()
    internal = external = empty = 0
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if href in ("", "#", "javascript:void(0)", "javascript:;"):
            empty += 1
        elif href.startswith("http") and urlparse(href).hostname != host:
            external += 1
        else:
            internal += 1
    inputs = soup.find_all("input")
    f.update({
        "n_internal_links": internal, "n_external_links": external, "n_empty_links": empty,
        "has_login_form": int(any((i.get("type") or "").lower() == "password" for i in inputs)),
        "html_len": len(html), "n_forms": len(soup.find_all("form")), "n_iframes": len(soup.find_all("iframe")),
        "n_scripts": len(soup.find_all("script")), "n_inputs": len(inputs),
        "n_hidden_inputs": sum((i.get("type") or "").lower() == "hidden" for i in inputs),
        "n_images": len(soup.find_all("img")),
        "title_len": len(soup.title.string) if soup.title and soup.title.string else 0,
    })
    for el in soup.find_all(style=True):  # feature 1: obfuscated/hidden FWB banner
        st = re.sub(r"\s", "", el["style"].lower())
        if "visibility:hidden" in st or "display:none" in st:
            blob = (el.get_text(" ", strip=True) + " " + str(el.get("id", "")) + " " + " ".join(el.get("class", []))).lower()
            if any(w in blob for w in BANNER_WORDS):
                f["hidden_banner"] = 1
                break
    for m in soup.find_all("meta"):  # feature 2: <meta name="robots" content="...noindex...">
        if (m.get("name") or "").lower() in ("robots", "googlebot") and "noindex" in (m.get("content") or "").lower():
            f["has_noindex"] = 1
            break
    return f


def fetch_html(url, snapshot_dir=None, timeout=10):
    if "//" not in url:
        url = "http://" + url
    try:
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
        html = r.text if r.status_code < 400 else ""
    except Exception:
        return ""
    if html and snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
        with open(os.path.join(snapshot_dir, hashlib.sha1(url.encode()).hexdigest() + ".html"), "w", encoding="utf-8") as fh:
            fh.write(html)
    return html


def featurize(url, html):
    d = url_features(url); d.update(html_features(html, url))
    return [d[k] for k in ALL_FEATS]


class StackModel:
    """Two-layer stack: layer 1 = GBDT/XGBoost/LightGBM out-of-fold probs via K-fold; layer 2 = final GBDT on
    [features, layer-1 probs, majority vote]."""

    def __init__(self, n_splits=5, seed=42):
        self.n_splits, self.seed = n_splits, seed

    def _base(self):
        s = self.seed
        return [GradientBoostingClassifier(random_state=s),
                XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=s, verbosity=0),
                LGBMClassifier(n_estimators=200, random_state=s, verbose=-1)]

    @staticmethod
    def _stack(X, probs):
        vote = (np.mean(np.round(probs), axis=1) > 0.5).astype(float)[:, None]
        return np.hstack([X, probs, vote])

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        oof = np.zeros((len(X), 3))
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(X, y):
            for j, m in enumerate(self._base()):
                m.fit(X[tr], y[tr]); oof[va, j] = m.predict_proba(X[va])[:, 1]
        self.base_ = [m.fit(X, y) for m in self._base()]  # refit on all training data for inference
        self.final_ = GradientBoostingClassifier(random_state=self.seed).fit(self._stack(X, oof), y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_])
        return self.final_.predict_proba(self._stack(X, probs))[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ------------------------------------------------------------------ commands
HTML_COL_CANDIDATES = ["html", "html_content", "content", "page_source", "source", "raw_html", "body"]
POS_DEFAULT = {"phishing", "phish", "malicious", "bad", "1", "1.0", "true"}


def load_table(path):
    if not os.path.exists(path):
        folder = os.path.dirname(path) or "."
        near = sorted(f for f in os.listdir(folder) if f.endswith((".parquet", ".csv")))
        raise SystemExit(f"FILE NOT FOUND: '{path}' (current folder: {os.getcwd()}). Data files here: {near or 'none'}.\n"
                         "If this is sample.parquet, the download step did not run or did not finish: run the download cell again "
                         "and wait for the line 'saved ... rows to sample.parquet' before training.")
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def find_html_col(columns, wanted=None):
    if wanted:
        if wanted not in columns:
            raise SystemExit(f"--html-col '{wanted}' not in columns: {list(columns)}")
        return wanted
    hits = [c for c in HTML_COL_CANDIDATES if c in columns]
    return hits[0] if hits else None


def to_binary(series, pos_label=None):
    """Map labels to 1=phishing. Never guesses silently: prints values, stops if ambiguous."""
    s = series.astype(str).str.strip().str.lower()
    vals = sorted(s.unique())
    print("label values found:", vals)
    if pos_label is not None:
        pos = {str(pos_label).strip().lower()}
        if not pos <= set(vals):
            raise SystemExit(f"--pos-label {pos_label!r} not among label values {vals}")
    else:
        pos = POS_DEFAULT & set(vals)
        if len(pos) != 1 or len(vals) != 2:
            raise SystemExit(f"Cannot infer which label means phishing from {vals}. Re-run with --pos-label <value>.")
        if pos <= {"1", "1.0", "true"}:
            print("WARNING: numeric labels - assumed 1 = phishing (some datasets use 1 = legitimate; pass --pos-label if unsure).")
    print("positive (phishing) label(s):", pos)
    return s.isin(pos).astype(int)


def _feat_row(u, h, max_chars):
    t0 = time.perf_counter()
    return featurize(u, (h or "")[:max_chars]), time.perf_counter() - t0


def group_key(url):
    """Leakage-control group: non-FWB = last two host labels (approximation); FWB = full hostname."""
    host = (urlparse(url if "//" in url else "//" + url).hostname or "").lower()
    return host if is_fwb(url) else ".".join(host.split(".")[-2:])


def prepare(df, a, html_col):
    """Label mapping, filtering, dedup, balancing -> (urls, htmls, y)."""
    df = df.dropna(subset=[a.url_col, a.label_col]).copy()
    df["_y"] = to_binary(df[a.label_col], a.pos_label)
    n0 = len(df)
    df = df.drop_duplicates(subset=[a.url_col])
    if a.dedup and html_col:
        h = df[html_col].fillna("").astype(str)
        key = h.map(lambda x: hashlib.sha1(x.encode("utf-8", "ignore")).hexdigest())
        df = df[~(key.duplicated() & (h != ""))]
    print(f"dedup: {n0} -> {len(df)} rows")
    if a.fwb_only:
        df = df[df[a.url_col].astype(str).map(is_fwb)]
        print(f"FWB-only filter -> {len(df)} rows; per class: {df['_y'].value_counts().to_dict()}")
        if df["_y"].nunique() < 2 or df["_y"].value_counts().min() < 30:
            raise SystemExit("Too few FWB samples in one class to train. Download more shards or drop --fwb-only.")
    if a.balance:
        n = df["_y"].value_counts().min()
        df = df.groupby("_y", group_keys=False).sample(n, random_state=42)
    if a.max_rows and len(df) > a.max_rows:
        df = df.groupby("_y", group_keys=False).sample(a.max_rows // 2, random_state=42)
    print("class counts:", df["_y"].value_counts().to_dict())
    if len(df) < 100 or df["_y"].value_counts().min() < 20:
        raise SystemExit(f"Only {len(df)} usable rows (class counts {df['_y'].value_counts().to_dict()}) - too few to train/evaluate. "
                         "Try without --dedup/--fwb-only, or download more data.")
    htmls = df[html_col].fillna("").astype(str).tolist() if html_col else [""] * len(df)
    return df[a.url_col].astype(str).tolist(), htmls, df["_y"].values


def build_features(urls, htmls, a, cache=None):
    key = hashlib.sha1("\n".join(urls).encode()).hexdigest()
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if str(z["key"]) == key and "ft" in z.files:
            print("loaded cached features:", cache)
            return z["X"], z["ft"]
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs)(delayed(_feat_row)(u, h, a.max_html_chars) for u, h in zip(urls, htmls))
    X, ft = np.array([r[0] for r in res], float), np.array([r[1] for r in res], float)
    print(f"feature extraction {time.time() - t0:.1f}s wall-clock")
    if cache:
        np.savez(cache, X=X, ft=ft, key=key)
    return X, ft


def split_idx(y, groups, mode, seed=42):
    if mode == "group":
        return next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(np.zeros(len(y)), y, groups))
    return train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)  # paper: 70/30


def time_inference(model, X, ft, cap=0):
    """Per-URL runtime = feature extraction (measured earlier) + single-sample inference; page FETCH time excluded."""
    n = len(X) if not cap else min(cap, len(X))
    inf = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter(); model.predict(X[i:i + 1]); inf[i] = time.perf_counter() - t0
    per_url = ft[:n] + inf
    return float(per_url.sum()), float(np.median(per_url)), n


def report(name, y, pred, timing=None):
    t = f" | total {timing[0]:.2f}s  median {timing[1]:.4f}s/URL (n={timing[2]})" if timing else ""
    print(f"  {name:<28} acc {accuracy_score(y, pred):.3f}  prec {precision_score(y, pred, zero_division=0):.3f}  "
          f"rec {recall_score(y, pred, zero_division=0):.3f}  F1 {f1_score(y, pred, zero_division=0):.3f}{t}")


def cmd_train(a):
    df = load_table(a.csv)
    print("columns:", df.columns.tolist(), "| rows:", len(df))
    for c in (a.url_col, a.label_col):
        if c not in df.columns:
            raise SystemExit(f"column '{c}' not in {df.columns.tolist()}")
    html_col = find_html_col(df.columns, a.html_col)
    urls, htmls, y = prepare(df, a, html_col)
    print("using HTML column:", html_col) if html_col else print("WARNING: no HTML -> HTML/FWB features are zero (URL-only).")
    X, ft = build_features(urls, htmls, a, a.cache)
    groups = np.array([group_key(u) for u in urls])
    print(f"unique domain groups: {len(set(groups))}")
    tr, te = split_idx(y, groups, a.split)
    overlap = np.mean([g in set(groups[tr]) for g in groups[te]])
    print(f"\n== Main run ({a.split} split) | test rows sharing a domain with train: {overlap:.1%} ==")
    print("---- WITHOUT enhanced features (original 22 FreePhish-style features) ----")
    t0 = time.time(); model = StackModel().fit(X[tr], y[tr]); t1 = time.time()
    pred = model.predict(X[te])
    print(f"train time {t1 - t0:.1f}s")
    base_timing = time_inference(model, X[te], ft[te], a.timing_samples)
    report(f"StackModel n_test={len(te)}", y[te], pred, base_timing)
    if a.results_out:
        save_results(a.results_out, "baseline", np.asarray(urls)[te], y[te], pred, model.predict_proba(X[te]), base_timing,
                     {"feature_set": "original 22 (paper-style baseline)", "n_features": len(ALL_FEATS), "split": a.split,
                      "seed": 42, "n_rows_used": len(urls), "n_train": len(tr), "n_test": len(te),
                      "test_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y[te], return_counts=True))},
                      "domain_overlap": float(overlap), "csv": a.csv, "balance": a.balance, "dedup": a.dedup,
                      "fwb_only": a.fwb_only, "max_rows": a.max_rows})
    joblib.dump(model, a.model)
    print("saved", a.model)


def save_results(path, name, urls_te, y_te, pred, proba, timing, meta):
    """Stage output: metrics JSON + per-row test predictions CSV (so stages can be compared on identical rows)."""
    import datetime
    from sklearn.metrics import roc_auc_score
    y_te = np.asarray(y_te)
    m = {"accuracy": float(accuracy_score(y_te, pred)), "precision": float(precision_score(y_te, pred, zero_division=0)),
         "recall": float(recall_score(y_te, pred, zero_division=0)), "f1": float(f1_score(y_te, pred, zero_division=0)),
         "auc": float(roc_auc_score(y_te, proba)) if len(set(y_te.tolist())) > 1 else None,
         "total_time_s": float(timing[0]), "median_runtime_s": float(timing[1]), "timed_n": int(timing[2])}
    out = {"name": name, "created": datetime.datetime.now().isoformat(timespec="seconds"), "metrics": m, "meta": meta}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pp = os.path.splitext(path)[0] + ".predictions.csv"
    pd.DataFrame({"url": list(urls_te), "y_true": y_te, "y_pred": np.asarray(pred), "p_phish": np.asarray(proba)}).to_csv(pp, index=False)
    print(f"saved results -> {path} and {pp}")


def cmd_compare(a):
    """STAGE 3: compare the saved baseline and enhanced results (only meaningful if both used the identical test rows)."""
    A, B = json.load(open(a.baseline)), json.load(open(a.enhanced))
    for tag, R in (("BASELINE (without enhanced)", A), ("ENHANCED", B)):
        mt = R["meta"]
        print(f"{tag}: {mt.get('feature_set')} | split={mt.get('split')} seed={mt.get('seed')} | "
              f"train {mt.get('n_train')} / test {mt.get('n_test')} | data={mt.get('csv')} | created {R['created']}")
    ma, mb = A["metrics"], B["metrics"]
    print("\n== SIDE-BY-SIDE ==")
    print(f"{'metric':<28}{'WITHOUT enhanced':>18}{'WITH enhanced':>16}{'difference':>13}")
    for nm, k in (("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1")):
        if ma.get(k) is not None and mb.get(k) is not None:
            print(f"{nm:<28}{ma[k]:>18.4f}{mb[k]:>16.4f}{mb[k] - ma[k]:>+13.4f}")
    print(f"{'Total time (s)':<28}{ma['total_time_s']:>18.2f}{mb['total_time_s']:>16.2f}{mb['total_time_s'] - ma['total_time_s']:>+13.2f}")
    print(f"{'Median runtime (s/URL)':<28}{ma['median_runtime_s']:>18.4f}{mb['median_runtime_s']:>16.4f}{mb['median_runtime_s'] - ma['median_runtime_s']:>+13.4f}")


def cmd_download(a):
    """Stream a PhreshPhish-style parquet dataset and save a balanced sample. Reads one row group at a time
    so 36 GB never has to fit in RAM. Shard order is unknown, so sampling is NOT guaranteed random."""
    import glob
    import pyarrow.parquet as pq
    if a.local:
        files = sorted(glob.glob(a.local))
    else:
        from huggingface_hub import HfApi, hf_hub_download
        allf = [f for f in HfApi().list_repo_files(a.repo, repo_type="dataset") if f.endswith(".parquet")]
        print(f"{len(allf)} parquet files in repo; first few: {allf[:5]}")
        if a.pattern:
            allf = [f for f in allf if a.pattern in f]
        files = allf[: a.max_shards]
    if not files:
        raise SystemExit("no parquet files selected (check --pattern / --local)")
    got, label_values_seen = {0: [], 1: []}, None
    for f in files:
        path = f if a.local else hf_hub_download(a.repo, repo_type="dataset", filename=f)
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        print("shard:", f, "| columns:", cols, "| rows:", pf.metadata.num_rows)
        html_col = find_html_col(cols, a.html_col)
        use = [c for c in (a.url_col, a.label_col, html_col) if c]
        for rg in range(pf.num_row_groups):
            d = pf.read_row_group(rg, columns=use).to_pandas()
            if label_values_seen is None:
                print("first row group label values:", sorted(d[a.label_col].astype(str).unique()))
            d["_y"] = d[a.label_col].astype(str).str.strip().str.lower().isin(
                {str(a.pos_label).lower()} if a.pos_label else POS_DEFAULT).astype(int)
            label_values_seen = True
            if a.fwb_only:
                d = d[d[a.url_col].astype(str).map(is_fwb)]
            for y in (0, 1):
                need = a.per_class - sum(len(x) for x in got[y])
                if need > 0:
                    got[y].append(d[d["_y"] == y].head(need))
            if all(sum(len(x) for x in got[y]) >= a.per_class for y in (0, 1)):
                break
        print("collected so far:", {y: sum(len(x) for x in got[y]) for y in (0, 1)})
        if all(sum(len(x) for x in got[y]) >= a.per_class for y in (0, 1)):
            break
    if sum(len(x) for x in got[1]) == 0 or sum(len(x) for x in got[0]) == 0:
        raise SystemExit("One class is empty: check the label values printed above and pass --pos-label <phishing value> "
                         "(or, with --fwb-only, raise --max-shards: FWB pages may be rare).")
    out = pd.concat([x for y in (0, 1) for x in got[y]], ignore_index=True).drop(columns="_y")
    if len(out) == 0:
        raise SystemExit("nothing collected - check --pos-label and column names above")
    mb = out.memory_usage(deep=True).sum() / 1e6
    print(f"collected {len(out)} rows (~{mb:.0f} MB). Writing {a.out} in chunks (progress shown) - DO NOT interrupt "
          "until 'saved' appears...", flush=True)
    t0 = time.time()
    import pyarrow as pa
    import pyarrow.parquet as pq
    chunk = max(1, min(500, len(out) // 20 or len(out)))  # ~20 progress updates regardless of size
    writer = None
    try:
        for i in range(0, len(out), chunk):
            part = out.iloc[i:i + chunk]
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(a.out, table.schema, compression="snappy")  # snappy: faster than zstd, avoids long silent pauses
            writer.write_table(table)
            print(f"  ...written {min(i + chunk, len(out))}/{len(out)} rows ({time.time() - t0:.0f}s elapsed)", flush=True)
    finally:
        if writer is not None:
            writer.close()
    print(f"saved {len(out)} rows to {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB, {time.time() - t0:.0f}s); "
          f"label counts:", out[a.label_col].value_counts().to_dict())


def classify(model, url, snapshot_dir=None):
    html = fetch_html(url, snapshot_dir)
    return float(model.predict_proba([featurize(url, html)])[0]), bool(html)


def cmd_monitor(a):
    """Streaming module stand-in: re-reads a URL file every --interval seconds (paper: 10 min)."""
    model, seen = joblib.load(a.model), set()
    while True:
        if os.path.exists(a.source):
            new = (u for u in (l.strip() for l in open(a.source)) if u and u not in seen and is_fwb(u))
            for u in new:
                seen.add(u)
                p, got = classify(model, u, a.snapshots)
                rec = {"url": u, "p_phish": p, "phishing": p >= .5, "html_fetched": got, "first_seen": time.time()}
                with open(a.out, "a") as fh: fh.write(json.dumps(rec) + "\n")
                print(("PHISH " if rec["phishing"] else "ok    ") + f"{p:.3f} {u}")
        if a.once:
            break
        time.sleep(a.interval)


def vt_detections(url, key, exclude):
    uid = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
    r = requests.get(f"https://www.virustotal.com/api/v3/urls/{uid}", headers={"x-apikey": key}, timeout=20)
    if r.status_code != 200:
        return None
    res = r.json()["data"]["attributes"]["last_analysis_results"]
    return sum(v["category"] in ("malicious", "suspicious") for k, v in res.items() if k.lower() not in exclude)


def cmd_analyze(a):
    """Liveness + VirusTotal engine detections for flagged URLs, run every ~10 min."""
    key = os.environ.get("VT_API_KEY")
    exclude = {"google safebrowsing", "phishtank", "openphish"}
    now = time.time()
    if not os.path.exists(a.obs):
        open(a.obs, "a").close()  # create up front: else zero matches below would never write the file -> FileNotFoundError downstream
    n_checked = n_skipped_not_phishing = n_skipped_too_old = 0
    for line in open(a.log):
        r = json.loads(line)
        if not r["phishing"]:
            n_skipped_not_phishing += 1; continue
        if now - r["first_seen"] > a.max_age_h * 3600:
            n_skipped_too_old += 1; continue
        try:
            alive = requests.get(r["url"], headers=UA, timeout=10).status_code < 400
        except Exception:
            alive = False
        det = vt_detections(r["url"], key, exclude) if key else None
        with open(a.obs, "a") as fh: fh.write(json.dumps({"url": r["url"], "t": now, "alive": alive, "vt_detections": det}) + "\n")
        n_checked += 1
        if key:
            time.sleep(a.vt_sleep)  # respect VT rate limits
    print(f"analyze: checked liveness for {n_checked} URL(s) flagged phishing; skipped {n_skipped_not_phishing} "
          f"not flagged phishing and {n_skipped_too_old} older than --max-age-h ({a.max_age_h}h). "
          f"{a.obs} now has {sum(1 for _ in open(a.obs))} line(s) total.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("download", help="stream a parquet dataset (e.g. phreshphish/phreshphish) and save a balanced sample")
    d.add_argument("--repo", default="phreshphish/phreshphish"); d.add_argument("--local", help="glob of local parquet files (skips HF)")
    d.add_argument("--pattern", help="substring filter on parquet file paths, e.g. 'train'")
    d.add_argument("--max-shards", type=int, default=2); d.add_argument("--per-class", type=int, default=5000)
    d.add_argument("--url-col", default="url"); d.add_argument("--label-col", default="label")
    d.add_argument("--html-col"); d.add_argument("--pos-label"); d.add_argument("--fwb-only", action="store_true")
    d.add_argument("--out", default="sample.parquet")
    t = sp.add_parser("train")
    t.add_argument("--csv", required=True, help="csv or parquet"); t.add_argument("--url-col", default="url")
    t.add_argument("--label-col", default="label"); t.add_argument("--html-col"); t.add_argument("--pos-label")
    t.add_argument("--balance", action="store_true"); t.add_argument("--max-rows", type=int)
    t.add_argument("--fwb-only", action="store_true"); t.add_argument("--n-jobs", type=int, default=-1)
    t.add_argument("--max-html-chars", type=int, default=500000)
    t.add_argument("--split", choices=["group", "random"], default="group", help="group = no domain shared between train/test")
    t.add_argument("--results-out", default="results_baseline.json", help="STAGE 1 output; compare later with `compare`")
    t.add_argument("--timing-samples", type=int, default=0, help="time only first N test rows (0 = all)")
    t.add_argument("--dedup", action="store_true", help="also drop exact-duplicate HTML")
    t.add_argument("--cache", help="npz path to cache extracted features")
    t.add_argument("--snapshots", default="snapshots"); t.add_argument("--model", default="model.joblib")
    m = sp.add_parser("monitor"); m.add_argument("--model", default="model.joblib")
    m.add_argument("--source", required=True); m.add_argument("--out", default="detections.jsonl")
    m.add_argument("--interval", type=int, default=600); m.add_argument("--once", action="store_true")
    m.add_argument("--snapshots", default="snapshots")
    cp = sp.add_parser("compare", help="STAGE 3: compare saved baseline vs enhanced results")
    cp.add_argument("--baseline", default="results_baseline.json"); cp.add_argument("--enhanced", default="results_enhanced.json")
    an = sp.add_parser("analyze"); an.add_argument("--log", default="detections.jsonl")
    an.add_argument("--obs", default="observations.jsonl"); an.add_argument("--max-age-h", type=float, default=168)
    an.add_argument("--vt-sleep", type=float, default=15)
    a = ap.parse_args()
    {"download": cmd_download, "compare": cmd_compare, "train": cmd_train, "monitor": cmd_monitor, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()

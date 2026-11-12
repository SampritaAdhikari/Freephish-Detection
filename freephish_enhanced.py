#!/usr/bin/env python3
"""FreePhish ENHANCED feature extraction ('Algorithm 1' / Section 4.1 spec). Requires freephish_old.py in the
same folder. Simplified build: only `train`/`benchmark` are kept (collect/predict/monitor/--fetch-html/
--compare-groups were never used - the data always came from the downloaded parquet). Feature extraction
itself (all groups, including the screenshot-only 'visual' sub-features) is untouched, so a trained model
here is identical to the full version's.

Algorithm 1 mapping: X(W) = [F_FreePhish, F_URL, F_HTML, F_FWB, F_Behavior, F_Visual, F_Brand].
  freephish/url/html/fwb/brand: static, from URL + HTML. behavior: static HTML proxies (+ live redirect
  features, always NaN here since pages are never fetched). visual: HTML/CSS proxies (+ screenshot stats,
  always NaN here since no screenshots are ever supplied). Preprocessor: median-impute + log1p counts +
  standardize, fit on TRAIN split only. Then split -> StackModel -> accuracy/precision/recall/F1.

  python freephish_enhanced.py train --csv sample.parquet --cache feat_enh.npz
  python freephish_enhanced.py benchmark --csv sample.parquet --balance --dedup --repeats 5 --fast
(download/analyze/compare: use freephish_old.py)
"""
import argparse, hashlib, json, math, os, re, time, warnings
from collections import Counter
from urllib.parse import urlparse, parse_qsl

import joblib
import numpy as np
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import freephish_old as fp

# --------------------------------------------------------------------------- feature registry
GROUP_ORDER = ["freephish", "url", "html", "fwb", "brand", "behavior", "visual"]
GROUPS = {
    "freephish": fp.URL_FEATS + fp.HTML_FEATS + fp.FWB_FEATS,
    "url": ["host_len", "path_len", "n_path_components", "n_query_params", "special_char_freq", "entropy_url",
            "entropy_host", "entropy_path", "max_digit_run", "consonant_ratio_host", "n_susp_keywords",
            "n_hyphen_host", "has_port", "has_at_symbol", "n_encoded_chars"],
    "html": ["n_password_fields", "n_text_inputs", "n_email_inputs", "n_hidden_elements", "n_buttons",
             "n_submit_controls", "n_textareas", "n_selects", "n_noscript", "n_meta_tags", "n_cred_keywords",
             "has_card_field", "n_inline_scripts", "has_signin_title"],
    "fwb": ["n_ext_scripts", "n_ext_css", "n_ext_imgs", "n_ext_iframes", "ext_resource_ratio", "n_ext_domains",
            "n_meta_refresh", "n_js_redirects", "dom_depth", "n_tags", "n_unique_tags", "text_html_ratio",
            "anchor_tag_ratio", "has_builder_generator", "has_fwb_footer_text"],
    "brand": ["host_has_brand", "path_has_brand", "n_brand_title", "n_brand_text", "n_brand_img",
              "brand_mismatch", "n_distinct_brands"],
    "behavior": ["n_ext_form_actions", "has_empty_form_action", "n_form_post", "n_js_open", "n_onsubmit",
                 "has_right_click_block", "n_hidden_iframes", "has_mailto_action",
                 "n_redirects", "host_changed_on_redirect", "final_url_len", "resp_time"],  # last 4 = LIVE only
    "visual": ["n_logo_like_imgs", "n_favicon_links", "has_external_favicon", "n_css_bg_images", "n_css_colors",
               "n_style_tags", "inline_style_ratio", "n_font_families",
               "shot_aspect", "shot_mean_r", "shot_mean_g", "shot_mean_b", "shot_color_std",
               "shot_nonwhite_frac", "shot_edge_density", "shot_n_colors"],  # shot_* = screenshot only
}
_all = [n for g in GROUP_ORDER for n in GROUPS[g]]
assert len(_all) == len(set(_all)), "duplicate feature names"
BINARY = {n for n in _all if n.startswith(("has_", "is_"))} | {"hidden_banner", "brand_mismatch", "host_has_brand",
                                                              "path_has_brand", "host_changed_on_redirect"}
TIMING_FEATS = ["resp_time"]  # latency: a collection-method artifact, not page behavior; excluded unless --keep-timing
DROPPED = set()


def is_count(n):
    return n.startswith("n_") or "len" in n or n in {"max_digit_run", "dom_depth", "shot_n_colors"}


def names_for(groups, dropped=None):
    dropped = DROPPED if dropped is None else dropped  # explicit arg: safe across joblib worker processes
    return [n for g in GROUP_ORDER if g in groups for n in GROUPS[g] if n not in dropped]


def colgroups_for(groups, dropped=None):
    dropped = DROPPED if dropped is None else dropped
    return [g for g in GROUP_ORDER if g in groups for n in GROUPS[g] if n not in dropped]


# --------------------------------------------------------------------------- helpers
def entropy(s):
    if not s:
        return 0.0
    c, n = Counter(s), len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def hostname(u):
    u = (u or "").strip()
    u = "http:" + u if u.startswith("//") else u
    try:
        return (urlparse(u).hostname or "").lower()
    except ValueError:
        return ""


def is_ext(u, host):
    u = (u or "").strip()
    return u.startswith(("http://", "https://", "//")) and hostname(u) != host


SUSP_KW = fp.SENSITIVE + ["webscr", "ebayisapi", "signon", "authenticate", "recover", "suspend", "unlock", "alert",
                          "invoice", "payment", "security", "validate", "click", "free", "gift", "prize", "claim"]
BRANDS2 = sorted(set(fp.BRANDS + ["dhl", "ups", "usps", "linkedin", "instagram", "whatsapp", "outlook", "steam",
                                  "adobe", "docusign", "wellsfargo", "bankofamerica", "citibank", "hsbc", "santander"]))
CRED_RE = re.compile(r"\b(?:password|passwd|passcode|ssn|social security|card number|cvv|cvc|expiry|otp|"
                     r"verification code|pin|iban|routing|account number)\b", re.I)
BUILDER_RE = re.compile(r"wix|weebly|squarespace|wordpress|blogger|godaddy|yola|mailchimp|google sites", re.I)
FOOTER_WORDS = ["powered by", "create your own", "made with", "built with", "free website"]


def brand_hits(text):
    low = (text or "").lower()
    toks = set(re.split(r"[^a-z0-9]+", low))
    return {b for b in BRANDS2 if b in toks or (len(b) >= 5 and b in low)}


# --------------------------------------------------------------------------- feature groups
def url_new(url):
    p = urlparse(url if "//" in url else "//" + url)
    host, path, low = (p.hostname or "").lower(), p.path or "", url.lower()
    letters = [c for c in host if c.isalpha()]
    try:
        port = int(p.port is not None)
    except ValueError:
        port = 1
    return {
        "host_len": len(host), "path_len": len(path),
        "n_path_components": len([x for x in path.split("/") if x]), "n_query_params": len(parse_qsl(p.query)),
        "special_char_freq": sum(not c.isalnum() for c in url) / max(len(url), 1),
        "entropy_url": entropy(url), "entropy_host": entropy(host), "entropy_path": entropy(path),
        "max_digit_run": max((len(r) for r in re.findall(r"\d+", url)), default=0),
        "consonant_ratio_host": sum(c not in "aeiou" for c in letters) / max(len(letters), 1),
        "n_susp_keywords": sum(w in low for w in SUSP_KW), "n_hyphen_host": host.count("-"),
        "has_port": port, "has_at_symbol": int("@" in url), "n_encoded_chars": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
    }


def html_new(soup):
    inputs = soup.find_all("input")
    types = [(i.get("type") or "text").lower() for i in inputs]
    attr_blob = " ".join(" ".join((i.get(k) or "") for k in ("name", "placeholder", "id", "autocomplete")) for i in inputs)
    vis = soup.get_text(" ", strip=True)
    hidden_styled = sum(1 for el in soup.find_all(style=True)
                        if re.search(r"visibility\s*:\s*hidden|display\s*:\s*none", el["style"], re.I))
    title = (soup.title.string or "") if soup.title and soup.title.string else ""
    return {
        "n_password_fields": types.count("password"),
        "n_text_inputs": sum(t in ("text", "tel", "number") for t in types), "n_email_inputs": types.count("email"),
        "n_hidden_elements": types.count("hidden") + hidden_styled + len(soup.find_all(attrs={"hidden": True})),
        "n_buttons": len(soup.find_all("button")),
        "n_submit_controls": types.count("submit") + len(soup.find_all("button", attrs={"type": "submit"})),
        "n_textareas": len(soup.find_all("textarea")), "n_selects": len(soup.find_all("select")),
        "n_noscript": len(soup.find_all("noscript")), "n_meta_tags": len(soup.find_all("meta")),
        "n_cred_keywords": len(CRED_RE.findall(attr_blob)) + len(CRED_RE.findall(vis[:50000])),
        "has_card_field": int(bool(re.search(r"card|cvv|cvc|cc-num", attr_blob, re.I))),
        "n_inline_scripts": sum(1 for s in soup.find_all("script") if not s.get("src")),
        "has_signin_title": int(bool(re.search(r"sign.?in|log.?in|verify|account|secure", title, re.I))),
    }


def fwb_new(soup, html, url):
    host = hostname(url)
    scripts = [s.get("src") for s in soup.find_all("script", src=True)]
    css = [l.get("href") for l in soup.find_all("link") if "stylesheet" in " ".join(l.get("rel") or []).lower()]
    imgs = [i.get("src") for i in soup.find_all("img")]
    ifr = [i.get("src") for i in soup.find_all("iframe")]
    allres = scripts + css + imgs + ifr
    ext = [r for r in allres if is_ext(r, host)]
    vis, tags = soup.get_text(" ", strip=True), soup.find_all(True)
    gen = [m.get("content") or "" for m in soup.find_all("meta") if (m.get("name") or "").lower() == "generator"]
    return {
        "n_ext_scripts": sum(is_ext(r, host) for r in scripts), "n_ext_css": sum(is_ext(r, host) for r in css),
        "n_ext_imgs": sum(is_ext(r, host) for r in imgs), "n_ext_iframes": sum(is_ext(r, host) for r in ifr),
        "ext_resource_ratio": len(ext) / max(len(allres), 1), "n_ext_domains": len({hostname(r) for r in ext}),
        "n_meta_refresh": sum((m.get("http-equiv") or "").lower() == "refresh" for m in soup.find_all("meta")),
        "n_js_redirects": len(re.findall(r"location\s*(?:\.href\s*)?=(?!=)|location\.(?:replace|assign)\s*\(", html)),
        "dom_depth": max((len(list(t.parents)) for t in tags), default=0),
        "n_tags": len(tags), "n_unique_tags": len({t.name for t in tags}),
        "text_html_ratio": len(vis) / max(len(html), 1), "anchor_tag_ratio": len(soup.find_all("a")) / max(len(tags), 1),
        "has_builder_generator": int(any(BUILDER_RE.search(g) for g in gen)),
        "has_fwb_footer_text": int(any(w in vis.lower() for w in FOOTER_WORDS)),
    }


def brand_new(soup, url):
    p = urlparse(url if "//" in url else "//" + url)
    host_h, path_h = brand_hits(p.hostname or ""), brand_hits((p.path or "") + " " + (p.query or ""))
    title = (soup.title.string or "") if soup.title and soup.title.string else ""
    title_h = brand_hits(title)
    img_h = brand_hits(" ".join(" ".join((i.get(k) or "") for k in ("alt", "src", "title")) for i in soup.find_all("img")))
    text_h = brand_hits(soup.get_text(" ", strip=True)[:50000])
    mentioned = title_h | img_h | path_h
    return {
        "host_has_brand": int(bool(host_h)), "path_has_brand": int(bool(path_h)),
        "n_brand_title": len(title_h), "n_brand_text": len(text_h), "n_brand_img": len(img_h),
        "brand_mismatch": int(len(mentioned - host_h) > 0),  # brand shown on page/path but absent from host (crude)
        "n_distinct_brands": len(mentioned | host_h),
    }


def behavior_new(soup, html, url, page):
    host = hostname(url)
    forms = soup.find_all("form") if soup is not None else []
    actions = [(f.get("action") or "").strip() for f in forms]
    d = {}
    if soup is not None:
        d.update({
            "n_ext_form_actions": sum(is_ext(a, host) for a in actions),
            "has_empty_form_action": int(any(a in ("", "#", "about:blank") for a in actions)),
            "n_form_post": sum((f.get("method") or "get").lower() == "post" for f in forms),
            "n_js_open": len(re.findall(r"window\.open\s*\(", html)),
            "n_onsubmit": sum(1 for f in forms if f.get("onsubmit")),
            "has_right_click_block": int(bool(re.search(r"oncontextmenu\s*=\s*[\"']?\s*return\s+false|"
                                                        r"contextmenu[^;]{0,60}preventDefault", html, re.I))),
            "n_hidden_iframes": sum(1 for i in soup.find_all("iframe") if
                                    re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", i.get("style") or "", re.I)
                                    or i.get("width") in ("0", "1") or i.get("height") in ("0", "1")
                                    or i.get("hidden") is not None),
            "has_mailto_action": int(any(a.lower().startswith("mailto:") for a in actions)),
        })
    if page and page.get("ok"):  # LIVE features (never populated here: pages are never fetched live)
        d.update({"n_redirects": page["n_redirects"], "host_changed_on_redirect": page["host_changed"],
                  "final_url_len": page["final_len"], "resp_time": page["resp_time"]})
    return d


def visual_html(soup, html, url):
    host = hostname(url)
    imgs = soup.find_all("img")
    favs = [l for l in soup.find_all("link") if "icon" in " ".join(l.get("rel") or []).lower()]
    css_text = " ".join(s.get_text() for s in soup.find_all("style")) + " " + " ".join(el["style"] for el in soup.find_all(style=True))
    ntags = max(len(soup.find_all(True)), 1)
    return {
        "n_logo_like_imgs": sum(1 for i in imgs if re.search(
            r"logo|brand|icon", " ".join([i.get("alt") or "", i.get("src") or "", " ".join(i.get("class") or []),
                                          i.get("id") or ""]), re.I)),
        "n_favicon_links": len(favs), "has_external_favicon": int(any(is_ext(l.get("href"), host) for l in favs)),
        "n_css_bg_images": len(re.findall(r"background(?:-image)?\s*:[^;}]*url\(", css_text, re.I)),
        "n_css_colors": len(set(re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", css_text))),
        "n_style_tags": len(soup.find_all("style")), "inline_style_ratio": len(soup.find_all(style=True)) / ntags,
        "n_font_families": len(set(re.findall(r"font-family\s*:\s*([^;}]+)", css_text, re.I))),
    }


def shot_path_for(url, shot_dir):
    if not shot_dir:
        return None
    h = hashlib.sha1((url if "//" in url else "http://" + url).encode()).hexdigest()  # same naming as HTML snapshots
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join(shot_dir, h + ext)
        if os.path.exists(p):
            return p
    return None


def shot_features(path):
    """Simple image statistics from a screenshot, not logo/layout similarity."""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        w, h = im.size
        a = np.asarray(im.resize((128, 128)), dtype=np.float32)
    except Exception:
        return {}
    gray = a.mean(axis=2)
    gx, gy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    return {
        "shot_aspect": w / max(h, 1), "shot_mean_r": float(a[..., 0].mean()), "shot_mean_g": float(a[..., 1].mean()),
        "shot_mean_b": float(a[..., 2].mean()), "shot_color_std": float(a.std()),
        "shot_nonwhite_frac": float((a.min(axis=2) < 240).mean()),
        "shot_edge_density": float(((gx[:-1, :] > 30) | (gy[:, :-1] > 30)).mean()),
        "shot_n_colors": len(np.unique((a // 64).astype(int).reshape(-1, 3), axis=0)),
    }


def feature_row(url, html, groups, page=None, shot_dir=None, max_chars=500000, dropped=None):
    """One URL -> feature vector in names_for(groups, dropped) order. `dropped` is threaded explicitly (not
    read from the module-level DROPPED) since this runs inside joblib worker PROCESSES."""
    html = (html or "")[:max_chars]
    soup = BeautifulSoup(html, "html.parser") if html else None
    d = {}
    d.update(fp.url_features(url)); d.update(fp.html_features(html, url, soup))
    if "url" in groups:
        d.update(url_new(url))
    if soup is not None:
        if "html" in groups:
            d.update(html_new(soup))
        if "fwb" in groups:
            d.update(fwb_new(soup, html, url))
        if "brand" in groups:
            d.update(brand_new(soup, url))
        if "visual" in groups:
            d.update(visual_html(soup, html, url))
    if "behavior" in groups:
        d.update(behavior_new(soup, html, url, page))
    if "visual" in groups:
        sp = shot_path_for(url, shot_dir)
        if sp:
            d.update(shot_features(sp))
    return [float(d.get(n, np.nan)) for n in names_for(groups, dropped)]


# --------------------------------------------------------------------------- preprocessing
class Preprocessor:
    """Median-impute missing values, log1p count features, standardize non-binary features. Fit on TRAIN only."""

    def __init__(self, names):
        self.names = list(names)
        self.log_idx = [i for i, n in enumerate(self.names) if is_count(n)]
        self.std_idx = [i for i, n in enumerate(self.names) if n not in BINARY]

    def _clean(self, X):
        X = np.array(X, dtype=float); X[~np.isfinite(X)] = np.nan
        return X

    def _log(self, X):
        X[:, self.log_idx] = np.log1p(np.clip(X[:, self.log_idx], 0, None))
        return X

    def fit(self, X):
        X = self._clean(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isfinite(med), med, 0.0)  # all-missing column -> 0
        Xi = self._log(np.where(np.isnan(X), self.median_, X))
        sc = StandardScaler().fit(Xi[:, self.std_idx])
        self.mean_, self.scale_ = sc.mean_, sc.scale_  # scale_ is 1 for constant columns
        return self

    def transform(self, X):
        X = self._clean(X)
        Xi = self._log(np.where(np.isnan(X), self.median_, X))
        Xi[:, self.std_idx] = (Xi[:, self.std_idx] - self.mean_) / self.scale_
        return Xi

    def state(self):
        return {"names": self.names, "median": self.median_, "mean": self.mean_, "scale": self.scale_}

    @classmethod
    def from_state(cls, st):
        p = cls(st["names"]); p.median_, p.mean_, p.scale_ = st["median"], st["mean"], st["scale"]
        return p


# --------------------------------------------------------------------------- training
def parse_groups(a):
    if a.groups:
        g = [x.strip() for x in a.groups.split(",") if x.strip()]
        bad = [x for x in g if x not in GROUP_ORDER]
        if bad:
            raise SystemExit(f"unknown groups {bad}; choose from {GROUP_ORDER}")
        return g if "freephish" in g else ["freephish"] + g
    return ["freephish"] if a.features == "basic" else list(GROUP_ORDER)


def _row(u, h, page, groups, shot_dir, max_chars, dropped=None):
    t0 = time.perf_counter()
    return feature_row(u, h, groups, page, shot_dir, max_chars, dropped), time.perf_counter() - t0


def build(urls, htmls, pages, groups, a):
    dropped = frozenset(DROPPED)  # snapshot NOW, in the parent, and pass it explicitly to every worker task
    key = hashlib.sha1(("\n".join(urls) + "|" + ",".join(groups) + "|" + str(a.screenshot_dir) + "|"
                        + ",".join(sorted(dropped))).encode()).hexdigest()
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        if str(z["key"]) == key:
            print("loaded cached features:", a.cache)
            return z["X"], z["ft"]
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs)(delayed(_row)(u, h, pg, groups, a.screenshot_dir, a.max_html_chars, dropped)
                                    for u, h, pg in zip(urls, htmls, pages))
    X, ft = np.array([r[0] for r in res], float), np.array([r[1] for r in res], float)
    print(f"feature extraction {time.time() - t0:.1f}s wall-clock")
    if a.cache:
        np.savez(a.cache, X=X, ft=ft, key=key)
    return X, ft


def load_and_featurize(a):
    """Shared by train/benchmark: load data, build features. Returns groups, names, colgroups, urls, y, X, ft, grp."""
    groups = parse_groups(a)
    DROPPED.clear()
    if not a.keep_timing:
        DROPPED.update(TIMING_FEATS)
    if a.drop_features:
        DROPPED.update(x.strip() for x in a.drop_features.split(",") if x.strip())
    bad = [x for x in DROPPED if x not in _all]
    if bad:
        raise SystemExit(f"--drop-features: unknown feature names {bad}")
    if DROPPED:
        print("excluded features:", sorted(DROPPED))
    df = fp.load_table(a.csv)
    print("columns:", df.columns.tolist(), "| rows:", len(df))
    for c in (a.url_col, a.label_col):
        if c not in df.columns:
            raise SystemExit(f"column '{c}' not in {df.columns.tolist()}")
    names, colgroups = names_for(groups), colgroups_for(groups)
    print("feature groups:", groups)
    html_col = fp.find_html_col(df.columns, a.html_col)
    urls, htmls, y = fp.prepare(df, a, html_col)
    pages = [None] * len(urls)  # no live fetch in this pipeline -> behavior's LIVE_ONLY features stay NaN
    if not html_col:
        print("WARNING: no HTML available -> most groups are missing/zero.")
    if a.screenshot_dir:
        found = sum(shot_path_for(u, a.screenshot_dir) is not None for u in urls)
        print(f"screenshots found: {found}/{len(urls)}")
    X, ft = build(urls, htmls, pages, groups, a)
    grp = np.array([fp.group_key(u) for u in urls])
    return groups, names, colgroups, urls, y, X, ft, grp


def cmd_train(a):
    groups, names, colgroups, urls, y, X, ft, grp = load_and_featurize(a)
    tr, te = fp.split_idx(y, grp, a.split)
    overlap = np.mean([g in set(grp[tr]) for g in grp[te]])
    print(f"\n== {a.split} split | train {len(tr)} / test {len(te)} | test rows sharing a domain with train: {overlap:.1%} ==")
    prep = Preprocessor(names).fit(X[tr])
    Xp = prep.transform(X)
    t0 = time.time(); model = fp.StackModel().fit(Xp[tr], y[tr])
    print(f"train time {time.time() - t0:.1f}s")
    enh_pred = model.predict(Xp[te])
    enh_timing = fp.time_inference(model, Xp[te], ft[te], a.timing_samples)
    fp.report(f"StackModel n_test={len(te)}", y[te], enh_pred, enh_timing)
    print("(timing = feature extraction + single-sample inference; preprocessing and page fetch not included)")
    if a.results_out:
        fp.save_results(a.results_out, "enhanced", np.asarray(urls)[te], y[te], enh_pred, model.predict_proba(Xp[te]), enh_timing,
                        {"feature_set": "enhanced", "n_features": len(names), "groups": groups,
                         "excluded_features": sorted(DROPPED), "split": a.split, "seed": 42, "n_rows_used": len(urls),
                         "n_train": len(tr), "n_test": len(te),
                         "test_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y[te], return_counts=True))},
                         "domain_overlap": float(overlap), "csv": a.csv, "balance": a.balance, "dedup": a.dedup,
                         "fwb_only": a.fwb_only, "max_rows": a.max_rows})
    joblib.dump({"model": model, "prep": prep.state(), "names": names, "groups": groups,
                 "max_html_chars": a.max_html_chars, "version": 2, "dropped": sorted(DROPPED)}, a.model)
    print("saved", a.model)


def cmd_benchmark(a):
    """Fair comparison of feature sets: SAME data, SAME splits, repeated with different seeds.
    Compares Base (22 minus the 2 FWB features), Original 22 and Enhanced."""
    import pandas as pd
    groups, names, colgroups, urls, y, X, ft, grp = load_and_featurize(a)
    fi = [i for i, c in enumerate(colgroups) if c == "freephish"]
    sets = {"Base (22 - 2 FWB)": [i for i in fi if names[i] not in fp.FWB_FEATS], "Original 22": fi}
    if len(names) > len(fi):
        sets["Enhanced"] = list(range(len(names)))
    else:
        print("NOTE: only the original feature group is active -> nothing enhanced to compare.")
    enh = [k for k in sets if k.startswith("Enhanced")]
    mk = (lambda sd: LGBMClassifier(n_estimators=200, random_state=sd, verbose=-1)) if a.fast else (lambda sd: fp.StackModel(seed=sd))
    modes = ["group", "random"] if a.split == "both" else [a.split]
    rows = []
    for mode in modes:
        for rep in range(a.repeats):
            seed = a.seed + rep
            tr, te = fp.split_idx(y, grp, mode, seed)
            if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
                print(f"  skipping {mode} repeat {rep}: a split has only one class"); continue
            Xp = Preprocessor(names).fit(X[tr]).transform(X)
            overlap = np.mean([g in set(grp[tr]) for g in grp[te]])
            for nm, idx in sets.items():
                m = mk(seed).fit(Xp[tr][:, idx], y[tr])
                pr = m.predict(Xp[te][:, idx])
                rows.append({"split": mode, "rep": rep, "seed": seed, "features": nm, "n_test": len(te),
                             "domain_overlap": overlap, "acc": fp.accuracy_score(y[te], pr),
                             "prec": fp.precision_score(y[te], pr, zero_division=0),
                             "rec": fp.recall_score(y[te], pr, zero_division=0), "f1": f1_score(y[te], pr, zero_division=0)})
            print(f"  done {mode} repeat {rep + 1}/{a.repeats} (seed {seed}, test {len(te)}, domain overlap {overlap:.0%})")
    r = pd.DataFrame(rows)
    if r.empty:
        raise SystemExit("no valid splits (check class balance / number of domain groups)")
    r.to_csv(a.out, index=False)
    pm = lambda x: f"{x.mean():.3f}+-{x.std(ddof=1) if len(x) > 1 else 0:.3f}"
    for mode in modes:
        d = r[r.split == mode]
        print(f"\n=== {mode} split | {d.rep.nunique()} repeats | model: {'LightGBM' if a.fast else 'StackModel'} "
              f"| mean test size {d.n_test.mean():.0f} | mean domain overlap {d.domain_overlap.mean():.0%} ===")
        print(f"{'features':<22}{'accuracy':<14}{'precision':<14}{'recall':<14}{'F1':<14}")
        for nm in sets:
            g = d[d.features == nm]
            print(f"{nm:<22}{pm(g.acc):<14}{pm(g.prec):<14}{pm(g.rec):<14}{pm(g.f1):<14}")
        piv = d.pivot(index="rep", columns="features", values="f1")
        pairs = [("Original 22", "Base (22 - 2 FWB)")] + [(e, "Original 22") for e in enh]
        for hi, lo in pairs:
            if hi in piv and lo in piv:
                df_ = (piv[hi] - piv[lo]).dropna()
                print(f"  paired F1 difference [{hi} minus {lo}]: {df_.mean():+.4f} +- {df_.std(ddof=1) if len(df_) > 1 else 0:.4f} "
                      f"| {hi} better in {int((df_ > 0).sum())}/{len(df_)} repeats, tied in {int((df_ == 0).sum())}")
    if a.previous_acc is not None and "random" in modes:
        g = r[(r.split == "random") & (r.features == "Original 22")]
        print(f"\nYour earlier result: accuracy {a.previous_acc:.3f}. Same-pipeline re-run (random split, original 22 features): "
              f"{pm(g.acc)}. A gap means the earlier run used different data/settings; compare feature sets only within this table.")
    print("\nCaveats: repeats reuse the same data, so they are NOT independent and the +- understates the true uncertainty; "
          "differences within about one +- are not evidence of improvement. Group split = unseen domains (stricter).")
    print("saved", a.out)


# --------------------------------------------------------------------------- inference (used by freephish_gan.py)
def load_bundle(path):
    import sys, __main__
    sys.modules.setdefault("freephish_simplified", fp)  # models pickled before the rename used the old module name
    if not hasattr(__main__, "StackModel"):
        __main__.StackModel = fp.StackModel
    obj = joblib.load(path)
    DROPPED.clear()
    if isinstance(obj, dict):
        DROPPED.update(obj.get("dropped", []))
    if isinstance(obj, fp.StackModel):  # legacy v1 model
        return {"model": obj, "prep": None, "groups": ["freephish"], "names": names_for(["freephish"]), "max_html_chars": 500000}
    obj["prep"] = Preprocessor.from_state(obj["prep"]) if obj.get("prep") else None
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv", required=True); common.add_argument("--url-col", default="url")
    common.add_argument("--label-col", default="label"); common.add_argument("--html-col"); common.add_argument("--pos-label")
    common.add_argument("--features", choices=["basic", "enhanced"], default="enhanced")
    common.add_argument("--groups", help=f"comma list from {GROUP_ORDER} (freephish always included)")
    common.add_argument("--screenshot-dir", help="folder with <sha1(url)>.png screenshots (optional)")
    common.add_argument("--balance", action="store_true"); common.add_argument("--max-rows", type=int)
    common.add_argument("--fwb-only", action="store_true"); common.add_argument("--dedup", action="store_true")
    common.add_argument("--n-jobs", type=int, default=-1); common.add_argument("--max-html-chars", type=int, default=500000)
    common.add_argument("--cache")
    common.add_argument("--keep-timing", action="store_true", help="include resp_time (a latency artifact)")
    common.add_argument("--drop-features", help="comma list of extra feature names to exclude")
    t = sp.add_parser("train", parents=[common])
    t.add_argument("--split", choices=["group", "random"], default="group")
    t.add_argument("--timing-samples", type=int, default=0)
    t.add_argument("--model", default="model_enh.joblib")
    t.add_argument("--results-out", default="results_enhanced.json", help="STAGE 2 output; compare with `freephish_old.py compare`")
    bm = sp.add_parser("benchmark", parents=[common], help="fair repeated-split comparison of feature sets")
    bm.add_argument("--split", choices=["group", "random", "both"], default="both")
    bm.add_argument("--repeats", type=int, default=5); bm.add_argument("--seed", type=int, default=42)
    bm.add_argument("--fast", action="store_true", help="use LightGBM alone instead of the stacking model (much faster)")
    bm.add_argument("--previous-acc", type=float, help="your earlier accuracy, printed next to a same-pipeline re-run")
    bm.add_argument("--out", default="benchmark.csv")
    a = ap.parse_args()
    {"train": cmd_train, "benchmark": cmd_benchmark}[a.cmd](a)


if __name__ == "__main__":
    main()

"""freephish_classify_all.py - live classification of ONE shared list of URLs across up to four already-
trained & saved FreePhish models side by side:
  --old-model      freephish_old.py      `train`                    output (.joblib StackModel)
  --enhanced-model  freephish_enhanced.py `train`                    output (.joblib bundle)
  --adv-model       freephish_gan.py      `tune`/`retrain --save-model` output (.joblib bundle)
  --de-model        freephish_de.py       `train`                    output (.joblib bundle)

Each URL's HTML is fetched live, ONCE, and a single raw feature matrix is extracted covering the UNION
of feature groups every supplied model actually needs. Each model then selects its own columns out of
that shared matrix by name and classifies with its own training-time convention - exactly the same
predict_original/predict_enhanced/predict_adv/predict_de helpers that freephish_gan.py already uses for
the original/enhanced/adversarial/DE models, so the prediction logic itself is not duplicated here.

This script only LOADS already-saved models and runs inference - it never trains or retrains anything.
At least one of the four --*-model flags is required; give as many as you have for a side-by-side table.

Example:
  python freephish_classify_all.py --urls urls.txt \\
      --old-model model_baseline.joblib --enhanced-model model_enhanced.joblib \\
      --adv-model model_adv.joblib --de-model model_de.joblib
"""
import argparse, json, time

import joblib
import numpy as np
from joblib import Parallel, delayed

import freephish_old as fp
import freephish_enhanced as E
from freephish_gan import (load_enh_bundle, load_adv_bundle, load_de_bundle,
                            predict_original, predict_enhanced, predict_adv, predict_de)


def read_urls(path):
    urls = []
    for line in open(path):
        u = line.strip()
        if u and not u.startswith("#"):
            urls.append(u)
    if not urls:
        raise SystemExit(f"{path}: no URLs found (one per line; blank lines and '#' comments are ignored)")
    return urls


def fetch_all(urls, snapshot_dir, n_jobs, timeout, quiet=False):
    """Live-fetch every URL's HTML exactly once, in parallel; shared by every model below."""
    t0 = time.time()
    htmls = Parallel(n_jobs=n_jobs)(delayed(fp.fetch_html)(u, snapshot_dir, timeout) for u in urls)
    if not quiet:
        n_ok = sum(1 for h in htmls if h)
        print(f"fetched HTML for {len(urls)} URL(s) in {time.time() - t0:.1f}s "
              f"({n_ok} succeeded, {len(urls) - n_ok} failed/empty - fp.fetch_html returns '' on failure)", flush=True)
    return htmls


def load_old_model(path):
    import __main__
    if not hasattr(__main__, "StackModel"):  # models pickled while running freephish_old.py directly
        __main__.StackModel = fp.StackModel
    model = joblib.load(path)
    if not isinstance(model, fp.StackModel):
        raise SystemExit(f"--old-model {path} is not a freephish_old.py StackModel (.joblib from `train`).")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", required=True, help="text file, one URL per line ('#' comments/blank lines ignored)")
    ap.add_argument("--old-model", help="freephish_old.py `train` output (.joblib StackModel)")
    ap.add_argument("--enhanced-model", help="freephish_enhanced.py `train` output (.joblib bundle)")
    ap.add_argument("--adv-model", help="freephish_gan.py `tune`/`retrain --save-model` output (.joblib bundle)")
    ap.add_argument("--de-model", help="freephish_de.py `train` output (.joblib bundle)")
    ap.add_argument("--snapshots", help="optional dir to save each fetched page's raw HTML")
    ap.add_argument("--max-html-chars", type=int, default=500000)
    ap.add_argument("--timeout", type=int, default=10, help="per-URL HTML fetch timeout (s)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default="classify_all_results.jsonl")
    ap.add_argument("--actual-labels", help="optional JSON file of {url: 'phishing'/'benign'} ground truth "
                                             "(e.g. url_actual_labels.json); if given, prints a correct/total "
                                             "accuracy summary per model at the end, against this ground truth")
    ap.add_argument("-q", "--quiet", action="store_true",
                     help="suppress progress/diagnostic lines (feature groups, fetch summary, visual-group "
                          "note); the final table and --actual-labels summary still print")
    a = ap.parse_args()

    if not any([a.old_model, a.enhanced_model, a.adv_model, a.de_model]):
        raise SystemExit("give at least one of --old-model / --enhanced-model / --adv-model / --de-model")

    urls = read_urls(a.urls)
    if not a.quiet:
        print(f"{len(urls)} URL(s) to classify", flush=True)

    bundles = {}
    if a.enhanced_model:
        bundles["Enhanced"] = load_enh_bundle(a.enhanced_model)
    if a.adv_model:
        bundles["Adv-trained enhanced"] = load_adv_bundle(a.adv_model)
    if a.de_model:
        bundles["DE-weighted enhanced"] = load_de_bundle(a.de_model)
    old_model = load_old_model(a.old_model) if a.old_model else None

    # ---- one shared raw feature matrix: union of every group any requested model needs, nothing dropped
    #      at extraction time (dropping/selection happens per-model, by name, right before each predict_*) ----
    union_groups = {"freephish"} if old_model is not None else set()
    for b in bundles.values():
        union_groups.update(g for g in b.get("groups") or E.GROUP_ORDER if g in E.GROUP_ORDER)
    union_groups = [g for g in E.GROUP_ORDER if g in union_groups] or ["freephish"]
    if not a.quiet:
        if "visual" in union_groups:
            print("note: a model uses the 'visual' (screenshot) feature group; this live script takes no "
                  "screenshots, so those columns will come back NaN/imputed-away for every model.", flush=True)
        print("feature groups extracted (union across all requested models):", union_groups, flush=True)

    htmls = fetch_all(urls, a.snapshots, a.n_jobs, a.timeout, quiet=a.quiet)

    E.DROPPED.clear()  # full, undropped layout for these groups - each model below selects its own columns by name
    ns = argparse.Namespace(n_jobs=a.n_jobs, max_html_chars=a.max_html_chars, cache=None, screenshot_dir=None)
    X_full, _ = E.build(urls, htmls, [None] * len(urls), union_groups, ns)
    full_names = [n for g in E.GROUP_ORDER if g in union_groups for n in E.GROUPS[g]]

    # ---- classify with each model; every predict_* returns a flat 1-D P(phish) array (fp.StackModel's own
    #      convention, not sklearn's 2-column one - see predict_original/predict_enhanced/predict_adv/predict_de
    #      in freephish_gan.py, none of which index [:, 1]) ----
    model_order, preds = [], {}
    if old_model is not None:
        model_order.append("Original (22-feat)")
        _, p = predict_original(old_model, X_full, full_names)
        preds["Original (22-feat)"] = np.asarray(p, float)
    predict_fn = {"Enhanced": predict_enhanced, "Adv-trained enhanced": predict_adv, "DE-weighted enhanced": predict_de}
    for label, b in bundles.items():
        model_order.append(label)
        _, p = predict_fn[label](b, X_full, full_names)
        preds[label] = np.asarray(p, float)

    # ---- print + save a per-URL, per-model side-by-side table ----
    col_w = max(10, max(len(m) for m in model_order) + 1)
    header = f"{'URL':<60}" + "".join(f"{m:<{col_w}}" for m in model_order)
    print("\n" + header)
    print("-" * len(header))
    with open(a.out, "w") as fh:
        for i, u in enumerate(urls):
            row = {"url": u, "html_fetched": bool(htmls[i])}
            cells = []
            for m in model_order:
                p = float(preds[m][i])
                row[m] = {"p_phish": p, "phishing": p >= 0.5}
                cells.append(f"{'PHISH' if p >= 0.5 else 'ok'} {p:.3f}".ljust(col_w))
            print(f"{u[:58]:<60}" + "".join(cells))
            fh.write(json.dumps(row) + "\n")
    print(f"\nsaved {a.out}")

    # ---- optional: correct/total per model, against ground truth (which URL list each URL came from -
    #      this script has no notion of ground truth on its own, so it only prints this if you pass it one) ----
    if a.actual_labels:
        with open(a.actual_labels) as fh:
            truth = json.load(fh)
        known = [u for u in urls if u in truth]
        if not known:
            print(f"\nnote: none of these URLs were found in {a.actual_labels}; skipping the accuracy summary.")
        else:
            print(f"\nACCURACY vs ground truth ({a.actual_labels}; {len(known)}/{len(urls)} URLs had a known label)")
            for m in model_order:
                correct = sum((preds[m][urls.index(u)] >= 0.5) == (truth[u] == "phishing") for u in known)
                print(f"  {m:<24}{correct}/{len(known)} correct ({correct / len(known):.1%})")


if __name__ == "__main__":
    main()

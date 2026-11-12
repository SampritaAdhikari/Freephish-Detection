#!/usr/bin/env python3
"""FreePhish: GAN attack on the enhanced (100-feature) model - Algorithm 4 style. NO surrogate model and NO
classifier anywhere in the GAN: D only separates real vectors from G's output, G only tries to fool D
(L_D = -log D(x_real) - log(1 - D(G(z))), L_G = -log D(G(z)) + a small L2 penalty on the perturbation).
Requires freephish_old.py and freephish_enhanced.py in the same folder, and PyTorch.

* Works in FEATURE SPACE: perturbs feature vectors, does not produce real modified web pages.
* Only --perturb-groups features may change (default: 22 original + url/html groups), each by at most
  --epsilon std-devs; protected features (has_login_form, n_password_fields) never change. Results are
  clipped to the training range and rounded for integer/binary features.
* --gan-mode evade (default): G turns phishing -> benign-looking (D real = benign). --gan-mode slide:
  literal slide reading, G turns benign -> phishing-like (D real = phishing). Trained on a pool whose
  domains never appear in the test pages.
* Plain and adversarially-trained models are tested on exactly the same adversarial pages (GAN never
  sees a classifier) - how much that fools a model is empirical, not guaranteed.

  python freephish_gan.py adv --csv sample.parquet --enhanced-model model_enh.joblib --adv-train
"""
import argparse, json, os, time, warnings
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import freephish_old as fp
import freephish_enhanced as E

PROTECT_DEFAULT = ["has_login_form", "n_password_fields"]  # features the attacker may never change


def load_enh_bundle(path):
    b = E.load_bundle(path)
    if "prep" not in b or b.get("prep") is None:
        raise SystemExit(f"{path} is not an enhanced-model bundle (no preprocessor). Use freephish_enhanced.py train to make one.")
    return b


def align_columns(X, names, all_names):
    """Match X to the model's own feature list `names`. X is either already `names`-shaped, or one column
    per name in `all_names` (the full group layout) - anything else is an error, never a silent guess."""
    X = np.asarray(X, float)
    if X.shape[1] == len(names):
        return X
    if X.shape[1] == len(all_names) and all(n in all_names for n in names):
        print(f"note: extractor returned {X.shape[1]} columns (model uses {len(names)}); "
              f"dropping {sorted(set(all_names) - set(names))} to match the model.")
        return X[:, [all_names.index(n) for n in names]]
    raise SystemExit(f"Feature mismatch: extractor gave {X.shape[1]} columns, the model expects {len(names)} and the "
                     f"full group layout has {len(all_names)}. Check that freephish_enhanced.py is the same version "
                     f"that trained the model.")


def raw_features_for_bundle(urls, htmls, bundle, n_jobs=-1, cache=None):
    """Build the RAW (pre-preprocessing) feature matrix in the bundle's own feature space (optionally cached)."""
    import hashlib
    groups = [g for g in bundle["groups"] if g != "interact"]
    E.DROPPED.clear(); E.DROPPED.update(bundle.get("dropped", []))
    names = list(bundle["names"])
    all_names = [n for g in E.GROUP_ORDER if g in groups for n in E.GROUPS[g]]
    key = hashlib.sha1(("\n".join(urls) + "|" + ",".join(names)).encode()).hexdigest()
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if str(z["key"]) == key:
            print(f"loaded cached features: {cache}")
            return z["X"], names
    ns = argparse.Namespace(n_jobs=n_jobs, max_html_chars=bundle["max_html_chars"], cache=None, screenshot_dir=None)
    X = align_columns(E.build(urls, htmls, [None] * len(urls), groups, ns)[0], names, all_names)
    if cache:
        np.savez(cache, X=X, key=key)
    return X, names


def predict_original(model, X_full, full_names):
    """freephish_old's StackModel: raw features, same 22 names as E.GROUPS['freephish']."""
    Xr = np.nan_to_num(X_full[:, [full_names.index(n) for n in fp.ALL_FEATS]])
    return model.predict(Xr), model.predict_proba(Xr)


def predict_enhanced(bundle, X_full, full_names):
    """freephish_enhanced bundle: raw features in bundle['names'] order -> its own preprocessor -> model."""
    Xp = bundle["prep"].transform(X_full[:, [full_names.index(n) for n in bundle["names"]]])
    return bundle["model"].predict(Xp), bundle["model"].predict_proba(Xp)


def predict_adv(bundle, X_full, full_names):
    """Adversarially-trained bundle saved by `tune --save-model` (a RawModel, not an enhanced Preprocessor
    pipeline): raw features in bundle['names'] order, median-imputed with the model's own training median,
    then straight into the StackModel - same convention as predict_original/predict_enhanced above."""
    Xr = X_full[:, [full_names.index(n) for n in bundle["names"]]]
    med = np.asarray(bundle["med"], float)
    Xr = np.where(np.isfinite(Xr), Xr, med)
    return bundle["model"].predict(Xr), bundle["model"].predict_proba(Xr)


def load_adv_bundle(path):
    """Load a bundle saved by `tune --save-model`. Uses E.load_bundle (not raw joblib.load) for the same
    legacy-module-aliasing handling as the enhanced/DE loaders, even though this bundle has no preprocessor
    of its own (E.load_bundle leaves bundle['prep'] as None when the dict has no 'prep' key)."""
    b = E.load_bundle(path)
    if "med" not in b or "model" not in b:
        raise SystemExit(f"{path} is not an adversarially-trained bundle (no median vector/model). "
                          f"Use `freephish_gan.py tune --save-model ...` to make one.")
    return b


def predict_de(bundle, X_full, full_names):
    """freephish_de bundle: raw features in bundle['names'] order -> its own preprocessor -> per-feature
    DE-searched weights -> model. Same preprocessor machinery as predict_enhanced, plus the weight multiply."""
    Xp = bundle["prep"].transform(X_full[:, [full_names.index(n) for n in bundle["names"]]])
    Xp = Xp * np.asarray(bundle["weights"], float)
    return bundle["model"].predict(Xp), bundle["model"].predict_proba(Xp)


def load_de_bundle(path):
    """Load a bundle saved by `freephish_de.py train`, via E.load_bundle (reconstructs a live Preprocessor)."""
    b = E.load_bundle(path)
    if "weights" not in b or b.get("prep") is None:
        raise SystemExit(f"{path} is not a DE-weighted bundle (no weights/preprocessor). "
                          f"Use `freephish_de.py train` to make one.")
    return b


def build_torch():
    try:
        import torch
    except ImportError as e:
        raise SystemExit(f"PyTorch is required. Install it with: pip install torch\n({type(e).__name__}: {e})")
    return torch


def mlp(torch, nn, sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.LeakyReLU(0.2)]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if out_act:
        layers.append(out_act)
    return nn.Sequential(*layers)


class LocalScaler:
    """A scaler the GAN owns itself (independent of any model's Preprocessor), fit on ALL rows so both
    classes are in a comparable range for the discriminator."""

    def fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # all-NaN columns (e.g. unused screenshot features) are expected
            col_med = np.nanmedian(X, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        Xc = np.where(np.isnan(X), col_med, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_, std_ = np.nanmean(Xc, axis=0), np.nanstd(Xc, axis=0)
        self.mean_ = np.where(np.isfinite(mean_), mean_, 0.0)
        self.std_ = np.where(np.isfinite(std_) & (std_ > 1e-6), std_, 1.0)
        return self

    def to_std(self, X):
        return (np.nan_to_num(X, nan=0.0) - self.mean_) / self.std_

    def to_raw(self, Xs):
        return Xs * self.std_ + self.mean_


def read_url_set(path):
    """URLs of a results *.predictions.csv (or any csv with a 'url' column)."""
    d = pd.read_csv(path)
    col = next((c for c in d.columns if c.lower() == "url"), None)
    if col is None:
        raise SystemExit(f"{path} has no 'url' column; columns are {d.columns.tolist()}")
    return set(d[col].astype(str))


class RawModel:
    """Median-imputes raw features, then fits freephish_old.StackModel (tree models are scale-invariant)."""

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # an all-NaN column (e.g. unused screenshot features) is expected
            med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.med = np.where(np.isfinite(med), med, 0.0)
        try:
            self.sm = fp.StackModel(seed=self.seed)
        except TypeError:
            self.sm = fp.StackModel()
        self.sm.fit(self._imp(X), np.asarray(y))
        return self

    def _imp(self, X):
        X = np.asarray(X, float)
        return np.where(np.isfinite(X), X, self.med)

    def predict_fn(self, X):
        Xi = self._imp(X)
        return self.sm.predict(Xi), self.sm.predict_proba(Xi)


def four(y, p):
    return {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0)}


def gkey(u):
    f = getattr(fp, "group_key", None)
    return f(u) if f else urlparse(u).netloc


def load_pages(a, rows_filter, per_class=None, seed=42):
    df = fp.load_table(a.csv)
    hc = fp.find_html_col(df.columns, a.html_col)
    df = df[df[a.url_col].astype(str).map(rows_filter)].drop_duplicates(a.url_col).reset_index(drop=True)
    y = np.asarray(fp.to_binary(df[a.label_col], a.pos_label))
    if per_class:
        rs = np.random.RandomState(seed)
        keep = np.sort(np.concatenate([rs.choice(np.where(y == c)[0], min(per_class, int((y == c).sum())), replace=False)
                                       for c in (0, 1)]))
        df, y = df.iloc[keep].reset_index(drop=True), y[keep]
    urls, htmls = df[a.url_col].astype(str).tolist(), df[hc].fillna("").astype(str).tolist()
    del df
    return urls, htmls, y


def make_plain_gan(a, X_raw, y, names, seed):
    """Algorithm-4 style GAN, no surrogate/classifier anywhere: D separates real vectors from G's output,
    G only fools D. mode 'evade': phishing -> benign-looking. mode 'slide': benign -> phishing-like.
    Only --perturb-groups features change (protected ones never do), clipped to the training range and
    rounded for integer/binary features."""
    torch = build_torch()
    import torch.nn as nn
    from types import SimpleNamespace
    torch.manual_seed(seed)
    grp_of = {n: gname for gname in E.GROUP_ORDER for n in E.GROUPS[gname]}
    protect = set(a.protect.split(",")) if a.protect else set(PROTECT_DEFAULT)
    allowed = [i for i, n in enumerate(names) if grp_of.get(n) in set(a.perturb_groups) and n not in protect]
    if not allowed:
        raise SystemExit(f"no perturbable features: --perturb-groups {a.perturb_groups} matched none of {sorted(set(grp_of.values()))}")
    fixed = [i for i in range(len(names)) if i not in set(allowed)]
    sc = LocalScaler().fit(X_raw)
    Xs = torch.tensor(sc.to_std(X_raw), dtype=torch.float32)
    src_cls, real_cls = (1, 0) if a.gan_mode == "evade" else (0, 1)
    src, real = Xs[torch.tensor(y == src_cls)], Xs[torch.tensor(y == real_cls)]
    F, Z = len(names), a.zdim
    allowed_t = torch.tensor(allowed)
    G = mlp(torch, nn, [F + Z, 128, 64, len(allowed)], out_act=nn.Tanh())
    D = mlp(torch, nn, [F, 64, 32, 1], out_act=nn.Sigmoid())
    og, od = torch.optim.Adam(G.parameters(), lr=a.lr), torch.optim.Adam(D.parameters(), lr=a.lr)
    bce = nn.functional.binary_cross_entropy

    def perturb(Xb, z):
        d = G(torch.cat([Xb, z], dim=1))
        Xa = Xb.clone()
        Xa[:, allowed_t] = Xb[:, allowed_t] + a.epsilon * d
        return Xa, d

    for _ in range(a.epochs):
        perm = torch.randperm(len(src))
        for i in range(0, len(perm), a.batch_size):
            xs = src[perm[i:i + a.batch_size]]
            n = len(xs)
            xr = real[torch.randint(len(real), (n,))]
            xa, d = perturb(xs, torch.randn(n, Z))
            od.zero_grad()
            (bce(D(xr).squeeze(1), torch.ones(n)) + bce(D(xa.detach()).squeeze(1), torch.zeros(n))).backward()
            od.step()
            og.zero_grad()
            (bce(D(xa).squeeze(1), torch.ones(n)) + a.lambda_l2 * (d ** 2).mean()).backward()
            og.step()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, hi = np.nanmin(X_raw, axis=0), np.nanmax(X_raw, axis=0)
    lo, hi = np.where(np.isfinite(lo), lo, -np.inf), np.where(np.isfinite(hi), hi, np.inf)
    is_int = np.array([bool(np.isfinite(c).any()) and bool(np.all(c[np.isfinite(c)] == np.round(c[np.isfinite(c)])))
                       for c in X_raw.T])

    def transform(X_src_raw, seed2):
        gen = torch.Generator().manual_seed(seed2)
        Xsrc = torch.tensor(sc.to_std(X_src_raw), dtype=torch.float32)
        with torch.no_grad():
            Xa, _ = perturb(Xsrc, torch.randn(len(Xsrc), Z, generator=gen))
        Xr = np.clip(sc.to_raw(Xa.numpy()), lo, hi)
        Xr[:, is_int] = np.round(Xr[:, is_int])
        Xr[:, fixed] = X_src_raw[:, fixed]  # features the attacker does not control stay exactly as they were
        Xr[np.isnan(X_src_raw)] = np.nan
        return Xr

    return SimpleNamespace(transform=transform, n_perturbable=len(allowed), n_fixed=len(fixed))


def cmd_adv(a):
    enh = load_enh_bundle(a.enhanced_model)
    names = enh["names"]
    orig = None
    if a.original_model:  # optional: include the original (22-feature) model alongside Enhanced
        import __main__
        if not hasattr(__main__, "StackModel"):  # models pickled while running freephish_old.py directly
            __main__.StackModel = fp.StackModel
        orig = joblib.load(a.original_model)
        if not isinstance(orig, fp.StackModel):
            raise SystemExit(f"--original-model {a.original_model} is not a freephish_old.py StackModel (.joblib from `train`).")

    test_urls = (read_url_set(a.baseline_results_pred) & read_url_set(a.enhanced_results_pred)) \
        if orig is not None and a.baseline_results_pred else read_url_set(a.enhanced_results_pred)
    test_dom = {gkey(u) for u in test_urls}
    print(f"test pages: {len(test_urls)} (from your earlier run)", flush=True)

    # ---- test set and training pool (training domains never appear in the test set) ----
    te_urls, te_html, yte = load_pages(a, lambda u: u in test_urls)
    Xte, _ = raw_features_for_bundle(te_urls, te_html, enh, a.n_jobs, cache=a.cache_test); del te_html
    tr_urls, tr_html, ytr = load_pages(a, lambda u: u not in test_urls and gkey(u) not in test_dom, a.train_per_class)
    print(f"training pool: {len(tr_urls)} pages ({int((ytr == 1).sum())} phishing / {int((ytr == 0).sum())} benign)", flush=True)
    Xtr, _ = raw_features_for_bundle(tr_urls, tr_html, enh, a.n_jobs, cache=a.cache_train); del tr_html

    src_cls = 1 if a.gan_mode == "evade" else 0

    models = []
    if orig is not None:
        models.append(("Original", lambda X: predict_original(orig, X, names)))
    models.append(("Enhanced", lambda X: predict_enhanced(enh, X, names)))
    if a.adv_train:  # optional: retrain the enhanced model with GAN-generated phishing pages added
        print("adversarial training: training the GAN used to generate training pages ...", flush=True)
        g_aug = make_plain_gan(a, Xtr, ytr, names, seed=100)
        add = g_aug.transform(Xtr[ytr == src_cls], seed2=100)
        print(f"retraining the enhanced model: clean pool + {len(add)} GAN-generated phishing pages ...", flush=True)
        m1 = RawModel().fit(np.vstack([Xtr, add]), np.concatenate([ytr, np.ones(len(add), int)]))
        models += [("Enhanced + adversarial training", m1.predict_fn)]

    # ---- the SAME adversarial test pages for every model (the GAN never sees any classifier) ----
    evalsets = []
    for sd in a.seeds:
        print(f"training attack GAN, seed {sd} ...", flush=True)
        g = make_plain_gan(a, Xtr, ytr, names, seed=sd)
        if a.gan_mode == "evade":
            idx = np.where(yte == 1)[0]
            Xev = Xte.copy(); Xev[idx] = g.transform(Xte[idx], seed2=sd); yev = yte
        else:
            ben, n_ph = np.where(yte == 0)[0], int((yte == 1).sum())
            pick = np.random.RandomState(sd).choice(ben, n_ph, replace=True)
            Xev = np.vstack([Xte[ben], g.transform(Xte[pick], seed2=sd)])
            yev = np.concatenate([np.zeros(len(ben), int), np.ones(n_ph, int)])
        evalsets.append((Xev, yev))

    rows = []
    for nm, fn in models:
        per_seed = [four(yev, fn(Xev)[0]) for Xev, yev in evalsets]
        att = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
        sd_ = {k: float(np.std([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0 for k in per_seed[0]}
        rows.append({"model": nm, "mean": att, "sd": sd_})

    print(f"\nRESULTS UNDER GAN ATTACK (mean +- sd over seeds {a.seeds}; test pages: {len(yte)})")
    print(f"{'Model':<34}{'Accuracy':>16}{'Precision':>16}{'Recall':>16}{'F1':>16}")
    for r in rows:
        cells = [f"{r['mean'][k]:.3f} +-{r['sd'][k]:.3f}" for k in ("accuracy", "precision", "recall", "f1")]
        print(f"{r['model']:<34}" + "".join(f"{c:>16}" for c in cells))
    with open(a.out, "w") as fh:
        json.dump({"gan_mode": a.gan_mode, "epsilon": a.epsilon, "perturb_groups": a.perturb_groups, "seeds": a.seeds,
                   "n_test": int(len(yte)), "n_train_pool": int(len(ytr)), "results": rows}, fh, indent=2)
    pd.DataFrame([{"model": r["model"], **r["mean"]} for r in rows]).to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)
    print(f"\nsaved {a.out} (+ .csv)")


def cmd_retrain(a):
    """Simple version: GAN-train (no surrogate) on the training pool, add the generated adversarial phishing
    rows to the enhanced model's training data, retrain, and evaluate on the ORIGINAL (unperturbed) test set -
    just before vs after on the same clean test pages, no attack table."""
    enh = load_enh_bundle(a.enhanced_model)
    names = enh["names"]
    test_urls = read_url_set(a.enhanced_results_pred)
    test_dom = {gkey(u) for u in test_urls}
    print(f"test pages: {len(test_urls)} (from your earlier Stage 2 run)", flush=True)

    te_urls, te_html, yte = load_pages(a, lambda u: u in test_urls)
    Xte, _ = raw_features_for_bundle(te_urls, te_html, enh, a.n_jobs, cache=a.cache_test); del te_html
    tr_urls, tr_html, ytr = load_pages(a, lambda u: u not in test_urls and gkey(u) not in test_dom, a.train_per_class)
    print(f"training pool: {len(tr_urls)} pages ({int((ytr == 1).sum())} phishing / {int((ytr == 0).sum())} benign)", flush=True)
    Xtr, _ = raw_features_for_bundle(tr_urls, tr_html, enh, a.n_jobs, cache=a.cache_train); del tr_html

    print("training the GAN (no surrogate) on the training pool ...", flush=True)
    g = make_plain_gan(a, Xtr, ytr, names, seed=a.seed)
    src_cls = 1 if a.gan_mode == "evade" else 0
    add = g.transform(Xtr[ytr == src_cls], seed2=a.seed)
    print(f"generated {len(add)} adversarial phishing rows from the training pool", flush=True)

    pred_before, _ = predict_enhanced(enh, Xte, names)
    m_before = four(yte, pred_before)
    if not a.after_only:
        print(f"\nEnhanced (on the clean, unattacked test set):")
        print(f"  acc {m_before['accuracy']:.3f}  prec {m_before['precision']:.3f}  rec {m_before['recall']:.3f}  f1 {m_before['f1']:.3f}")

    print("\nretraining the enhanced model on clean training pool + adversarial rows ...", flush=True)
    m1 = RawModel().fit(np.vstack([Xtr, add]), np.concatenate([ytr, np.ones(len(add), int)]))
    pred_after, _ = m1.predict_fn(Xte)
    m_after = four(yte, pred_after)
    print(f"\nEnhanced + adversarial training (on the SAME clean, unattacked test set):")
    print(f"  acc {m_after['accuracy']:.3f}  prec {m_after['precision']:.3f}  rec {m_after['recall']:.3f}  f1 {m_after['f1']:.3f}")

    if a.save_model:
        joblib.dump({"model": m1.sm, "med": m1.med, "names": names, "groups": enh.get("groups"),
                     "max_html_chars": enh.get("max_html_chars"), "version": 1, "dropped": sorted(E.DROPPED),
                     "kind": "raw_adv", "epsilon": a.epsilon, "epochs": a.epochs}, a.save_model)
        print(f"saved adversarially-trained model: {a.save_model}")

    METRIC_KEYS = ("accuracy", "precision", "recall", "f1")
    baseline_metrics, baseline_source = None, None
    if a.baseline_accuracy is not None:  # manual override: only accuracy is known this way, not the other 3 metrics
        baseline_metrics = {"accuracy": a.baseline_accuracy}
        baseline_source = "--baseline-accuracy (you typed this; precision/recall/f1 unknown)"
    elif a.enhanced_results_json:  # auto: read all 4 metrics from your Stage 2 run instead of typing a remembered number
        try:
            with open(a.enhanced_results_json) as fh:
                m = json.load(fh)["metrics"]
            baseline_metrics = {k: m[k] for k in METRIC_KEYS}
            baseline_source = f"{a.enhanced_results_json} (read automatically)"
        except (OSError, KeyError, json.JSONDecodeError) as e:
            print(f"\nnote: could not read metrics from {a.enhanced_results_json} ({type(e).__name__}: {e}); "
                  f"skipping the Stage 2 baseline check (pass --baseline-accuracy to set accuracy by hand instead).")

    # Stage 2 baseline is saved to the output json for reference but not printed - screen output is the two
    # standalone metric blocks above only, nothing compared on screen.
    baseline_check = None
    if baseline_metrics is not None:
        diffs = {k: m_before[k] - baseline_metrics[k] for k in baseline_metrics}
        baseline_check = {"baseline_source": baseline_source, "baseline": baseline_metrics,
                           "before_this_run": {k: m_before[k] for k in baseline_metrics}, "difference": diffs}

    out = {"n_test": int(len(yte)), "n_train_pool": int(len(ytr)), "n_adv_added": int(len(add)),
           "before": m_before, "after": m_after, "baseline_check": baseline_check}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {a.out}")


def cmd_tune(a):
    """Same adversarial-training procedure as `retrain` (no surrogate, no new methodology) - a proper
    hyperparameter search instead of hand-picking --epsilon/--epochs against the real test set. A
    VALIDATION split is carved from the training pool (never the test set); every (epsilon, epochs) grid
    combo is scored ONLY on that validation split. The test set is touched exactly ONCE at the end, using
    whichever combo won on validation - so the reported test number is not data-leaked by the search."""
    enh = load_enh_bundle(a.enhanced_model)
    names = enh["names"]
    test_urls = read_url_set(a.enhanced_results_pred)
    test_dom = {gkey(u) for u in test_urls}
    print(f"test pages (held out - touched ONCE, at the very end): {len(test_urls)}", flush=True)

    pool_urls, pool_html, pool_y = load_pages(a, lambda u: u not in test_urls and gkey(u) not in test_dom, a.train_per_class)
    print(f"training pool: {len(pool_urls)} pages ({int((pool_y == 1).sum())} phishing / {int((pool_y == 0).sum())} benign)", flush=True)
    Xpool, _ = raw_features_for_bundle(pool_urls, pool_html, enh, a.n_jobs, cache=a.cache_train); del pool_html

    # ---- carve a validation split out of the POOL ONLY (the test set is never involved here) ----
    rs = np.random.RandomState(a.val_seed)
    val_idx = np.sort(np.concatenate([
        rs.choice(np.where(pool_y == c)[0], max(1, int(round(a.val_frac * int((pool_y == c).sum())))), replace=False)
        for c in (0, 1)]))
    val_mask = np.zeros(len(pool_y), dtype=bool); val_mask[val_idx] = True
    Xtr, ytr = Xpool[~val_mask], pool_y[~val_mask]
    Xval, yval = Xpool[val_mask], pool_y[val_mask]
    print(f"tuning split: {len(ytr)} train / {len(yval)} validation "
          f"(validation is carved from the training pool - the real test set is untouched until the end)", flush=True)

    src_cls = 1 if a.gan_mode == "evade" else 0
    grid = [(eps, ep) for eps in a.epsilon_grid for ep in a.epochs_grid]
    print(f"\nsearching {len(grid)} configuration(s): epsilon in {a.epsilon_grid}, epochs in {a.epochs_grid} "
          f"(scored on VALIDATION only)", flush=True)

    results, best = [], None
    for eps, ep in grid:
        a.epsilon, a.epochs = eps, ep  # make_plain_gan reads these straight off `a`
        g = make_plain_gan(a, Xtr, ytr, names, seed=a.seed)
        add = g.transform(Xtr[ytr == src_cls], seed2=a.seed)
        m1 = RawModel().fit(np.vstack([Xtr, add]), np.concatenate([ytr, np.ones(len(add), int)]))
        pred_val, _ = m1.predict_fn(Xval)
        mv = four(yval, pred_val)
        results.append({"epsilon": eps, "epochs": ep, "n_adv_added": int(len(add)), "validation": mv})
        print(f"  epsilon={eps:<5} epochs={ep:<5} -> validation acc {mv['accuracy']:.3f}  "
              f"prec {mv['precision']:.3f}  rec {mv['recall']:.3f}  f1 {mv['f1']:.3f}", flush=True)
        if best is None or mv["accuracy"] > best["validation"]["accuracy"]:
            best = results[-1]

    print(f"\nbest on VALIDATION: epsilon={best['epsilon']}  epochs={best['epochs']}  (validation acc {best['validation']['accuracy']:.3f})")

    # ---- refit on the FULL pool (train+validation) with the winning config, then touch the test set once ----
    a.epsilon, a.epochs = best["epsilon"], best["epochs"]
    print(f"\nrefitting on the full training pool ({len(pool_y)} rows) with the winning config ...", flush=True)
    g_final = make_plain_gan(a, Xpool, pool_y, names, seed=a.seed)
    add_final = g_final.transform(Xpool[pool_y == src_cls], seed2=a.seed)
    m_final = RawModel().fit(np.vstack([Xpool, add_final]), np.concatenate([pool_y, np.ones(len(add_final), int)]))

    te_urls, te_html, yte = load_pages(a, lambda u: u in test_urls)
    Xte, _ = raw_features_for_bundle(te_urls, te_html, enh, a.n_jobs, cache=a.cache_test); del te_html

    pred_baseline, _ = predict_enhanced(enh, Xte, names)
    m_baseline = four(yte, pred_baseline)
    pred_final, _ = m_final.predict_fn(Xte)
    m_test = four(yte, pred_final)

    if a.save_model:
        joblib.dump({"model": m_final.sm, "med": m_final.med, "names": names, "groups": enh.get("groups"),
                     "max_html_chars": enh.get("max_html_chars"), "version": 1, "dropped": sorted(E.DROPPED),
                     "kind": "raw_adv", "epsilon": best["epsilon"], "epochs": best["epochs"]}, a.save_model)
        print(f"\nsaved adversarially-trained model: {a.save_model}")

    print(f"\nEnhanced, no adversarial training (on the held-out test set, for reference):")
    print(f"  acc {m_baseline['accuracy']:.3f}  prec {m_baseline['precision']:.3f}  rec {m_baseline['recall']:.3f}  f1 {m_baseline['f1']:.3f}")
    print(f"\nEnhanced + adversarial training, tuned config (on the held-out test set, touched once):")
    print(f"  acc {m_test['accuracy']:.3f}  prec {m_test['precision']:.3f}  rec {m_test['recall']:.3f}  f1 {m_test['f1']:.3f}")

    out = {"n_pool": int(len(pool_y)), "n_train": int(len(ytr)), "n_val": int(len(yval)), "n_test": int(len(yte)),
           "grid": results, "best_on_validation": best,
           "test_baseline_no_adv_training": m_baseline, "test_final_tuned": m_test}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    v = sp.add_parser("adv", help="Accuracy/Precision/Recall/F1 under a GAN attack on the enhanced model "
                                   "(optionally also the original model, with --original-model)")
    v.add_argument("--csv", required=True); v.add_argument("--url-col", default="url")
    v.add_argument("--label-col", default="label"); v.add_argument("--html-col"); v.add_argument("--pos-label")
    v.add_argument("--enhanced-model", required=True)
    v.add_argument("--enhanced-results-pred", default="results_enhanced.predictions.csv")
    v.add_argument("--original-model", default=None,
                   help="optional: a freephish_old.py StackModel (.joblib from `train`). Adds an 'Original' row.")
    v.add_argument("--baseline-results-pred", default="results_baseline.predictions.csv",
                   help="used only with --original-model, to restrict test pages to ones both models were evaluated on.")
    v.add_argument("--gan-mode", choices=["evade", "slide"], default="evade",
                   help="evade: G maps phishing->benign-looking. slide: literal slide, G maps benign->phishing-like")
    v.add_argument("--perturb-groups", nargs="+", default=["freephish", "url", "html"],
                   help="feature groups the attacker may change (others stay fixed)")
    v.add_argument("--epsilon", type=float, default=1.0, help="max change per perturbable feature, in std units")
    v.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    v.add_argument("--epochs", type=int, default=150); v.add_argument("--batch-size", type=int, default=64)
    v.add_argument("--lr", type=float, default=2e-4); v.add_argument("--zdim", type=int, default=16)
    v.add_argument("--lambda-l2", type=float, default=0.1, help="small penalty keeping the perturbation small")
    v.add_argument("--adv-train", action="store_true", help="optional: also retrain with GAN pages and add it to the table")
    v.add_argument("--protect"); v.add_argument("--train-per-class", type=int, default=2400)
    v.add_argument("--cache-test", default="feat_gan_test.npz"); v.add_argument("--cache-train", default="feat_adv_train.npz")
    v.add_argument("--n-jobs", type=int, default=-1); v.add_argument("--out", default="adv_results.json")

    r = sp.add_parser("retrain", help="simple: GAN-train (no surrogate) the enhanced model and evaluate on the clean test set")
    r.add_argument("--csv", required=True); r.add_argument("--url-col", default="url")
    r.add_argument("--label-col", default="label"); r.add_argument("--html-col"); r.add_argument("--pos-label")
    r.add_argument("--enhanced-model", required=True)
    r.add_argument("--enhanced-results-pred", default="results_enhanced.predictions.csv")
    r.add_argument("--gan-mode", choices=["evade", "slide"], default="evade")
    r.add_argument("--perturb-groups", nargs="+", default=["freephish", "url", "html"])
    r.add_argument("--epsilon", type=float, default=1.0)
    r.add_argument("--epochs", type=int, default=150); r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--lr", type=float, default=2e-4); r.add_argument("--zdim", type=int, default=16)
    r.add_argument("--lambda-l2", type=float, default=0.1)
    r.add_argument("--protect"); r.add_argument("--train-per-class", type=int, default=2400)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--cache-test", default="feat_gan_test.npz"); r.add_argument("--cache-train", default="feat_adv_train.npz")
    r.add_argument("--n-jobs", type=int, default=-1); r.add_argument("--out", default="retrain_results.json")
    r.add_argument("--enhanced-results-json", default="results_enhanced.json",
                    help="Stage 2's results file; metrics.accuracy read automatically as the baseline. Pass '' to skip.")
    r.add_argument("--baseline-accuracy", type=float, default=None,
                    help="override: set the baseline accuracy by hand instead of reading --enhanced-results-json.")
    r.add_argument("--after-only", action="store_true",
                    help="print only the adversarially-trained model's metrics on the clean test set, no comparison.")
    r.add_argument("--save-model", help="if set, save the adversarially-trained model to this path for later "
                                         "reuse, e.g. by freephish_classify_all.py")

    t = sp.add_parser("tune", help="same adversarial training as `retrain`, but search --epsilon/--epochs on a "
                                    "validation split; the real test set is touched once, with the winning config")
    t.add_argument("--csv", required=True); t.add_argument("--url-col", default="url")
    t.add_argument("--label-col", default="label"); t.add_argument("--html-col"); t.add_argument("--pos-label")
    t.add_argument("--enhanced-model", required=True)
    t.add_argument("--enhanced-results-pred", default="results_enhanced.predictions.csv")
    t.add_argument("--gan-mode", choices=["evade", "slide"], default="evade")
    t.add_argument("--perturb-groups", nargs="+", default=["freephish", "url", "html"])
    t.add_argument("--epsilon-grid", type=float, nargs="+", default=[0.5, 1.0, 1.5], help="--epsilon values to try")
    t.add_argument("--epochs-grid", type=int, nargs="+", default=[100, 150], help="GAN --epochs values to try")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-4); t.add_argument("--zdim", type=int, default=16)
    t.add_argument("--lambda-l2", type=float, default=0.1)
    t.add_argument("--protect"); t.add_argument("--train-per-class", type=int, default=2400)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--val-frac", type=float, default=0.2, help="fraction of each class held out as validation")
    t.add_argument("--val-seed", type=int, default=7, help="seed for the train/validation split (separate from --seed)")
    t.add_argument("--cache-test", default="feat_gan_test.npz"); t.add_argument("--cache-train", default="feat_adv_train.npz")
    t.add_argument("--n-jobs", type=int, default=-1); t.add_argument("--out", default="tune_results.json")
    t.add_argument("--save-model", help="if set, save the final adversarially-trained model (winning epsilon/epochs, "
                                         "refit on the full pool) to this path for later reuse, e.g. by freephish_classify_all.py")

    a = ap.parse_args()
    {"adv": cmd_adv, "retrain": cmd_retrain, "tune": cmd_tune}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FreePhish: Differential Evolution (DE) feature-weight search, enhanced (100-feature) model ONLY.
Requires freephish_old.py and freephish_enhanced.py in the same folder.

DE/rand/1/bin, matching the course-slide algorithm:
  Step 1 (init):      N individuals, each a vector of per-feature weights (one weight per standardized
                       enhanced feature), drawn uniformly from [--w-min, --w-max].
  Step 2 (mutation):   for each target, pick 3 other individuals a/b/c at random -> mutant = a + F*(b - c).
  Step 3 (crossover):  trial[gene] = mutant[gene] if rand() <= CR else target[gene] (>=1 gene forced from
                       the mutant, standard DE/rand/1/bin).
  Step 4 (selection):  keep the trial in place of the target if its validation accuracy is >= the target's.
  Step 5 (final):      after all generations, the best weight vector scales the feature matrix
                       (X * best_weights), a StackModel is trained on it, and reported on the held-out
                       test set -- exactly once, so the DE search can never see or fit the test set.

Fitness = validation accuracy, scored on a split carved out of the TRAINING pool only (same
val-carve-out / touch-the-test-set-once discipline as freephish_gan.py's `tune` command).

Cost: StackModel is retrained from scratch for every individual, every generation, so a run does
roughly pop-size * (generations + 1) full stack fits. Keep --pop-size/--generations small while
testing, then raise them for a final result.

  python freephish_de.py train --csv sample.parquet --pop-size 10 --generations 10
"""
import argparse, json, time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

import freephish_old as fp
import freephish_enhanced as E


def de_search(Xtr, ytr, Xval, yval, D, a):
    """Returns (best_weights, best_val_accuracy, history) for DE/rand/1/bin on per-feature weights."""
    rng = np.random.RandomState(a.seed)
    pop = rng.uniform(a.w_min, a.w_max, size=(a.pop_size, D))

    def fitness(w):
        m = fp.StackModel(seed=a.seed).fit(Xtr * w, ytr)
        return accuracy_score(yval, m.predict(Xval * w))

    t0 = time.time()
    fit = np.array([fitness(w) for w in pop])
    history = [float(fit.max())]
    print(f"  gen 0 (initial population): best val acc {fit.max():.3f}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    for gen in range(1, a.generations + 1):
        for i in range(a.pop_size):
            r1, r2, r3 = pop[rng.choice([j for j in range(a.pop_size) if j != i], 3, replace=False)]
            mutant = np.clip(r1 + a.F * (r2 - r3), a.w_min, a.w_max)
            trial = pop[i].copy()
            cross = rng.random(D) <= a.CR
            cross[rng.randint(D)] = True  # DE/rand/1/bin: force >=1 gene from the mutant
            trial[cross] = mutant[cross]
            f_trial = fitness(trial)
            if f_trial >= fit[i]:
                pop[i], fit[i] = trial, f_trial
        history.append(float(fit.max()))
        print(f"  gen {gen}/{a.generations}: best val acc {fit.max():.3f}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    best = np.argmax(fit)
    return pop[best], float(fit[best]), history


def cmd_train(a):
    groups, names, colgroups, urls, y, X, ft, grp = E.load_and_featurize(a)
    tr, te = fp.split_idx(y, grp, a.split)
    overlap = np.mean([g in set(grp[tr]) for g in grp[te]])
    print(f"\n== {a.split} split | train {len(tr)} / test {len(te)} | test rows sharing a domain with train: {overlap:.1%} ==")

    # validation split carved out of the TRAINING pool only -- DE never sees or fits the test set
    rs = np.random.RandomState(a.val_seed)
    y_tr = y[tr]
    val_pos = np.sort(np.concatenate([
        rs.choice(np.where(y_tr == c)[0], max(1, int(round(a.val_frac * int((y_tr == c).sum())))), replace=False)
        for c in (0, 1)]))
    val_mask = np.zeros(len(tr), dtype=bool); val_mask[val_pos] = True
    tr2, val = tr[~val_mask], tr[val_mask]
    print(f"DE search split: {len(tr2)} train / {len(val)} validation (carved from the training pool; "
          f"the held-out test set stays untouched until the final step)")

    search_prep = E.Preprocessor(names).fit(X[tr2])
    Xtr2, Xval = search_prep.transform(X[tr2]), search_prep.transform(X[val])

    print(f"\nDifferential Evolution: pop-size {a.pop_size}, generations {a.generations}, F {a.F}, CR {a.CR}, "
          f"weight range [{a.w_min}, {a.w_max}] ({a.pop_size * (a.generations + 1)} StackModel fits total -- this is slow)")
    best_w, best_val_acc, history = de_search(Xtr2, y[tr2], Xval, y[val], len(names), a)
    print(f"\nDE best validation accuracy: {best_val_acc:.3f}")

    # final model: refit on the FULL training pool with the best weights, test set touched exactly once
    prep = E.Preprocessor(names).fit(X[tr])
    Xp = prep.transform(X)
    t0 = time.time()
    model = fp.StackModel(seed=a.seed).fit(Xp[tr] * best_w, y[tr])
    print(f"final train time {time.time() - t0:.1f}s")
    pred = model.predict(Xp[te] * best_w)
    timing = fp.time_inference(model, Xp[te] * best_w, ft[te], a.timing_samples)
    fp.report(f"StackModel+DE n_test={len(te)}", y[te], pred, timing)

    if a.results_out:
        fp.save_results(a.results_out, "enhanced_de", np.asarray(urls)[te], y[te], pred, model.predict_proba(Xp[te] * best_w), timing,
                        {"feature_set": "enhanced+DE", "n_features": len(names), "groups": groups,
                         "split": a.split, "seed": a.seed, "n_rows_used": len(urls), "n_train": len(tr), "n_test": len(te),
                         "n_de_train": len(tr2), "n_de_val": len(val),
                         "de": {"pop_size": a.pop_size, "generations": a.generations, "F": a.F, "CR": a.CR,
                                "w_min": a.w_min, "w_max": a.w_max, "val_frac": a.val_frac, "val_seed": a.val_seed,
                                "best_val_accuracy": best_val_acc, "history": history},
                         "test_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y[te], return_counts=True))},
                         "domain_overlap": float(overlap), "csv": a.csv, "balance": a.balance, "dedup": a.dedup,
                         "fwb_only": a.fwb_only, "max_rows": a.max_rows})
    joblib.dump({"model": model, "prep": prep.state(), "names": names, "groups": groups, "weights": best_w,
                 "max_html_chars": a.max_html_chars, "version": 1, "dropped": sorted(E.DROPPED)}, a.model)
    print("saved", a.model)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train", help="DE-search feature weights on the enhanced model, report on the held-out test set")
    t.add_argument("--csv", required=True); t.add_argument("--url-col", default="url")
    t.add_argument("--label-col", default="label"); t.add_argument("--html-col"); t.add_argument("--pos-label")
    t.add_argument("--features", choices=["basic", "enhanced"], default="enhanced")
    t.add_argument("--groups", help=f"comma list from {E.GROUP_ORDER} (freephish always included)")
    t.add_argument("--screenshot-dir")
    t.add_argument("--balance", action="store_true"); t.add_argument("--max-rows", type=int)
    t.add_argument("--fwb-only", action="store_true"); t.add_argument("--dedup", action="store_true")
    t.add_argument("--n-jobs", type=int, default=-1); t.add_argument("--max-html-chars", type=int, default=500000)
    t.add_argument("--cache"); t.add_argument("--keep-timing", action="store_true")
    t.add_argument("--drop-features")
    t.add_argument("--split", choices=["group", "random"], default="group")
    t.add_argument("--timing-samples", type=int, default=0)
    t.add_argument("--model", default="model_de.joblib")
    t.add_argument("--results-out", default="results_de.json")
    t.add_argument("--pop-size", type=int, default=10, help="N individuals in the DE population")
    t.add_argument("--generations", type=int, default=10)
    t.add_argument("--F", type=float, default=0.5, help="DE mutation factor")
    t.add_argument("--CR", type=float, default=0.7, help="DE crossover probability")
    t.add_argument("--w-min", type=float, default=0.0, help="min per-feature weight")
    t.add_argument("--w-max", type=float, default=2.0, help="max per-feature weight")
    t.add_argument("--val-frac", type=float, default=0.2, help="fraction of each class, from the training pool, used as DE validation")
    t.add_argument("--val-seed", type=int, default=7)
    t.add_argument("--seed", type=int, default=42, help="DE RNG + StackModel seed")
    a = ap.parse_args()
    {"train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()

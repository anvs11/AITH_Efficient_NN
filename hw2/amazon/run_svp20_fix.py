"""Перезапуск руки svp20 с правилом noise-drop: never-learned (inf) исключаются"""
import os, time, json
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from run_cnn_svp import (prepare_features, make_proxy, make_target, fit, evaluate,
                         stratified_select, ForgettingTracker,
                         PROXY_EPOCHS, TARGET_EPOCHS, RES, SEED)

if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    X, y, Xte, yte = prepare_features()
    N = len(y)
    cnt = np.bincount(y, minlength=3).astype(np.float64)
    cw = torch.tensor(np.sqrt(N / (3 * cnt)), dtype=torch.float32); cw = cw / cw.mean()

    scores_path = os.path.join(RES, "amazon_fe_scores.npz")
    if os.path.exists(scores_path):
        scores = np.load(scores_path)["scores"]
        print(f"Скоры загружены: {scores_path}")
    else:
        print("Скоры не найдены — обучаем proxy")
        tracker = ForgettingTracker(N)
        proxy = make_proxy()
        t0 = time.time()
        fit(proxy, X, y, None, None, PROXY_EPOCHS, cw, "proxy", tracker=tracker)
        print(f"Proxy готов за {time.time() - t0:.0f}s")
        scores = tracker.scores()
        np.savez(scores_path, scores=scores, y=y)
        print(f"Скоры сохранены: {scores_path}")

    inf_mask = np.isinf(scores)
    fixed = scores.copy()
    fixed[inf_mask] = -np.inf
    print(f"Исключаем как шум: {inf_mask.sum():,} | по классам: {np.bincount(y[inf_mask], minlength=3)}")

    idx = stratified_select(fixed, y, 0.2, seed=SEED)
    np.save(os.path.join(RES, "amazon_cnn_svp20_dropinf_idx.npy"), idx)

    Xa, ya = X[idx], y[idx]
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.1, stratify=ya, random_state=SEED)
    model = make_target()
    t0 = time.time()
    fit(model, Xtr, ytr, Xva, yva, TARGET_EPOCHS, cw, "target-svp20-dropinf")
    sec = time.time() - t0
    met = evaluate(model, Xte, yte, "svp20 drop-inf")
    met.update({"n_train": int(len(idx)), "target_sec": round(sec)})

    out = os.path.join(RES, "amazon_cnn_svp20_dropinf.json")
    with open(out, "w") as f:
        json.dump({"rule": "noise-drop: never-learned excluded", "fraction": 0.2,
                   "dropped_per_class": np.bincount(y[inf_mask], minlength=3).tolist(),
                   "arm": met}, f, indent=2)
    print(f"Готово: {out}")

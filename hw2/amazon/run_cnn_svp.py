"""CNN + Selection via Proxy (Forgetting Events) на Amazon

Proxy  : маленький multi-branch 1D-CNN (1 conv-слой на ветку), 100% данных,
         по эпохам считаем forgetting events.
Target : глубокий multi-branch 1D-CNN (3 conv-слоя на ветку), учится с нуля
         на стратифицированном подмножестве, отобранном по forgetting events.
Baseline: stratified random того же размера; Reference: target на 100%.
"""
import os, gc, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)
DEVICE = torch.device("cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "amazon")
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(DATA, "cnn_features_cache.npz")

PROXY_EPOCHS, TARGET_EPOCHS = 40, 60
BATCH, EVAL_BATCH, LR, WD = 2048, 8192, 1e-3, 1e-4
FRACTIONS = [0.2, 0.5]
GROUPS = [(0, 8), (8, 40), (40, 48), (48, 72), (72, 104)]


def prepare_features():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["X_train"], z["y_train"], z["X_test"], z["y_test"]
    print("Готовим признаки (первый раз ~5-7 мин, потом кэш)...")
    df = pd.read_parquet(os.path.join(DATA, "df_products_clean.parquet"))
    df_reviews = pd.read_parquet(os.path.join(DATA, "df_reviews.parquet"))
    df["first_review"] = pd.to_datetime(df["first_review"], errors="coerce")
    tr = df[df["first_review"] < "2022-07-01"].copy().reset_index(drop=True)
    te = df[df["first_review"] >= "2022-07-01"].copy().reset_index(drop=True)
    med = tr["price"].median()
    for d in (tr, te):
        d["log_price"] = np.log1p(d["price"].fillna(med))
        d["has_price"] = (d["price"] > 0).fillna(False).astype(int)
        d["title_length"] = d["title"].astype(str).str.len()
        d["desc_length"] = d["description"].astype(str).str.len()
        d["has_exclamation"] = (d["title"].astype(str).str.contains("!") |
                                d["description"].astype(str).str.contains("!")).astype(int)
        d["is_minimalist_desc"] = (d["desc_length"] <= 50).astype(int)
        d["store"] = d["store"].fillna("Unknown").astype(str)
    sf = (tr["store"].value_counts() / len(tr)).to_dict()
    tr["store_freq"] = tr["store"].map(sf).fillna(0.0)
    te["store_freq"] = te["store"].map(sf).fillna(0.0)
    df_reviews["review_date"] = pd.to_datetime(df_reviews["timestamp"], unit="s", errors="coerce").dt.floor("D")
    first = pd.concat([tr[["parent_asin", "first_review"]], te[["parent_asin", "first_review"]]]).drop_duplicates(
        "parent_asin")
    m = df_reviews.merge(first, on="parent_asin", how="inner")
    m["days_since_first"] = (m["review_date"] - m["first_review"]).dt.days
    w2 = m[m["days_since_first"] <= 14].copy()

    stats = w2.groupby("parent_asin").agg(
        week2_review_count=("parent_asin", "count"),
        week2_avg_rating=("rating", "mean"),
        week2_rating_std=("rating", "std"),
        week2_text_len_avg=("text", lambda x: x.astype(str).str.len().mean()),
        week2_emotion_count=("text", lambda x: x.astype(str).str.contains("!").sum()),
        week2_5_star_ratio=("rating", lambda x: (x == 5.0).mean())
    ).reset_index()

    stats["week2_review_velocity"] = stats["week2_review_count"] / 14.0
    for c in ["week2_review_count", "week2_emotion_count"]:
        stats[c] = stats[c].fillna(0).astype(int)
    for c in ["week2_avg_rating", "week2_rating_std", "week2_text_len_avg", "week2_5_star_ratio"]:
        stats[c] = stats[c].fillna(0)

    tr = tr.merge(stats, on="parent_asin", how="left")
    te = te.merge(stats, on="parent_asin", how="left")
    for c in stats.columns:
        if c != "parent_asin":
            tr[c] = tr[c].fillna(0)
            te[c] = te[c].fillna(0)

    tr["price_x_week2_reviews"] = tr["log_price"] * tr["week2_review_count"]
    te["price_x_week2_reviews"] = te["log_price"] * te["week2_review_count"]

    e_tr = np.memmap(os.path.join(DATA, "nlp_train_embs.npy"), dtype="float32", mode="r", shape=(len(tr), 384))
    e_te = np.memmap(os.path.join(DATA, "nlp_test_embs.npy"), dtype="float32", mode="r", shape=(len(te), 384))

    ipca = IncrementalPCA(n_components=32, batch_size=2000)
    for i in range(0, len(tr), 2000):
        ipca.partial_fit(e_tr[i:i + 2000])

    nlp_tr = ipca.transform(e_tr).astype(np.float32)
    nlp_te = ipca.transform(e_te).astype(np.float32)

    for i in range(32):
        tr[f"nlp_sbert_{i}"] = nlp_tr[:, i]
        te[f"nlp_sbert_{i}"] = nlp_te[:, i]

    del e_tr, e_te, nlp_tr, nlp_te, ipca
    gc.collect()

    em = np.memmap(os.path.join(DATA, "review_embs.npy"), dtype="float32", mode="r", shape=(len(w2), 384))
    ec = [f"emb_{i}" for i in range(384)]

    w2 = w2.reset_index(drop=True)
    w2[ec] = np.array(em)

    agg_mean = w2.groupby("parent_asin")[ec].mean().reset_index()
    agg_std = w2.groupby("parent_asin")[ec].std().reset_index()

    mc = [f"m{i}" for i in range(384)]
    sc = [f"s{i}" for i in range(384)]

    agg_mean.columns = ["parent_asin"] + mc
    agg_std.columns = ["parent_asin"] + sc

    agg = agg_mean.merge(agg_std, on="parent_asin", how="left")
    agg = agg.fillna(0).replace([np.inf, -np.inf], 0)

    del em, w2, agg_mean, agg_std
    gc.collect()

    a_tr = agg[agg["parent_asin"].isin(set(tr["parent_asin"]))].copy()
    pm = PCA(16, random_state=SEED).fit(a_tr[mc].values)
    ps = PCA(8, random_state=SEED).fit(a_tr[sc].values)

    rcols = [f"review_sbert_mean_{i}" for i in range(16)] + [f"review_sbert_std_{i}" for i in range(8)]

    rt = pd.DataFrame({"parent_asin": a_tr["parent_asin"].values})
    rt[rcols[:16]] = pm.transform(a_tr[mc].values).astype(np.float32)
    rt[rcols[16:]] = ps.transform(a_tr[sc].values).astype(np.float32)

    a_te = agg[agg["parent_asin"].isin(set(te["parent_asin"]))].copy()
    re_ = pd.DataFrame({"parent_asin": a_te["parent_asin"].values})
    re_[rcols[:16]] = pm.transform(a_te[mc].values).astype(np.float32)
    re_[rcols[16:]] = ps.transform(a_te[sc].values).astype(np.float32)

    tr = tr.merge(rt, on="parent_asin", how="left")
    te = te.merge(re_, on="parent_asin", how="left")

    for c in rcols:
        tr[c] = tr[c].fillna(0).astype(np.float32)
        te[c] = te[c].fillna(0).astype(np.float32)

    # one-hot топ-32 магазинов и сборка матрицы
    top = tr["store"].value_counts().head(32).index.tolist()
    for d in (tr, te):
        for i, s in enumerate(top):
            d[f"st_{i}"] = (d["store"] == s).astype(np.float32)
    e_tr = np.memmap(os.path.join(DATA, 'nlp_train_embs.npy'), dtype='float32',
                     mode='r', shape=(len(tr), 384))
    e_te = np.memmap(os.path.join(DATA, 'nlp_test_embs.npy'), dtype='float32',
                     mode='r', shape=(len(te), 384))
    ipca = IncrementalPCA(n_components=32, batch_size=2000)
    for i in range(0, len(tr), 2000):
        ipca.partial_fit(e_tr[i:i + 2000])
    nlp_tr = ipca.transform(e_tr).astype(np.float32)
    nlp_te = ipca.transform(e_te).astype(np.float32)
    del e_tr, e_te, ipca
    gc.collect()
    for i in range(32):
        tr[f'nlp_sbert_{i}'] = nlp_tr[:, i]
        te[f'nlp_sbert_{i}'] = nlp_te[:, i]
    base = ["log_price", "has_price", "num_images", "title_length", "desc_length",
            "has_exclamation", "is_minimalist_desc", "store_freq"]
    nlp = [f"nlp_sbert_{i}" for i in range(32)]
    w2c = ["week2_review_count", "week2_avg_rating", "week2_rating_std", "week2_text_len_avg",
           "week2_emotion_count", "week2_review_velocity", "price_x_week2_reviews", "week2_5_star_ratio"]
    cols = base + nlp + w2c + rcols + [f"st_{i}" for i in range(32)]
    X_tr = tr[cols].to_numpy(np.float32); X_te = te[cols].to_numpy(np.float32)
    mu = X_tr[:, :72].mean(0); sd = X_tr[:, :72].std(0); sd[sd == 0] = 1.0
    X_tr[:, :72] = (X_tr[:, :72] - mu) / sd
    X_te[:, :72] = (X_te[:, :72] - mu) / sd
    y_tr = tr["target_class"].to_numpy(np.int64); y_te = te["target_class"].to_numpy(np.int64)
    np.savez(CACHE, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
    return X_tr, y_tr, X_te, y_te


class Branch(nn.Module):
    def __init__(self, chans, kers):
        super().__init__()
        layers, c = [], 1
        for ch, k in zip(chans, kers):
            layers += [nn.Conv1d(c, ch, k, padding=k // 2), nn.BatchNorm1d(ch), nn.ReLU()]
            c = ch
        self.net = nn.Sequential(*layers)
        self.out = c * 2

    def forward(self, x):
        h = self.net(x.unsqueeze(1))
        return torch.cat([h.amax(-1), h.mean(-1)], dim=-1)


class GroupCNN(nn.Module):
    def __init__(self, chans, kers, fc, drop, n_cls=3):
        super().__init__()
        self.groups = GROUPS
        self.branches = nn.ModuleList([Branch(chans, kers) for _ in GROUPS])
        d = sum(b.out for b in self.branches)
        head = []
        for h in fc:
            head += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        head += [nn.Linear(d, n_cls)]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        outs = [b(x[:, s:e]) for b, (s, e) in zip(self.branches, self.groups)]
        return self.head(torch.cat(outs, dim=-1))


def make_proxy():
    return GroupCNN([24], [5], [64], 0.1)

def make_target():
    return GroupCNN([64, 128, 128], [5, 3, 3], [256, 128], 0.3)


class ForgettingTracker:
    def __init__(self, n):
        self.nevents = np.zeros(n, np.int64)
        self.last = np.zeros(n, bool)
        self.ever = np.zeros(n, bool)

    def update(self, correct):
        self.nevents += self.last & ~correct
        self.last = correct
        self.ever |= correct

    def scores(self):
        s = self.nevents.astype(np.float64)
        s[~self.ever] = np.inf  # как в статье - ни разу не выученные в топ
        return s


@torch.no_grad()
def predict(model, X, batch=EVAL_BATCH):
    model.eval()
    Xt = torch.from_numpy(X)
    return np.concatenate([model(Xt[i:i + batch]).softmax(-1).numpy()
                           for i in range(0, len(X), batch)])


def fit(model, X, y, Xv, yv, epochs, cw, tag, tracker=None):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Xt, yt, n = torch.from_numpy(X), torch.from_numpy(y), len(y)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
        sch.step()
        if tracker is not None and (ep + 1) % 2 == 0:
            tracker.update(predict(model, X).argmax(1) == y)
        msg = f"[{tag}] epoch {ep + 1:3d}"
        if Xv is not None:
            vf = f1_score(yv, predict(model, Xv).argmax(1), average="macro")
            if vf > best:
                best, best_state = vf, {k: v.clone() for k, v in model.state_dict().items()}
            msg += f" | val_f1 {vf:.4f} (best {best:.4f})"
        if (ep + 1) % 10 == 0 or ep == 0:
            print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def stratified_select(scores, y, fraction, seed=SEED, random=False):
    rng = np.random.default_rng(seed)
    sel = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = int(round(len(idx) * fraction))
        if random:
            pick = rng.choice(idx, k, replace=False)
        else:
            noise = rng.random(len(idx)) * 1e-12
            pick = idx[np.argsort(-(scores[idx] + noise))]
            pick = pick[:k]
        sel.append(pick)
    return np.sort(np.concatenate(sel))


def evaluate(model, X, y, name):
    p = predict(model, X)
    pred = p.argmax(1)
    f1 = f1_score(y, pred, average="macro")
    thr = np.percentile(p[:, 2], 95)
    hr = ((p[:, 2] >= thr) & (y == 2)).sum() / max((y == 2).sum(), 1)
    hit_f1 = f1_score(y == 2, pred == 2, zero_division=0)
    print(f"\nTARGET: {name}:")
    print(classification_report(y, pred, target_names=["Flop", "Stable", "Hit"]))
    print(f"Macro F1 {f1:.4f} | HitRate@5% {hr:.2%} | Hit F1 {hit_f1:.4f}")
    return {"macro_f1": float(f1), "hitrate5": float(hr), "hit_f1": float(hit_f1)}


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    X, y, Xte, yte = prepare_features()
    print(f"Train {X.shape} | Test {Xte.shape} | prep {time.time() - t0:.0f}s")

    X_pool, X_val, y_pool, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
    N_pool = len(y_pool)

    cnt = np.bincount(y_pool, minlength=3).astype(np.float64)
    cw = torch.tensor(np.sqrt(N_pool / (3 * cnt)), dtype=torch.float32)
    cw = cw / cw.mean()
    print(f"Class weights (sqrt-balanced): {cw.numpy().round(3)}")
    print(f"Proxy params: {sum(p.numel() for p in make_proxy().parameters()):,} | "
          f"Target params: {sum(p.numel() for p in make_target().parameters()):,}")

    tracker = ForgettingTracker(N_pool)
    proxy = make_proxy()
    t0 = time.time()
    fit(proxy, X_pool, y_pool, None, None, PROXY_EPOCHS, cw, "proxy", tracker=tracker)
    proxy_sec = time.time() - t0
    scores = tracker.scores()
    inf_mask = np.isinf(scores)
    print(f"Proxy готов за {proxy_sec:.0f}s | inf-примеров: {inf_mask.sum():,} "
          f"по классам: {np.bincount(y_pool[inf_mask], minlength=3)}")

    arms = {"full": np.arange(N_pool)}
    for fr in FRACTIONS:
        tag = int(fr * 100)
        arms[f"random{tag}"] = stratified_select(scores, y_pool, fr, random=True)
        arms[f"svp{tag}"] = stratified_select(scores, y_pool, fr, random=False)

    results = {}
    for name, idx in arms.items():
        Xa, ya = X_pool[idx], y_pool[idx]
        model = make_target()
        t0 = time.time()
        fit(model, Xa, ya, X_val, y_val, TARGET_EPOCHS, cw, f"target-{name}")
        sec = time.time() - t0
        met = evaluate(model, Xte, yte, name)
        met.update({"n_train": int(len(idx)), "target_sec": round(sec)})
        results[name] = met
        if name != "full":
            np.save(os.path.join(RES, f"amazon_cnn_{name}_idx.npy"), idx)
        del model;
        gc.collect()

    full_f1 = results["full"]["macro_f1"]
    print("\n" + "=" * 74)
    print(f"{'Arm':<11}{'N train':>9}{'MacroF1':>9}{'%full':>7}{'HitRate@5%':>12}{'HitF1':>7}{'sec':>7}")
    print("-" * 74)
    for name, m in results.items():
        print(f"{name:<11}{m['n_train']:>9,}{m['macro_f1']:>9.4f}"
              f"{100 * m['macro_f1'] / full_f1:>6.1f}%{m['hitrate5']:>11.2%}"
              f"{m['hit_f1']:>7.4f}{m['target_sec']:>7}")
    print(f"proxy_sec: {proxy_sec:.0f}")

    with open(os.path.join(RES, "amazon_cnn_svp.json"), "w") as f:
        json.dump({"proxy_sec": proxy_sec, "proxy_epochs": PROXY_EPOCHS,
                   "target_epochs": TARGET_EPOCHS, "arms": results}, f, indent=2)
    print("Готово. Результаты в results/amazon_cnn_svp.json")

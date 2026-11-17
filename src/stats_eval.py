"""
Ablasyon, istatistiksel anlamlılık, gerçek oranlı test kümesi ve açıklanabilirlik uyumu (İP6).

1. Üç model aynı kopma maskeleriyle karşılaştırılır: çift başlıklı MAE, 1D-ResNet (LeadDropout'suz,
   hipotezdeki "klasik mimari") ve 1D-ResNet + LeadDropout (ablasyon: kazanç mimariden mi, maskelemeden mi?).
2. Bootstrap %95 güven aralıkları (kayıt düzeyinde yeniden örnekleme) ve McNemar testi (MI kayıtlarında
   MAE ile karşılaştırma modelinin doğru/yanlış uyuşmazlığı, kesin binom testi).
3. Gerçek oranlı test: strat_fold 10'daki tüm kayıtlar; MI olmayan diğer tanılar (ST/T değişikliği,
   hipertrofi, ileti bozukluğu) "MI değil" sayılır. Karar eşikleri değişmez (doğrulamada seçilmişti).
4. Dikkat–duvar uyumu: tek duvarlı MI kayıtlarında Spatial Attention'ın en çok yöneldiği duvar grubunun
   gerçek MI duvarıyla örtüşme oranı.

Çalıştırma: python -m src.stats_eval
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import PTBXLProcessedDataset
from src.ecg_constants import WALL_LEAD_INDICES, WALL_NAMES
from src.evaluate import scenarios
from src.losses import MultiTaskLoss
from src.train import BASELINE_PATH, CHECKPOINT_PATH, autocast, evaluate, get_device, load_checkpoint, model_forward

MODELS = {
    "MAE": CHECKPOINT_PATH,
    "1D-ResNet": BASELINE_PATH,
    "1D-ResNet + LeadDropout": "models/resnet1d_leaddropout.pth",
}
N_BOOT = 2000


def predictions(model, ckpt, loader, device):
    """Senaryo -> (etiketler, olasılıklar); maskeler sabit tohumla üretildiğinden modeller arasında aynıdır."""
    criterion = MultiTaskLoss(focal_alpha=ckpt.get("focal_alpha"))
    out = {}
    for name, mask_fn in scenarios().items():
        r = evaluate(model, loader, criterion, device, mask_fn, seed=123, threshold=ckpt["threshold"])
        out[name] = (np.asarray(r["_labels"]), np.asarray(r["_probs"]))
    return out


def metrics(y, p, thr):
    pred = p >= thr
    tp, fn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 1))
    tn, fp = np.sum(~pred & (y == 0)), np.sum(pred & (y == 0))
    sens = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    return {"auroc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"),
            "f1": 2 * prec * sens / max(prec + sens, 1e-12), "sens": sens, "spec": tn / max(tn + fp, 1),
            "fnr": 1 - sens, "precision": prec}


def ci(values):
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap(preds, thresholds, drop_names, rng):
    """Her yeniden örneklemede aynı kayıt kümesi tüm senaryo ve modellere uygulanır (eşleştirilmiş)."""
    n = len(next(iter(next(iter(preds.values())).values()))[0])
    res = {m: {s: {k: [] for k in ("auroc", "f1", "sens", "spec")} for s in preds[m]} for m in preds}
    pooled = {m: [] for m in preds}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for m in preds:
            fn_total = mi_total = 0
            for s, (y, p) in preds[m].items():
                yb, pb = y[idx], p[idx]
                r = metrics(yb, pb, thresholds[m])
                for k in res[m][s]:
                    res[m][s][k].append(r[k])
                if s in drop_names:
                    fn_total += np.sum((pb < thresholds[m]) & (yb == 1))
                    mi_total += np.sum(yb == 1)
            pooled[m].append(fn_total / max(mi_total, 1))
    out = {m: {s: {k: ci(v) for k, v in d.items()} for s, d in res[m].items()} for m in res}
    pooled = {m: np.asarray(v) for m, v in pooled.items()}
    base = pooled["1D-ResNet"]
    reduction = {m: ci((base - pooled[m]) / np.where(base > 0, base, np.nan)) for m in pooled if m != "1D-ResNet"}
    return out, {m: ci(v) for m, v in pooled.items()}, reduction


def mcnemar(y, p_a, thr_a, p_b, thr_b, cls=1):
    """
    cls=1: MI kayıtları (duyarlılık), cls=0: MI olmayan kayıtlar (özgüllük).
    b = A doğru & B yanlış, c = A yanlış & B doğru; kesin (binom) McNemar p değeri.
    """
    sel = y == cls
    a_ok, b_ok = (p_a[sel] >= thr_a) == bool(cls), (p_b[sel] >= thr_b) == bool(cls)
    b, c = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"A_dogru_B_yanlis": b, "A_yanlis_B_dogru": c, "p": float(p)}


@torch.no_grad()
def attention_wall_agreement(model, ckpt, loader, device):
    """Tek duvarlı MI kayıtlarında (12 kanal): dikkat payı en yüksek duvar grubu = gerçek duvar mı?"""
    model.eval()
    groups = [torch.as_tensor(WALL_LEAD_INDICES[w]) for w in WALL_NAMES]
    true_walls, attn_pick, head_pick = [], [], []
    for x, target in loader:
        walls = target["walls"].numpy()
        keep = (target["mi"].numpy() == 1) & (walls.sum(axis=1) == 1)
        if not keep.any():
            continue
        x = x.to(device)
        tgt = {k: v.to(device) for k, v in target.items()}
        M = torch.ones(x.shape[0], 12, device=device)
        with autocast(device, True):
            out = model_forward(model, x, M, tgt)
        imp = out["lead_importance"].float().cpu()
        group_attn = torch.stack([imp[:, g].mean(dim=1) for g in groups], dim=1).numpy()
        head = out["wall_logits"].float().cpu().numpy()
        true_walls.append(walls[keep].argmax(axis=1))
        attn_pick.append(group_attn[keep].argmax(axis=1))
        head_pick.append(head[keep].argmax(axis=1))
    t, a, h = (np.concatenate(v) for v in (true_walls, attn_pick, head_pick))
    per_wall = {w: {"n": int(np.sum(t == i)), "dikkat_uyumu": float(np.mean(a[t == i] == i)) if np.any(t == i) else None,
                    "duvar_basligi_uyumu": float(np.mean(h[t == i] == i)) if np.any(t == i) else None}
                for i, w in enumerate(WALL_NAMES)}
    majority = max(np.mean(t == i) for i in range(len(WALL_NAMES)))
    return {"n": int(len(t)), "dikkat_uyumu": float(np.mean(a == t)), "duvar_basligi_uyumu": float(np.mean(h == t)),
            "sans_duzeyi_cogunluk_sinifi": float(majority), "duvar_basina": per_wall}


def main():
    parser = argparse.ArgumentParser(description="Ablasyon, istatistik, gerçek oranlı test, dikkat uyumu")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default="results/ablasyon/istatistik.json")
    args = parser.parse_args()

    device = get_device()
    rng = np.random.default_rng(0)
    balanced = DataLoader(PTBXLProcessedDataset("test", args.processed_dir), batch_size=64, shuffle=False)
    full_set = PTBXLProcessedDataset("test", args.processed_dir, balanced=False, include_other=True)
    full = DataLoader(full_set, batch_size=64, shuffle=False)
    drop_names = [s for s in scenarios() if not s.startswith("0 kopuk")]

    preds, preds_full, thresholds, info = {}, {}, {}, {}
    for name, path in MODELS.items():
        model, ckpt = load_checkpoint(path, device)
        thresholds[name] = ckpt["threshold"]
        info[name] = {"checkpoint": path, "epoch": ckpt["epoch"], "val_auroc": ckpt["val_metrics"]["auroc"],
                      "threshold": ckpt["threshold"]}
        print(f"{name}: {path} (epoch {ckpt['epoch']}, val AUROC {ckpt['val_metrics']['auroc']:.4f}, "
              f"eşik {ckpt['threshold']:.2f})")
        preds[name] = predictions(model, ckpt, balanced, device)
        preds_full[name] = predictions(model, ckpt, full, device)
        if name == "MAE":
            agreement = attention_wall_agreement(model, ckpt, balanced, device)

    point = {m: {s: metrics(y, p, thresholds[m]) for s, (y, p) in preds[m].items()} for m in preds}
    cis, pooled_ci, reduction_ci = bootstrap(preds, thresholds, drop_names, rng)
    pooled = {m: float(np.mean([point[m][s]["fnr"] for s in drop_names])) for m in preds}
    tests = {}
    for cls, label in ((1, "mi_kayitlari_duyarlilik"), (0, "normal_kayitlar_ozgulluk")):
        tests[label] = {}
        for other in ("1D-ResNet", "1D-ResNet + LeadDropout"):
            per = {}
            ys, pa, pb = [], [], []
            for s in drop_names:
                y, p_mae = preds["MAE"][s]
                _, p_o = preds[other][s]
                per[s] = mcnemar(y, p_mae, thresholds["MAE"], p_o, thresholds[other], cls)
                ys.append(y), pa.append(p_mae), pb.append(p_o)
            per["havuzlanmış (tüm kopma senaryoları)"] = mcnemar(np.concatenate(ys), np.concatenate(pa),
                                                                 thresholds["MAE"], np.concatenate(pb),
                                                                 thresholds[other], cls)
            tests[label][f"MAE vs {other}"] = per

    meta = full_set.meta
    other_classes = {c: ((meta["label"] == -1) & (meta[c] == 1)).to_numpy() for c in ("STTC", "HYP", "CD")}
    normal = (meta["label"] == 0).to_numpy()
    full_res = {}
    for m in preds_full:
        full_res[m] = {}
        for s, (y, p) in preds_full[m].items():
            r = metrics(y, p, thresholds[m])
            alarm = p >= thresholds[m]
            r["yanlis_alarm_orani"] = {"Normal": float(alarm[normal].mean()),
                                       **{c: float(alarm[v].mean()) for c, v in other_classes.items() if v.any()}}
            full_res[m][s] = r

    report = {
        "modeller": info,
        "dengeli_test": {"nokta": point, "guven_araligi_95": cis},
        "havuzlanmis_fnr": {"nokta": pooled, "guven_araligi_95": pooled_ci},
        "yanlis_negatif_azalmasi_resnete_gore": {
            m: {"nokta": (pooled["1D-ResNet"] - pooled[m]) / pooled["1D-ResNet"], "guven_araligi_95": reduction_ci[m]}
            for m in reduction_ci},
        "mcnemar": tests,
        "gercek_oranli_test": {"kayit": int(len(full_set)), "mi": int(full_set.labels.sum()),
                               "sinif_sayilari": {"Normal": int(normal.sum()),
                                                  **{c: int(v.sum()) for c, v in other_classes.items()}},
                               "sonuc": full_res},
        "dikkat_duvar_uyumu": agreement,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)

    print("\n=== Dengeli test (550 MI + 550 Normal): F1 [%95 GA] ===")
    for s in point["MAE"]:
        row = "  ".join(f"{m[:24]:24s} {point[m][s]['f1']:.3f} [{cis[m][s]['f1'][0]:.3f}, {cis[m][s]['f1'][1]:.3f}]"
                        for m in point)
        print(f"{s[:30]:30s} {row}")
    print("\n=== Havuzlanmış FNR ve 1D-ResNet'e göre azalma ===")
    for m in pooled:
        red = report["yanlis_negatif_azalmasi_resnete_gore"].get(m)
        extra = f"  azalma %{100 * red['nokta']:.1f} [%{100 * red['guven_araligi_95'][0]:.1f}, %{100 * red['guven_araligi_95'][1]:.1f}]" if red else ""
        print(f"{m:26s} FNR {pooled[m]:.3f} [{pooled_ci[m][0]:.3f}, {pooled_ci[m][1]:.3f}]{extra}")
    for label, block in tests.items():
        print(f"\n=== McNemar, havuzlanmış kopma senaryoları: {label} ===")
        for k, v in block.items():
            h = v["havuzlanmış (tüm kopma senaryoları)"]
            print(f"{k:38s} MAE doğru/diğeri yanlış {h['A_dogru_B_yanlis']:4d}  tersi {h['A_yanlis_B_dogru']:4d}  "
                  f"p = {h['p']:.2e}")
    print(f"\n=== Gerçek oranlı test: {len(full_set)} kayıt, {int(full_set.labels.sum())} MI ===")
    for m in full_res:
        r0 = full_res[m]["0 kopuk (tam 12 kanal)"]
        r4 = full_res[m]["rastgele 4 kanal"]
        print(f"{m:26s} 12 kanal: AUROC {r0['auroc']:.3f} F1 {r0['f1']:.3f} kesinlik {r0['precision']:.3f} "
              f"duyarlılık {r0['sens']:.3f} | 4 kopuk: F1 {r4['f1']:.3f} | yanlış alarm "
              + ", ".join(f"{c} {v:.2f}" for c, v in r0["yanlis_alarm_orani"].items()))
    a = agreement
    print(f"\n=== Dikkat–duvar uyumu (tek duvarlı {a['n']} MI kaydı, 12 kanal) ===")
    print(f"dikkat {a['dikkat_uyumu']:.3f}  duvar başlığı {a['duvar_basligi_uyumu']:.3f}  "
          f"çoğunluk sınıfı (şans) {a['sans_duzeyi_cogunluk_sinifi']:.3f}")
    for w, v in a["duvar_basina"].items():
        print(f"  {w:9s} n={v['n']:3d}  dikkat {v['dikkat_uyumu']}  başlık {v['duvar_basligi_uyumu']}")
    print(f"\nSonuçlar: {args.out}")


if __name__ == "__main__":
    main()

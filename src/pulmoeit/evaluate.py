"""Evaluation: image quality, clinical indices, robustness to shift, QC and quality gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import dataset
from .metrics import pearson, roc_auc, summarize
from .models import load_checkpoint, predict
from .monitoring import MeasurementQC, drift_report
from .phantom import SCENARIOS
from .recon import LinearReconstructor
from .signals import features

SPLITS = ("test", "test_shift")


def evaluate(params: dict, data_dir: Path, model_dir: Path, report_dir: Path) -> dict:
    baseline = LinearReconstructor.load(model_dir / "baseline.npz")
    model = load_checkpoint(model_dir / "model.pt")
    qc = MeasurementQC.load(model_dir / "qc.npz")
    results: dict = {}
    examples = None

    for split in SPLITS:
        d = dataset.load(data_dir / f"{split}.npz")
        x, y = features(d["v_ref"], d["v_insp"]), d["target"]
        pb, pm = baseline(x), predict(model, x)
        res = {"baseline": summarize(pb, y), "model": summarize(pm, y), "by_scenario": {}}
        for k, name in enumerate(SCENARIOS):
            sel = d["scenario"] == k
            if sel.any():
                res["by_scenario"][name] = {
                    "n": int(sel.sum()),
                    "baseline_corr": float(np.mean(pearson(pb[sel], y[sel]))),
                    "model_corr": float(np.mean(pearson(pm[sel], y[sel]))),
                }
        res["qc_drift"] = drift_report(qc.reference_scores, qc.score(x))
        results[split] = res
        if split == "test":
            examples = (d["scenario"], y, pb, pm)

    clean = features(*_frames(data_dir / "test.npz"))
    faulty_data = dataset.load(data_dir / "test_fault.npz")
    faulty = features(faulty_data["v_ref"], faulty_data["v_insp"])
    s_clean, s_fault = qc.score(clean), qc.score(faulty)
    flagged = s_fault > qc.threshold
    located = qc.suspect_electrode(faulty[flagged]) == faulty_data["fault"][flagged]
    results["qc"] = {
        "fault_auc": roc_auc(s_clean, s_fault),
        "fault_detection_rate": float(flagged.mean()),
        "false_alarm_rate": float((s_clean > qc.threshold).mean()),
        "electrode_localisation_acc": float(located.mean()) if located.size else 0.0,
    }
    results["gates"] = _gates(params.get("gates", {}), results)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (report_dir / "report.md").write_text(_markdown(results))
    _figure(report_dir / "figures" / "examples.png", *examples)
    return results


def _frames(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = dataset.load(path)
    return d["v_ref"], d["v_insp"]


def _gates(cfg: dict, r: dict) -> dict:
    test = r["test"]
    checks = {
        "min_corr": test["model"]["corr"] >= cfg.get("min_corr", 0.0),
        "min_corr_gain_over_baseline": test["model"]["corr"] - test["baseline"]["corr"]
        >= cfg.get("min_corr_gain_over_baseline", -1.0),
        "max_cov_mae": test["model"]["cov_mae"] <= cfg.get("max_cov_mae", np.inf),
        "min_fault_auc": r["qc"]["fault_auc"] >= cfg.get("min_fault_auc", 0.0),
    }
    return {**{k: bool(v) for k, v in checks.items()}, "passed": bool(all(checks.values()))}


def _markdown(r: dict) -> str:
    lines = ["# Rapport d'évaluation PulmoEIT", ""]
    for split in SPLITS:
        lines += [
            f"## {split}",
            "",
            "| méthode | RMSE | corrélation | erreur GI | erreur CoV (pts %) | erreur régionale (pts %) |",
            "|---|---|---|---|---|---|",
        ]
        for m in ("baseline", "model"):
            s = r[split][m]
            lines.append(
                f"| {m} | {s['rmse']:.4f} | {s['corr']:.3f} | {s['gi_mae']:.3f} | {s['cov_mae']:.2f} | {s['regional_mae']:.2f} |"
            )
        lines += ["", "| scénario | n | corr. baseline | corr. modèle |", "|---|---|---|---|"]
        for name, s in r[split]["by_scenario"].items():
            lines.append(f"| {name} | {s['n']} | {s['baseline_corr']:.3f} | {s['model_corr']:.3f} |")
        drift = r[split]["qc_drift"]
        lines += [
            "",
            f"Dérive QC (KS) : D = {drift['ks_statistic']:.3f}, p = {drift['p_value']:.2g}, dérive = {drift['drift']}",
            "",
        ]
    q = r["qc"]
    lines += [
        "## Contrôle qualité des mesures (électrode décollée)",
        "",
        f"- AUC détection : {q['fault_auc']:.3f}",
        f"- Taux de détection au seuil : {q['fault_detection_rate']:.1%} (fausses alarmes : {q['false_alarm_rate']:.1%})",
        f"- Localisation de l'électrode fautive : {q['electrode_localisation_acc']:.1%}",
        "",
        "## Quality gates",
        "",
    ]
    lines += [f"- {k} : {'OK' if v else 'ÉCHEC'}" for k, v in r["gates"].items()]
    return "\n".join(lines) + "\n"


def _figure(path: Path, scenario, target, base, model) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .grid import image_mask

    picks = [int(np.flatnonzero(scenario == k)[0]) for k in range(len(SCENARIOS)) if (scenario == k).any()]
    cols = (("Vérité terrain", target), ("NOSER (linéaire)", base), ("PostUNet (appris)", model))
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.3 * len(picks)), constrained_layout=True)
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("white")
    mask = image_mask()
    for row, i in enumerate(picks):
        vmax = max(float(target[i].max()), 1e-6)
        for col, (title, imgs) in enumerate(cols):
            ax = axes[row, col]
            im = ax.imshow(np.where(mask, imgs[i], np.nan), cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=10, color="#333333")
            if col == 0:
                ax.set_ylabel(SCENARIOS[scenario[i]].replace("_", " "), fontsize=9, color="#555555")
        fig.colorbar(im, ax=axes[row, :], shrink=0.8, label="ventilation tidale")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)

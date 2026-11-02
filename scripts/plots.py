"""Regenerate results/manifest.json, results/summary.md and the README plots from results/*.json."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/screenshots"
STY = {"rtn": ("#ffffff", "-"), "sr": ("#9a9a9a", "--"), "ef": ("#d0d0d0", ":"), "efm": ("#bdbdbd", (0, (5, 1, 1, 1))), "qurl": ("#6a6a6a", "-."), "fp32": ("#ffffff", (0, (2, 2)))}
plt.rcParams.update({"figure.facecolor": "#000", "axes.facecolor": "#0b0b0b", "axes.edgecolor": "#333", "axes.labelcolor": "#bbb",
                     "xtick.color": "#888", "ytick.color": "#888", "text.color": "#ddd", "grid.color": "#1d1d1d", "font.family": "monospace",
                     "font.size": 9, "axes.grid": True, "legend.frameon": False, "savefig.facecolor": "#000"})


def load():
    files = sorted(os.path.basename(f) for f in glob.glob("results/*.json") if not f.endswith("manifest.json"))
    json.dump(files, open("results/manifest.json", "w"), indent=1)
    data = [dict(json.load(open("results/" + f)), file=f) for f in files]
    return [d for d in data if d["file"].startswith("lag_")], [d for d in data if d["file"].startswith("grpo_")]


short = lambda m: m.split("/")[-1].replace("-Instruct", "")


def lag_runs(lag, model, lr):
    return [r for r in lag if r["config"]["model"] == model and r["config"]["lr"] == lr]


def exports(runs, field):
    out = {}
    for r in runs:
        for s in r["steps"]:
            for k, v in s["exports"].items():
                out.setdefault(k, []).append((s["step"], v[field]))
    return out


def last_kl(runs):
    o = {}
    for r in runs:
        ks = [k for k in r["kl"] if k["step"] == r["config"]["steps"]]
        if ks:
            o.update(ks[0])
    return o


def fig_changed(lag, model, lr, path):
    ch = exports(lag_runs(lag, model, lr), "changed_frac")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, g in zip(axes, ["bf16", "int8", "int4"]):
        for k, pts in ch.items():
            if k.startswith(g + "/"):
                c, ls = STY[k.split("/")[1]]
                x, y = zip(*pts[1:])
                ax.plot(x, np.maximum(y, 1e-9), color=c, ls=ls, lw=1.4, label=k.split("/")[1])
        ax.set_yscale("log"); ax.set_title(g, color="#ddd"); ax.set_xlabel("step")
    axes[0].set_ylabel("served weights changed / step"); axes[-1].legend(loc="center right")
    fig.suptitle(f"{short(model)}, AdamW lr {lr:g}: share of served block weights that change at each sync", color="#ccc", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_kl(lag, models, path):
    lrs = sorted({r["config"]["lr"] for r in lag})
    fig, axes = plt.subplots(len(models), 3, figsize=(12, 3.0 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        floor = {}
        for r in lag:
            if r["config"]["model"] == m:
                floor.update(next((k for k in r["kl"] if k["step"] == 0), {}))
        for ax, g in zip(axes[i], ["bf16", "int8", "int4"]):
            rules = ["rtn", "sr", "ef", "efm", "qurl"]
            w = 0.8 / (len(rules) + 1)
            for j, rule in enumerate(rules + ["stuck"]):
                vals = [last_kl(lag_runs(lag, m, lr)).get("stuck" if rule == "stuck" else f"{g}/{rule}", np.nan) for lr in lrs]
                c = "#444" if rule == "stuck" else STY[rule][0]
                ax.bar(np.arange(len(lrs)) + j * w, vals, w, color=c, label=rule, hatch="//" if rule == "qurl" else None, edgecolor="#000")
            if f"{g}/rtn" in floor and floor[f"{g}/rtn"] > 0:
                ax.axhline(floor[f"{g}/rtn"], color="#777", lw=0.8, ls="--", label="step-0 rtn")
            ax.set_yscale("log"); ax.set_xticks(np.arange(len(lrs)) + 0.4 - w / 2); ax.set_xticklabels([f"lr {x:g}" for x in lrs])
            ax.set_title(f"{short(m)} / {g}", color="#ddd")
        axes[i][0].set_ylabel("KL(master || served) / token")
    axes[0][-1].legend(fontsize=7, ncol=2)
    fig.suptitle("served vs master KL at the last step on 200 held-out GSM8K test problems", color="#ccc", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_grpo(grpo, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.4))
    for r in grpo:
        arm = r["config"]["arm"]
        c, ls = STY[arm]
        x = [s["step"] for s in r["steps"]]
        y = np.convolve([s["reward"] for s in r["steps"]], np.ones(3) / 3, mode="valid") if len(x) >= 3 else [s["reward"] for s in r["steps"]]
        a.plot(x[len(x) - len(y):], y, color=c, ls=ls, lw=1.2, label=f"{arm} s{r['config']['seed']}")
        if "changed_frac" in r["steps"][0]:
            b.plot(x, [max(s["changed_frac"], 1e-9) for s in r["steps"]], color=c, ls=ls, lw=1.2)
    a.set_title("GRPO train reward (3-step mean)", color="#ddd"); a.set_xlabel("step"); a.legend(fontsize=6, ncol=3)
    b.set_title("int8 served weights changed / step", color="#ddd"); b.set_yscale("log"); b.set_xlabel("step")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def summary(lag, grpo):
    lines = ["# Results summary (generated by scripts/plots.py from results/*.json)", ""]
    if lag:
        lines += ["## Phase 1: export lag", "", "| model | lr | export | changed / step (mean, steps 2+) | realized at last step | KL at last step |", "|---|---|---|---|---|---|"]
        for r in sorted(lag, key=lambda r: (r["config"]["model"], r["config"]["lr"], r["config"]["variant"])):
            kl = last_kl([r])
            for k in r["steps"][0]["exports"]:
                cf = np.mean([s["exports"][k]["changed_frac"] for s in r["steps"][1:]]) if len(r["steps"]) > 1 else np.nan
                re_ = r["steps"][-1]["exports"][k]["realized"]
                lines.append(f"| {short(r['config']['model'])} | {r['config']['lr']:g} | {k} | {100 * cf:.4f}% | {re_:.3f} | {kl.get(k, float('nan')):.3g} |")
            if "stuck" in kl:
                lines.append(f"| {short(r['config']['model'])} | {r['config']['lr']:g} | stuck ({r['config']['variant']}) | 0 | 0 | {kl['stuck']:.3g} |")
        lines.append("")
    if grpo:
        lines += ["## Phase 2: GRPO, int8 served rollouts", "", "| arm | seed | steps | s/step | eval step | n | acc |", "|---|---|---|---|---|---|---|"]
        for r in sorted(grpo, key=lambda r: (r["config"]["arm"], r["config"]["seed"])):
            per = np.mean([s["t_gen"] + s["t_train"] + s["t_sync"] for s in r["steps"]]) if r["steps"] else np.nan
            for e in r["evals"] or [{"step": "-", "n": "-", "acc": float("nan")}]:
                lines.append(f"| {r['config']['arm']} | {r['config']['seed']} | {len(r['steps'])} | {per:.0f} | {e['step']} | {e['n']} | {100 * e['acc']:.1f}% |")
    open("results/summary.md", "w").write("\n".join(lines) + "\n")


def main():
    lag, grpo = load()
    os.makedirs(OUT, exist_ok=True)
    models = [m for m in ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"] if any(r["config"]["model"] == m for r in lag)]
    for m in models:
        for lr in sorted({r["config"]["lr"] for r in lag if r["config"]["model"] == m}):
            fig_changed(lag, m, lr, f"{OUT}/changed_{short(m)}_lr{lr:g}.png")
    if models:
        fig_kl(lag, models, f"{OUT}/kl.png")
    grpo = [g for g in grpo if g["steps"]]
    if grpo:
        fig_grpo(grpo, f"{OUT}/grpo.png")
    summary(lag, grpo)


if __name__ == "__main__":
    main()

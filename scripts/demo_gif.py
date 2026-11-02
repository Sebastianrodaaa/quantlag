"""Animate one real Phase 1 run from results/*.json into docs/screenshots/demo.gif."""
import io, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, "scripts")
from plots import STY  # noqa: E402  (also applies the monochrome rcParams)

path = sys.argv[1] if len(sys.argv) > 1 else "results/lag_Qwen2.5-0.5B-Instruct_plain_lr1e-06.json"
run = json.load(open(path))
qurl = path.replace("_plain_", "_qurl_")
try:
    extra = json.load(open(qurl))["steps"]
except FileNotFoundError:
    extra = None
grid = "int8"
steps = run["steps"]
for i, s in enumerate(steps):
    if extra and i < len(extra):
        s["exports"].update(extra[i]["exports"])
keys = [k for k in steps[0]["exports"] if k.startswith(grid + "/")]
frames = []
for t in range(1, len(steps) + 1):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [3, 2]})
    for k in keys:
        rule = k.split("/")[1]
        c, ls = STY[rule]
        x = [s["step"] for s in steps[1:t]]
        a.plot(x, [max(s["exports"][k]["changed_frac"], 1e-8) for s in steps[1:t]], color=c, ls=ls, lw=1.5, label=rule)
    a.set_yscale("log"); a.set_xlim(1, len(steps)); a.set_ylim(1e-7, 1); a.set_xlabel("step"); a.set_title("served weights changed at this sync", color="#ddd")
    a.legend(loc="upper right", ncol=4, fontsize=7)
    vals = [steps[t - 1]["exports"][k]["realized"] for k in keys]
    b.bar([k.split("/")[1] for k in keys], vals, color=[STY[k.split("/")[1]][0] for k in keys], edgecolor="#000")
    b.axhline(1, color="#666", lw=0.8, ls="--"); b.set_ylim(0, max(1.2, max(vals) * 1.1)); b.set_title("share of the master update served", color="#ddd")
    m = run["config"]["model"].split("/")[-1]
    fig.suptitle(f"real run  {m}  {grid}  AdamW lr {run['config']['lr']:g}  step {steps[t - 1]['step']}/{len(steps)}", color="#ccc", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, dpi=90); plt.close(fig)
    frames.append(Image.open(buf).convert("L").convert("P"))
frames += [frames[-1]] * 15
frames[0].save("docs/screenshots/demo.gif", save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=True)
print("wrote docs/screenshots/demo.gif", len(frames), "frames")

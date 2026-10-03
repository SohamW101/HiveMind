"""
Presentation graphs for the communication ablation.

    python scripts/make_presentation_graphs.py --eval-dir <dir with *.jsonl> \
        [--comm-log tensorboard_logs/ppo_recurrent_comm_s0_2]

Training curves come from TensorBoard logs (no-comm: tensorboard_logs/ppo_recurrent_30M_1).
The comm curves are drawn only when --comm-log points at an existing run directory.
Evaluation graphs come from JSON-lines episode records (one file per config chunk).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "presentation_graphs")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Fixed identity per arm, never by rank: no-comm = slot 1 blue, comm = slot 2 orange.
ARM = {"nocomm": "#2a78d6", "comm": "#eb6834", "comm_zeroed": "#898781"}
ARM_LABEL = {"nocomm": "No comm", "comm": "Comm", "comm_zeroed": "Comm, messages zeroed"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "font.size": 13, "axes.titlesize": 17, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- training
def tb_series(logdir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]) / 1e6, np.array([e.value for e in ev])


def ema(y, alpha=0.05):
    out, acc = np.empty_like(y), y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def training_compare(nocomm_log, comm_log):
    """Both runs drawn over the same step range: the shorter run's length."""
    runs = [("nocomm", nocomm_log), ("comm", comm_log)]
    xmax = min(tb_series(log, "rollout/ep_len_mean")[0][-1] for _, log in runs)
    specs = [
        ("rollout/success_rate", "Training success rate", "Success rate (%)", 100, "t1_success_rate_compare.png", (0, 105), "lower right"),
        ("rollout/ep_len_mean", "Episode length (makespan) during training", "Steps per episode", 1, "t2_episode_length_compare.png", (0, None), "upper right"),
        ("rollout/ep_rew_mean", "Mean episode reward", "Reward", 1, "t3_episode_reward_compare.png", None, "lower right"),
        ("train/entropy_loss", "Policy entropy (entropy loss)", "Entropy loss", 1, "t4_entropy_compare.png", None, "lower right"),
        ("train/approx_kl", "Approximate KL per update", "Approx. KL", 1, "t5_approx_kl_compare.png", None, "lower right"),
        ("train/explained_variance", "Critic explained variance", "Explained variance", 1, "t6_explained_variance_compare.png", (0.9, 1.002), "lower right"),
        ("train/loss", "PPO total loss", "Loss", 1, "t8_total_loss_compare.png", None, "upper right"),
        ("train/value_loss", "Value (critic) loss", "Value loss (log scale)", 1, "t9_value_loss_compare.png", "log", "upper right"),
        ("train/policy_gradient_loss", "Policy-gradient (clipped surrogate) loss", "Policy-gradient loss", 1, "t10_policy_loss_compare.png", None, "upper right"),
    ]
    for tag, title, ylabel, scale, fname, ylim, loc in specs:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        if ylim == "log":
            ax.set_yscale("log")
            ylim = None
        for arm, log in runs:
            x, y = tb_series(log, tag)
            keep = x <= xmax
            x, y = x[keep], y[keep] * scale
            ax.plot(x, y, color=ARM[arm], lw=0.8, alpha=0.25)
            ax.plot(x, ema(y), color=ARM[arm], lw=2.2, label=ARM_LABEL[arm])
        ax.set_title(title)
        ax.set_xlabel("Training steps (millions)")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, xmax)
        ax.legend(loc=loc)
        save(fig, fname)

    # Curriculum ladder for both runs.
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for arm, log in runs:
        x, y = tb_series(log, "curriculum/target_cartons")
        pts = [(0.0, y[0])] + [(x[i], y[i]) for i in range(1, len(y)) if y[i] != y[i - 1]]
        xs = [p[0] for p in pts] + [xmax]
        ys = [p[1] for p in pts] + [pts[-1][1]]
        ax.step(xs, ys, where="post", color=ARM[arm], lw=2.2, label=ARM_LABEL[arm])
        full = next(p[0] for p in pts if p[1] == 12)
        ax.text(3.2, 10.6 if arm == "nocomm" else 9.6, f"{ARM_LABEL[arm]}: full task at {full:.2f}M",
                color=ARM[arm], fontsize=13, fontweight="bold")
    ax.set_title("Curriculum progression")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Cartons in play")
    ax.set_yticks([1, 2, 4, 8, 12])
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 13.5)
    ax.legend(loc="lower right")
    save(fig, "t7_curriculum_compare.png")


# --------------------------------------------------------------------------- evaluation
def load_eval(eval_dir):
    rows = defaultdict(list)
    for path in glob.glob(os.path.join(eval_dir, "*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            rows[(r["label"], r["cartons"])].append(r)
    for k in rows:
        rows[k].sort(key=lambda r: r["episode"])
    return rows


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summary(rows):
    out = {}
    for (arm, c), rs in rows.items():
        n = len(rs)
        succ = sum(r["is_success"] for r in rs)
        steps = np.array([r["steps"] for r in rs])
        ok = np.array([r["steps"] for r in rs if r["is_success"]]) if succ else np.array([np.nan])
        out[(arm, c)] = {
            "n": n,
            "success": succ,
            "success_rate": succ / n,
            "success_ci": wilson(succ, n),
            "makespan_all": steps.mean(),
            "makespan_success": np.nanmean(ok),
            "makespan_success_median": np.nanmedian(ok),
            "collisions": np.mean([r["collisions"] for r in rs]),
            "invalid": np.mean([r["invalid_actions"] for r in rs]),
            "delivered": np.mean([r["delivered"] for r in rs]),
        }
    return out


def grouped_success(stats, fname):
    cartons = sorted({c for (_, c) in stats})
    arms = [a for a in ("nocomm", "comm") if any((a, c) in stats for c in cartons)]
    x = np.arange(len(cartons))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, arm in enumerate(arms):
        vals = [stats[(arm, c)]["success_rate"] * 100 for c in cartons]
        lo = [max(0.0, v - stats[(arm, c)]["success_ci"][0] * 100) for v, c in zip(vals, cartons)]
        hi = [max(0.0, stats[(arm, c)]["success_ci"][1] * 100 - v) for v, c in zip(vals, cartons)]
        pos = x + (i - 0.5) * (w + 0.02)
        ax.bar(pos, vals, w, color=ARM[arm], label=ARM_LABEL[arm], zorder=2)
        ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        for p_, v, h in zip(pos, vals, hi):
            ax.text(p_, v + h + 2, f"{v:.0f}%", ha="center", va="bottom", color=INK, fontsize=12, fontweight="bold")
    ax.set_xticks(x, [f"{c} cartons" for c in cartons])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Evaluation success rate (100 episodes, 95% CI)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    save(fig, fname)


def makespan_box(rows, cartons, arms, fname, title):
    data, labels, colors = [], [], []
    for arm in arms:
        rs = rows.get((arm, cartons))
        if not rs:
            continue
        data.append([r["steps"] for r in rs])
        labels.append(ARM_LABEL[arm])
        colors.append(ARM[arm])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 2})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(col)
    rng = np.random.default_rng(0)
    for i, (d, col) in enumerate(zip(data, colors), start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=12, color=col, alpha=0.7, zorder=3, lw=0)
        ax.text(i + 0.3, np.median(d), f"median {np.median(d):.0f}", va="center", color=INK2, fontsize=11)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Steps to finish (episode length)")
    ax.set_title(title)
    save(fig, fname)


def metric_bars(stats, cartons, arms, key, ylabel, title, fname, fmt="{:.1f}"):
    arms = [a for a in arms if (a, cartons) in stats]
    vals = [stats[(a, cartons)][key] for a in arms]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar([ARM_LABEL[a] for a in arms], vals, width=0.5, color=[ARM[a] for a in arms], zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02 + 0.01, fmt.format(v), ha="center", va="bottom",
                color=INK, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(max(vals) * 1.2, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, fname)


def entropy_bits(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_info(x, y):
    """I(X;Y) in bits from paired integer samples."""
    xs, ys = np.unique(x), np.unique(y)
    joint = np.zeros((len(xs), len(ys)))
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    for a, b in zip(x, y):
        joint[xi[a], yi[b]] += 1
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def token_analysis(rows, cartons=12):
    res = {}
    for arm in ("comm", "nocomm"):
        rs = rows.get((arm, cartons))
        if not rs:
            continue
        tok = np.concatenate([np.array(r["tokens"]).ravel() for r in rs])
        carry = np.concatenate([np.array(r["carrying"]).ravel() for r in rs])
        agent = np.concatenate([np.tile(np.arange(4), len(r["tokens"])) for r in rs])
        counts = np.bincount(tok, minlength=16)
        res[arm] = {
            "counts": counts,
            "entropy": entropy_bits(counts),
            "tokens_used_90": int(np.searchsorted(np.cumsum(np.sort(counts)[::-1]) / counts.sum(), 0.9) + 1),
            "mi_carry": mutual_info(tok, carry),
            "mi_agent": mutual_info(tok, agent),
            "tok": tok,
            "carry": carry,
        }
    return res


def token_usage_plot(tres, fname):
    if "comm" not in tres:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(16)
    c = tres["comm"]["counts"]
    ax.bar(x, c / c.sum() * 100, 0.7, color=ARM["comm"], zorder=2)
    ax.axhline(100 / 16, color=MUTED, lw=1.2, ls=(0, (3, 3)), zorder=3, label="uniform (6.25% each)")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.13))
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel("Message token")
    ax.set_ylabel("Share of all messages (%)")
    ax.set_title("Comm model: message token usage (12 cartons, 100 episodes)", pad=26)
    ax.text(0, 1.02, f"entropy {tres['comm']['entropy']:.2f} of max 4.00 bits",
            transform=ax.transAxes, color=INK2, fontsize=12, va="bottom")
    ax.grid(axis="x", visible=False)
    save(fig, fname)


def token_by_state_plot(tres, fname):
    if "comm" not in tres:
        return
    tok, carry = tres["comm"]["tok"], tres["comm"]["carry"]
    m = np.zeros((2, 16))
    for s in (0, 1):
        c = np.bincount(tok[carry == s], minlength=16)
        m[s] = c / c.sum() * 100
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(m, aspect="auto", cmap="Blues", vmin=0)
    ax.set_yticks([0, 1], ["searching\n(not carrying)", "carrying"])
    ax.set_xticks(range(16))
    ax.set_xlabel("Message token")
    ax.grid(False)
    for s in (0, 1):
        for t in range(16):
            if m[s, t] >= 5:
                ax.text(t, s, f"{m[s, t]:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if m[s, t] > m.max() * 0.6 else INK)
    ax.set_title("Comm model: token choice vs robot state", pad=26)
    ax.text(0, 1.02, f"mutual information I(token; carrying) = {tres['comm']['mi_carry']:.3f} bits",
            transform=ax.transAxes, color=INK2, fontsize=12, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("% of messages in that state", color=INK2)
    save(fig, fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--nocomm-log", default=os.path.join(REPO, "tensorboard_logs/ppo_recurrent_30M_1"))
    ap.add_argument("--comm-log", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # Training comparison needs both runs' TensorBoard logs.
    if args.comm_log and os.path.isdir(args.comm_log):
        training_compare(args.nocomm_log, args.comm_log)
    else:
        print("comm TensorBoard log not given - skipping training comparison graphs")

    rows = load_eval(args.eval_dir)
    stats = summary(rows)
    tres = token_analysis(rows)
    print("\nSUMMARY")
    for k in sorted(stats, key=lambda k: (k[1], k[0])):
        s = stats[k]
        print(f"{k[0]:12s} c={k[1]:2d} n={s['n']} succ={s['success']:3d} ({s['success_rate']*100:5.1f}%, CI {s['success_ci'][0]*100:.0f}-{s['success_ci'][1]*100:.0f}) "
              f"steps_all={s['makespan_all']:6.1f} steps_succ={s['makespan_success']:6.1f} med={s['makespan_success_median']:6.1f} "
              f"coll={s['collisions']:5.2f} invalid={s['invalid']:6.1f} deliv={s['delivered']:5.2f}")
    for arm, t in tres.items():
        print(f"tokens {arm:7s} entropy={t['entropy']:.2f} bits  top-k for 90%={t['tokens_used_90']}  "
              f"I(tok;carry)={t['mi_carry']:.3f}  I(tok;agent)={t['mi_agent']:.3f}")

    grouped_success(stats, "e1_success_by_cartons.png")
    makespan_box(rows, 12, ["nocomm", "comm"], "e2_makespan_12.png", "Steps to finish, 12 cartons (100 episodes)")
    metric_bars(stats, 12, ["nocomm", "comm"], "collisions", "Collision events per episode",
                "Collisions per episode, 12 cartons", "e3_collisions_12.png")
    makespan_box(rows, 12, ["comm", "comm_zeroed"], "e4_intervention_makespan.png",
                 "Intervention: comm model with its messages zeroed")
    token_usage_plot(tres, "e5_token_usage.png")
    token_by_state_plot(tres, "e6_token_by_state.png")



if __name__ == "__main__":
    main()

"""Generates 1_system_architecture.svg and 2_env_step_flow.svg (render PNGs with any SVG renderer)."""

import os

BG, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#d6d4cc"
STY = {
    "plain": ("#ffffff", LINE), "blue": ("#eef4fc", "#2a78d6"), "orange": ("#fdf0ea", "#eb6834"),
    "dark": ("#1c1c1b", "#1c1c1b"), "grey": ("#f3f2ee", LINE), "green": ("#eaf6ea", "#0ca30c"),
}
FONT = "Liberation Sans, Arial, sans-serif"
HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.o = w, h, []

    def text(self, x, y, s, size=22, weight=400, fill=INK, anchor="middle"):
        self.o.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
                      f'fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>')

    def box(self, x, y, w, h, title, lines=(), kind="plain"):
        fill, stroke = STY[kind]
        self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')
        tc, sc = ("#ffffff", "#c3c2b7") if kind == "dark" else (INK, INK2)
        cy = y + h / 2 - len(lines) * 12 + 8
        self.text(x + w / 2, cy, title, 23, 700, tc)
        for i, ln in enumerate(lines):
            self.text(x + w / 2, cy + 28 + i * 24, ln, 17, 400, sc)

    def frame(self, x, y, w, h, label):
        self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="none" stroke="{LINE}" stroke-width="2" stroke-dasharray="8 8"/>')
        self.text(x + 16, y - 12, label, 20, 700, INK2, "start")

    def arrow(self, pts, color=MUTED, label=None, dash=False):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        da = ' stroke-dasharray="6 5"' if dash else ""
        self.o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"{da} marker-end="url(#ah)"/>')
        if label:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - 10, label, 16, 700, MUTED)

    def save(self, name):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
               f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs><rect width="{self.w}" height="{self.h}" fill="{BG}"/>'
               + "".join(self.o) + "</svg>")
        open(os.path.join(HERE, name), "w").write(svg)


# ---------------------------------------------------------------- 1. system architecture
s = Svg(2000, 900)
s.frame(30, 60, 640, 600, "Simulation (CPU, 8 worker processes)")
s.box(60, 100, 580, 120, "VecMonitor → VecNormalize", ["episode statistics · reward normalisation"], "grey")
s.box(60, 270, 580, 130, "HiveMindSubprocVecEnv", ["1 OS process per world · world → 4 agent slots", "8 worlds = 32 slots, auto-reset per world"], "grey")
s.box(60, 450, 580, 180, "HiveMindMultiAgentEnv  ×8", ["PyBullet world · 13×13 warehouse · 4 robots", "random shelf gaps · 12 cartons · depot",
                                                     "LiDAR · rewards · message channel"], "blue")
s.arrow([(330, 445), (330, 405)])
s.arrow([(330, 265), (330, 225)])
s.arrow([(370, 225), (370, 265)])
s.arrow([(370, 405), (370, 445)])

s.frame(740, 60, 620, 600, "Learning (PyTorch, Stable-Baselines3 / sb3-contrib)")
s.box(770, 100, 560, 170, "RecurrentPPO", ["shared policy for all 4 robots", "HiveMindExtractor → LSTM → π (move, token) / V",
                                             "rollout 512 × 32 · batch 4096 · 10 epochs"], "dark")
s.box(770, 320, 560, 150, "Callbacks", ["CurriculumCallback: 1→2→4→8→12 cartons at 85% success", "MetricsCallback · CheckpointCallback"], "orange")
s.box(770, 520, 560, 110, "scripts/train.py", ["CLI · builds env + model · --communication --curriculum"], "plain")
s.arrow([(640, 140), (765, 140)])
s.arrow([(770, 190), (645, 190)])
s.arrow([(1050, 320), (1050, 275)])
s.arrow([(1050, 515), (1050, 475)])

s.frame(1430, 60, 540, 600, "Outputs and evaluation")
s.box(1460, 100, 480, 110, "models/*.zip", ["periodic checkpoints + final weights"], "green")
s.box(1460, 250, 480, 120, "Evaluation scripts", ["evaluate_ablation · inference · record_demo", "→ JSON lines, MP4 videos"], "plain")
s.box(1460, 410, 480, 90, "make_presentation_graphs", ["training + evaluation + message analysis"], "plain")
s.box(1460, 540, 480, 90, "tensorboard_logs/", ["rollout · train · curriculum · metrics"], "green")
s.arrow([(1330, 155), (1455, 155)])
s.arrow([(1330, 420), (1395, 420), (1395, 585), (1455, 585)])
s.arrow([(1700, 210), (1700, 245)])
s.arrow([(1700, 370), (1700, 405)])
s.arrow([(1880, 540), (1880, 505)])
s.text(1000, 740, "Loop: PPO sends 32 actions (move, token) → 8 workers step physics in parallel → 32 observations (177-d) + rewards return.", 22, 700, INK)
s.text(1000, 780, "Every 16,384 transitions: 10 epochs of minibatch updates. The curriculum callback sets every world's carton count and step cap.", 20, 400, INK2)
s.save("1_system_architecture.svg")

# ---------------------------------------------------------------- 2. one env step
s = Svg(2000, 760)
steps = [
    ("Decode actions", ["(move, token) per robot", "token → one-hot message"], "orange"),
    ("Snap state", ["grid-cell centre · cardinal yaw", "fixed chassis height"], "grey"),
    ("Validate", ["shelf / wall / off-grid → refuse", "pick ≤1.5 cells · drop ≤1.5 of depot"], "blue"),
    ("Physics", ["N interpolated sub-steps", "arm, gripper, carried carton"], "dark"),
    ("Contacts", ["new robot–robot and", "robot–obstacle events"], "grey"),
]
steps2 = [
    ("Termination", ["all cartons delivered →", "terminated; cap → truncated"], "grey"),
    ("Reward", ["0.8·shared + 0.2·individual", "+ 30·(Φ(s′) − Φ(s))"], "green"),
    ("Observation", ["177-d per robot: state, cartons,", "72-ray LiDAR, teammates' tokens"], "blue"),
    ("Return", ["obs (4×177), rewards (4),", "terminated, truncated, info"], "plain"),
]
x0, bw, gap = 40, 340, 50
for i, (t, ln, k) in enumerate(steps):
    x = x0 + i * (bw + gap)
    s.box(x, 110, bw, 150, t, ln, k)
    s.text(x + 22, 140, str(i + 1), 18, 700, "#ffffff" if k == "dark" else MUTED, "start")
    if i:
        s.arrow([(x - gap + 2, 185), (x - 4, 185)])
s.arrow([(x0 + 4 * (bw + gap) + bw / 2, 260), (x0 + 4 * (bw + gap) + bw / 2, 320), (x0 + 3 * (bw + gap) + bw + 30, 320),
         (x0 + 3 * (bw + gap) + bw + 30, 395)])
for i, (t, ln, k) in enumerate(steps2):
    x = x0 + (3 - i) * (bw + gap) + 195
    s.box(x, 400, bw, 150, t, ln, k)
    s.text(x + 22, 430, str(i + 6), 18, 700, MUTED, "start")
    if i:
        s.arrow([(x + bw + gap - 2, 475), (x + bw + 4, 475)])
s.text(40, 70, "HiveMindMultiAgentEnv.step(actions): one decision step for all 4 robots, executed simultaneously", 26, 700, INK, "start")
s.text(40, 640, "Robots may enter each other's cells (charged as a collision, not prevented); invalid actions cost −0.5 and leave the robot in place.", 20, 400, INK2, "start")
s.text(40, 680, "Messages written in step t appear in teammates' observations returned by step t, i.e. they inform actions at t + 1.", 20, 400, INK2, "start")
s.save("2_env_step_flow.svg")
print("ok")

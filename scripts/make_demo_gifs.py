"""Generate reproducible animated GIF demos for the README.

Usage:
    python -m scripts.make_demo_gifs            # all demos
    python -m scripts.make_demo_gifs grid        # one demo
    python -m scripts.make_demo_gifs conformal agent_trace martingale insight

Outputs to docs/img/demo_*.gif.  Requires only matplotlib + Pillow
(no imageio, no playwright).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src.models.score_grid import score_grid, outcome_probs, over_under, btts, top_scorelines

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"

# -- project palette --
SURFACE = "#06080C"
SURFACE_800 = "#0F1318"
SURFACE_700 = "#181D24"
BRAND = "#F5B638"
POS = "#4ADE80"
NEG = "#F87171"
INK_100 = "#E8ECF1"
INK_400 = "#8B95A5"
INK_600 = "#5A6373"
BLUE = "#60A5FA"
PURPLE = "#A78BFA"
ORANGE = "#FB923C"


def _style_ax(ax, title=""):
    ax.set_facecolor(SURFACE_800)
    ax.tick_params(colors=INK_400, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(INK_600)
        spine.set_linewidth(0.5)
    if title:
        ax.set_title(title, color=INK_100, fontsize=9, fontweight="bold", pad=6)


def _save_gif(fig, update_fn, frames, out_name, fps=4, hold_last=8):
    """Render animation to GIF with PillowWriter, holding last frame."""
    total = frames + hold_last

    def wrapped(i):
        return update_fn(min(i, frames - 1))

    anim = FuncAnimation(fig, wrapped, frames=total, interval=1000 // fps, blit=False)
    path = OUT / out_name
    anim.save(str(path), writer=PillowWriter(fps=fps))
    mb = path.stat().st_size / 1024 / 1024
    print(f"  {out_name} ({mb:.1f} MB, {total} frames)")
    plt.close(fig)


# =========================================================================
# 1. Dixon-Coles Grid Morph -- "one grid, all markets"
# =========================================================================

def make_grid_morph():
    print("1. Dixon-Coles grid morph...")

    fig = plt.figure(figsize=(10, 5), facecolor=SURFACE)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.25)

    ax_grid = fig.add_subplot(gs[0])
    ax_markets = fig.add_subplot(gs[1])
    _style_ax(ax_grid, "Dixon-Coles Score Grid")
    _style_ax(ax_markets, "Derived Markets")

    N = 6
    mu_home_seq = np.concatenate([
        np.linspace(0.8, 2.2, 20),
        np.full(15, 2.2),
        np.linspace(2.2, 1.4, 10),
    ])
    mu_away_seq = np.concatenate([
        np.full(20, 1.0),
        np.linspace(1.0, 1.8, 15),
        np.linspace(1.8, 1.4, 10),
    ])
    rho_seq = np.concatenate([
        np.full(20, -0.03),
        np.linspace(-0.03, -0.25, 15),
        np.linspace(-0.25, -0.10, 10),
    ])
    n_frames = len(mu_home_seq)

    im = ax_grid.imshow(
        np.zeros((N, N)), cmap="Blues", vmin=0, vmax=0.16,
        origin="lower", aspect="equal", extent=(-0.5, N - 0.5, -0.5, N - 0.5)
    )
    texts_grid = []
    for i in range(N):
        row = []
        for j in range(N):
            t = ax_grid.text(j, i, "", ha="center", va="center",
                             fontsize=7, color=INK_100, fontweight="bold")
            row.append(t)
        texts_grid.append(row)

    ax_grid.set_xlabel("Away goals", color=INK_400, fontsize=8)
    ax_grid.set_ylabel("Home goals", color=INK_400, fontsize=8)
    ax_grid.set_xticks(range(N))
    ax_grid.set_yticks(range(N))

    params_text = ax_grid.text(
        0.02, 0.97, "", transform=ax_grid.transAxes,
        color=BRAND, fontsize=8, fontweight="bold", va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=INK_600, alpha=0.9)
    )

    bar_labels = ["Home", "Draw", "Away", "O 2.5", "U 2.5", "BTTS Y", "BTTS N"]
    bar_colors = [BRAND, INK_400, NEG, POS, BLUE, ORANGE, PURPLE]
    bars = ax_markets.barh(range(len(bar_labels)), [0]*len(bar_labels), color=bar_colors, height=0.6)
    ax_markets.set_yticks(range(len(bar_labels)))
    ax_markets.set_yticklabels(bar_labels, color=INK_100, fontsize=8)
    ax_markets.set_xlim(0, 0.75)
    ax_markets.set_xlabel("Probability", color=INK_400, fontsize=8)
    ax_markets.invert_yaxis()

    bar_texts = []
    for i in range(len(bar_labels)):
        t = ax_markets.text(0, i, "", va="center", fontsize=7, color=INK_100, fontweight="bold")
        bar_texts.append(t)

    scoreline_text = ax_markets.text(
        0.98, 0.02, "", transform=ax_markets.transAxes, ha="right", va="bottom",
        color=INK_100, fontsize=7, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor=INK_600, alpha=0.9)
    )

    fig.suptitle("One Grid, All Markets", color=INK_100, fontsize=12,
                 fontweight="bold", y=0.97)

    def update(frame):
        mh = mu_home_seq[frame]
        ma = mu_away_seq[frame]
        rho = rho_seq[frame]
        grid = score_grid(mh, ma, rho, max_goals=10)
        small = grid[:N, :N]

        im.set_data(small)
        im.set_clim(0, max(0.01, small.max() * 1.1))

        for i in range(N):
            for j in range(N):
                v = small[i, j]
                texts_grid[i][j].set_text(f"{v:.0%}" if v >= 0.005 else "")
                texts_grid[i][j].set_color("#000000" if v > 0.08 else INK_100)

        params_text.set_text(f"mu_h={mh:.2f}  mu_a={ma:.2f}  rho={rho:+.2f}")

        op = outcome_probs(grid)
        ou = over_under(grid, 2.5)
        bt = btts(grid)
        vals = [op["home"], op["draw"], op["away"], ou["over"], ou["under"], bt["yes"], bt["no"]]

        for bar, val, txt in zip(bars, vals, bar_texts):
            bar.set_width(val)
            txt.set_text(f" {val:.0%}")
            txt.set_x(val + 0.01)

        top = top_scorelines(grid, 3)
        lines = "\n".join(f"  {s['score']}  {s['prob']:.0%}" for s in top)
        scoreline_text.set_text(f"Top scorelines:\n{lines}")

    _save_gif(fig, update, n_frames, "demo_grid_morph.gif", fps=6, hold_last=12)


# =========================================================================
# 2. Conformal Coverage Convergence
# =========================================================================

def make_conformal():
    print("2. Conformal coverage convergence...")

    np.random.seed(42)
    n_matches = 120
    alpha = 0.10
    target = 1.0 - alpha

    probs_all = []
    true_classes = []
    for _ in range(n_matches):
        h = np.random.dirichlet([3, 2, 2])
        probs_all.append(h)
        true_classes.append(np.random.choice(3, p=h))
    probs_all = np.array(probs_all)
    true_classes = np.array(true_classes)

    cal_n = 40
    scores_cal = 1.0 - probs_all[:cal_n, :][np.arange(cal_n), true_classes[:cal_n]]
    import math
    level = math.ceil((cal_n + 1) * target) / cal_n
    q_hat = float(np.quantile(scores_cal, min(level, 1.0), method="higher"))

    fig, (ax_cov, ax_sets) = plt.subplots(2, 1, figsize=(8, 5), facecolor=SURFACE,
                                           gridspec_kw={"height_ratios": [2, 1]})
    _style_ax(ax_cov, "Running Coverage vs 90% Target")
    _style_ax(ax_sets, "Prediction Set Width")

    ax_cov.axhline(target, color=BRAND, linestyle="--", linewidth=1.5, alpha=0.7)
    ax_cov.text(1, target + 0.01, "90% target", color=BRAND, fontsize=7, va="bottom")
    ax_cov.set_xlim(0, n_matches - cal_n)
    ax_cov.set_ylim(0.6, 1.05)
    ax_cov.set_ylabel("Coverage", color=INK_400, fontsize=8)

    line_cov, = ax_cov.plot([], [], color=POS, linewidth=2)
    dot_cov, = ax_cov.plot([], [], "o", color=POS, markersize=5)
    match_text = ax_cov.text(0.98, 0.05, "", transform=ax_cov.transAxes,
                             ha="right", va="bottom", color=INK_100, fontsize=9,
                             fontweight="bold", fontfamily="monospace")

    ax_sets.set_xlim(0, n_matches - cal_n)
    ax_sets.set_ylim(0.5, 3.5)
    ax_sets.set_ylabel("Set size", color=INK_400, fontsize=8)
    ax_sets.set_xlabel("Match #", color=INK_400, fontsize=8)
    ax_sets.set_yticks([1, 2, 3])
    ax_sets.set_yticklabels(["{H}", "{H,D}", "{H,D,A}"], color=INK_400, fontsize=7)

    labels = ["H", "D", "A"]
    test_probs = probs_all[cal_n:]
    test_true = true_classes[cal_n:]
    n_test = len(test_probs)

    coverages = []
    set_sizes = []
    set_colors_list = []
    covered_list = []

    for i in range(n_test):
        row = test_probs[i]
        pred_set = [k for k, p in enumerate(row) if p >= 1.0 - q_hat]
        if not pred_set:
            pred_set = [int(np.argmax(row))]
        covered = test_true[i] in pred_set
        covered_list.append(covered)
        coverages.append(sum(covered_list) / len(covered_list))
        set_sizes.append(len(pred_set))
        set_colors_list.append(POS if covered else NEG)

    fig.suptitle("Conformal Prediction: Distribution-Free Coverage Guarantee",
                 color=INK_100, fontsize=11, fontweight="bold", y=0.97)

    def update(frame):
        idx = frame + 1
        xs = list(range(idx))
        line_cov.set_data(xs, coverages[:idx])
        dot_cov.set_data([xs[-1]], [coverages[idx - 1]])

        cov = coverages[idx - 1]
        color = POS if cov >= target - 0.02 else NEG
        dot_cov.set_color(color)
        line_cov.set_color(color)
        match_text.set_text(f"Match {idx}  Coverage: {cov:.1%}")

        ax_sets.cla()
        _style_ax(ax_sets)
        ax_sets.set_xlim(-0.5, n_test - 0.5)
        ax_sets.set_ylim(0.5, 3.5)
        ax_sets.set_ylabel("Set size", color=INK_400, fontsize=8)
        ax_sets.set_xlabel("Match #", color=INK_400, fontsize=8)
        ax_sets.set_yticks([1, 2, 3])
        ax_sets.set_yticklabels(["{H}", "{H,D}", "{H,D,A}"], color=INK_400, fontsize=7)
        ax_sets.bar(range(idx), set_sizes[:idx], color=set_colors_list[:idx],
                    width=0.8, alpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_gif(fig, update, n_test, "demo_conformal.gif", fps=8, hold_last=16)


# =========================================================================
# 3. Agent Trace -- MCP tool calls lighting up architecture
# =========================================================================

def make_agent_trace():
    print("3. Agent trace animation...")

    tool_calls = [
        {"server": "sports-data", "tool": "get_fixture_context", "latency": 29},
        {"server": "sports-data", "tool": "get_live_odds", "latency": 9},
        {"server": "sports-data", "tool": "get_team_stats", "latency": 0},
        {"server": "sports-data", "tool": "get_squad_props", "latency": 0},
        {"server": "sports-data", "tool": "get_team_stats", "latency": 0},
        {"server": "sports-data", "tool": "get_squad_props", "latency": 0},
        {"server": "news-sentiment", "tool": "get_availability_report", "latency": 3},
        {"server": "news-sentiment", "tool": "analyze_team_sentiment", "latency": 0},
        {"server": "news-sentiment", "tool": "get_availability_report", "latency": 1},
        {"server": "news-sentiment", "tool": "analyze_team_sentiment", "latency": 0},
        {"server": "ml-inference", "tool": "predict_match", "latency": 197},
    ]

    server_positions = {
        "gateway": (0.5, 0.92),
        "orchestrator": (0.5, 0.72),
        "sports-data": (0.15, 0.42),
        "news-sentiment": (0.50, 0.42),
        "ml-inference": (0.85, 0.42),
        "result": (0.5, 0.10),
    }

    server_colors = {
        "gateway": BRAND,
        "orchestrator": BLUE,
        "sports-data": POS,
        "news-sentiment": PURPLE,
        "ml-inference": ORANGE,
        "result": BRAND,
    }

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")

    fig.suptitle("Agent Trace: 11 Tool Calls Across 3 MCP Servers",
                 color=INK_100, fontsize=12, fontweight="bold", y=0.98)

    # Draw static edges
    edge_pairs = [
        ("gateway", "orchestrator"),
        ("orchestrator", "sports-data"),
        ("orchestrator", "news-sentiment"),
        ("orchestrator", "ml-inference"),
        ("orchestrator", "result"),
    ]
    edge_lines = {}
    for a, b in edge_pairs:
        x1, y1 = server_positions[a]
        x2, y2 = server_positions[b]
        line, = ax.plot([x1, x2], [y1, y2], color=INK_600, linewidth=1, alpha=0.3, zorder=1)
        edge_lines[(a, b)] = line

    # Draw static node backgrounds (dim)
    node_patches = {}
    node_labels = {}
    for name, (x, y) in server_positions.items():
        r = 0.08
        circle = mpatches.Circle((x, y), r, facecolor=SURFACE_700, edgecolor=INK_600,
                                 linewidth=1, zorder=2)
        ax.add_patch(circle)
        node_patches[name] = circle
        label = name.replace("-", "\n")
        t = ax.text(x, y, label, ha="center", va="center", color=INK_400,
                    fontsize=7, fontweight="bold", zorder=3)
        node_labels[name] = t

    # Call log area
    log_texts = []
    for i in range(11):
        t = ax.text(0.02, 0.28 - i * 0.025, "", color=INK_400, fontsize=5.5,
                    fontfamily="monospace", va="top", zorder=4)
        log_texts.append(t)

    latency_text = ax.text(0.98, 0.28, "", ha="right", va="top", color=BRAND,
                           fontsize=9, fontweight="bold", fontfamily="monospace", zorder=4)

    query_text = ax.text(0.5, 0.98, "", ha="center", va="top", color=INK_100,
                         fontsize=8, fontfamily="monospace", zorder=4, style="italic")

    n_frames = len(tool_calls) + 5  # extra frames for intro + result

    def update(frame):
        # Phase 1: query arrives (frame 0-1)
        if frame >= 0:
            query_text.set_text('"Predict Arsenal vs Man City"')
            node_patches["gateway"].set_edgecolor(BRAND)
            node_patches["gateway"].set_facecolor(SURFACE_700)
            node_labels["gateway"].set_color(BRAND)

        if frame >= 1:
            node_patches["orchestrator"].set_edgecolor(BLUE)
            node_labels["orchestrator"].set_color(BLUE)
            edge_lines[("gateway", "orchestrator")].set_color(BRAND)
            edge_lines[("gateway", "orchestrator")].set_alpha(0.8)
            edge_lines[("gateway", "orchestrator")].set_linewidth(2)

        # Phase 2: tool calls fire (frames 2 .. 2+len-1)
        call_idx = frame - 2
        if 0 <= call_idx < len(tool_calls):
            tc = tool_calls[call_idx]
            server = tc["server"]

            node_patches[server].set_edgecolor(server_colors[server])
            node_patches[server].set_facecolor(SURFACE_700)
            node_labels[server].set_color(server_colors[server])

            edge_key = ("orchestrator", server)
            if edge_key in edge_lines:
                edge_lines[edge_key].set_color(server_colors[server])
                edge_lines[edge_key].set_alpha(0.9)
                edge_lines[edge_key].set_linewidth(2.5)

            log_texts[call_idx].set_text(
                f"  {call_idx+1:2d}. {server:18s} {tc['tool']:28s} {tc['latency']:3d} ms"
            )
            log_texts[call_idx].set_color(server_colors[server])

            cum_latency = sum(t["latency"] for t in tool_calls[:call_idx + 1])
            latency_text.set_text(f"Total: {cum_latency} ms")

        # Phase 3: result returned
        if call_idx >= len(tool_calls):
            node_patches["result"].set_edgecolor(BRAND)
            node_labels["result"].set_color(BRAND)
            edge_lines[("orchestrator", "result")].set_color(BRAND)
            edge_lines[("orchestrator", "result")].set_alpha(0.9)
            edge_lines[("orchestrator", "result")].set_linewidth(2.5)
            latency_text.set_text("Total: 239 ms  --  DONE")

    _save_gif(fig, update, n_frames, "demo_agent_trace.gif", fps=2, hold_last=6)


# =========================================================================
# 4. Martingale Detonating Against the Guard
# =========================================================================

def make_martingale():
    print("4. Martingale blow-up...")

    np.random.seed(99)
    bankroll_0 = 1000.0
    base_stake = 10.0
    max_stake_frac = 0.05
    max_match_exposure_frac = 0.15

    n_bets = 40
    odds = 2.0
    win_prob = 0.45

    # Simulate unchecked martingale
    bank_unchecked = [bankroll_0]
    stake = base_stake
    for i in range(n_bets):
        won = np.random.random() < win_prob
        if won:
            bank_unchecked.append(bank_unchecked[-1] + stake * (odds - 1))
            stake = base_stake
        else:
            bank_unchecked.append(bank_unchecked[-1] - stake)
            stake *= 2

    # Simulate checked (guarded) martingale
    bank_checked = [bankroll_0]
    stake = base_stake
    rejected = []
    for i in range(n_bets):
        max_allowed = bank_checked[-1] * max_stake_frac
        if stake > max_allowed:
            rejected.append(i)
            clipped_stake = max_allowed
        else:
            clipped_stake = stake

        won = np.random.random() < win_prob
        if won:
            bank_checked.append(bank_checked[-1] + clipped_stake * (odds - 1))
            stake = base_stake
        else:
            bank_checked.append(bank_checked[-1] - clipped_stake)
            stake *= 2

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style_ax(ax, "Martingale Doubling vs Risk Guard")
    ax.set_xlabel("Bet #", color=INK_400, fontsize=8)
    ax.set_ylabel("Bankroll", color=INK_400, fontsize=8)
    ax.set_xlim(0, n_bets)
    y_min = min(min(bank_unchecked), min(bank_checked)) * 0.9
    y_max = max(max(bank_unchecked), max(bank_checked)) * 1.1
    ax.set_ylim(y_min, y_max)
    ax.axhline(bankroll_0, color=INK_600, linestyle=":", linewidth=0.8, alpha=0.5)

    line_unc, = ax.plot([], [], color=NEG, linewidth=2, label="Unchecked martingale")
    line_chk, = ax.plot([], [], color=POS, linewidth=2, label="Risk-guarded")
    ax.legend(loc="upper left", fontsize=7, facecolor=SURFACE_700,
              edgecolor=INK_600, labelcolor=INK_100)

    guard_text = ax.text(0.98, 0.95, "", transform=ax.transAxes,
                         ha="right", va="top", color=BRAND, fontsize=8,
                         fontweight="bold", fontfamily="monospace",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                                   edgecolor=INK_600, alpha=0.9))
    clip_markers = []

    fig.suptitle("Reward-Hacking Defence: Per-Bet Cap + Drawdown Halt",
                 color=INK_100, fontsize=11, fontweight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame):
        idx = frame + 1
        line_unc.set_data(range(idx + 1), bank_unchecked[:idx + 1])
        line_chk.set_data(range(idx + 1), bank_checked[:idx + 1])

        n_clips = sum(1 for r in rejected if r < idx)
        guard_text.set_text(f"Bets clipped: {n_clips}")

        if frame in rejected:
            ax.axvline(frame, color=BRAND, linewidth=0.8, alpha=0.4, linestyle="--")

    _save_gif(fig, update, n_bets, "demo_martingale.gif", fps=4, hold_last=12)


# =========================================================================
# 5. Agent Insight Interaction
# =========================================================================

def make_insight():
    print("5. Agent insight interaction...")

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    conversation = [
        {
            "role": "user",
            "text": "I'm thinking about Arsenal vs Man City this weekend.\nAny value in the home win?",
            "y": 0.92,
        },
        {
            "role": "agent",
            "text": (
                "Let me check... running 11 tool calls across 3 MCP servers.\n\n"
                "  sports-data  ->  get_fixture_context, get_live_odds, get_team_stats\n"
                "  news-sentiment  ->  get_availability_report, analyze_team_sentiment\n"
                "  ml-inference  ->  predict_match"
            ),
            "y": 0.77,
        },
        {
            "role": "agent",
            "text": (
                "Result:  Arsenal 48%  |  Draw 28%  |  Man City 24%\n"
                "xG:  1.44 - 0.81   |   Likely score: 1-0 (15%)\n\n"
                "Conformal set at 90%: {Home, Draw} -- the model\n"
                "cannot separate these two outcomes at this confidence."
            ),
            "y": 0.57,
        },
        {
            "role": "agent",
            "text": (
                "Market odds:  Home 2.10  |  Draw 3.40  |  Away 3.50\n"
                "Model fair:   Home 2.08  |  Draw 3.57  |  Away 4.17\n\n"
                "Edge on Home: +0.9% vs market.  Small but positive.\n"
                "Edge on Away: -16% vs market.  Avoid."
            ),
            "y": 0.37,
        },
        {
            "role": "agent",
            "text": (
                "Recommendation:  The home win has a marginal edge (+0.9%),\n"
                "but the conformal set says {Home, Draw} -- uncertainty is real.\n"
                "Kelly fraction: 0.4% of bankroll.  Proceed only with small stake.\n\n"
                "  1 value suggestion computed -- awaiting human approval."
            ),
            "y": 0.15,
        },
    ]

    role_colors = {"user": BRAND, "agent": POS}
    role_labels = {"user": "YOU", "agent": "AGENT"}

    text_objects = []
    label_objects = []
    for entry in conversation:
        lbl = ax.text(0.02, entry["y"], "", color=role_colors[entry["role"]],
                      fontsize=7, fontweight="bold", va="top", fontfamily="monospace")
        txt = ax.text(0.10, entry["y"] - 0.025, "", color=INK_100,
                      fontsize=6.5, va="top", fontfamily="monospace",
                      linespacing=1.5)
        label_objects.append(lbl)
        text_objects.append(txt)

    fig.suptitle("Agent Insight: Natural-Language Prediction with Evidence",
                 color=INK_100, fontsize=11, fontweight="bold", y=0.98)

    n_frames = len(conversation) * 4 + 2

    def update(frame):
        msg_idx = frame // 4
        char_phase = frame % 4

        for i in range(min(msg_idx, len(conversation))):
            label_objects[i].set_text(role_labels[conversation[i]["role"]])
            text_objects[i].set_text(conversation[i]["text"])

        if msg_idx < len(conversation):
            entry = conversation[msg_idx]
            label_objects[msg_idx].set_text(role_labels[entry["role"]])
            full = entry["text"]
            frac = (char_phase + 1) / 4
            shown = full[:int(len(full) * frac)]
            text_objects[msg_idx].set_text(shown)

    _save_gif(fig, update, n_frames, "demo_insight.gif", fps=3, hold_last=10)


# =========================================================================

DEMOS = {
    "grid": make_grid_morph,
    "conformal": make_conformal,
    "agent_trace": make_agent_trace,
    "martingale": make_martingale,
    "insight": make_insight,
}


def main():
    args = sys.argv[1:]
    if not args:
        targets = list(DEMOS.keys())
    else:
        targets = [a for a in args if a in DEMOS]
        unknown = [a for a in args if a not in DEMOS]
        if unknown:
            print(f"Unknown demos: {unknown}. Available: {list(DEMOS.keys())}")

    for name in targets:
        DEMOS[name]()

    print("\nDone!")


if __name__ == "__main__":
    main()

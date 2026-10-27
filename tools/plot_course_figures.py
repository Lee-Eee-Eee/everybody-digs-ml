"""Create editable course overview figures for the repository README.

These are conceptual schematics rather than measured data. Run from the project root:
    uv run python tools/plot_course_figures.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'course-figures',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
INK = '#172c3c'
MUTED = '#536b78'
TEAL = '#1b7c78'
BORDER = '#9abcb8'


def box(ax, xy, w, h, title, subtitle, fill='#f0f8f6'):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.06',
                                linewidth=1.2, edgecolor=BORDER, facecolor=fill))
    ax.text(x + w / 2, y + h * .64, title, ha='center', va='center',
            color=INK, weight='bold', fontsize=12)
    ax.text(x + w / 2, y + h * .28, subtitle, ha='center', va='center',
            color=MUTED, fontsize=9)


def arrow(ax, start, end, curve='arc3,rad=0', color=TEAL, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11,
                                 connectionstyle=curve, linewidth=1.35, color=color,
                                 linestyle='--' if dashed else '-', shrinkA=0, shrinkB=0))


def export(fig, name):
    svg = OUT / (name + '.svg')
    fig.savefig(svg, bbox_inches='tight', pad_inches=.15, metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    fig.savefig(OUT / (name + '.png'), bbox_inches='tight', pad_inches=.15, dpi=180)
    plt.close(fig)


def roadmap():
    fig, ax = plt.subplots(figsize=(12, 4.9))
    ax.set(xlim=(0, 12), ylim=(0, 4.9)); ax.axis('off')
    ax.text(.15, 4.6, 'A path from numbers to models', fontsize=17, color=INK, weight='bold')
    ax.text(.15, 4.28, 'Follow the shared foundation, then choose a model branch.', fontsize=10, color=MUTED)
    box(ax, (.2, 2.26), 2, .92, 'Python + math', 'values · arrays · shapes')
    box(ax, (2.8, 2.26), 2, .92, 'Learning basics', 'derivatives · MLP · training')
    arrow(ax, (2.2, 2.72), (2.8, 2.72))
    branches = [
        (3.43, 'Classical ML', 'neighbors · margins · clusters'),
        (2.32, 'Vision + sequence', 'CNN · residual · RNN / LSTM'),
        (1.21, 'Language models', 'attention · Transformer · GPT'),
        (.10, 'Other networks', 'graphs · liquid · diffusion'),
    ]
    for y, title, subtitle in branches:
        box(ax, (5.58, y), 3.0, .82, title, subtitle, fill='#f5f8fb')
        arrow(ax, (4.8, 2.72), (5.58, y + .41), color='#6c8c98')
    box(ax, (9.42, 2.26), 2.33, .92, 'Experiments', 'baselines · tests · projects', fill='#fbf6ef')
    for target_y, (y, _, _) in zip((3.02, 2.82, 2.62, 2.42), branches):
        arrow(ax, (8.58, y + .41), (9.42, target_y), color='#6c8c98')
    export(fig, 'course-roadmap')


def experiment():
    fig, ax = plt.subplots(figsize=(12, 3.85))
    ax.set(xlim=(0, 12), ylim=(0, 3.85)); ax.axis('off')
    ax.text(.15, 3.55, 'One experiment, three data roles', fontsize=17, color=INK, weight='bold')
    ax.text(.15, 3.23, 'Fit on training data; make choices on validation; evaluate the locked choice on test.',
            fontsize=10, color=MUTED)
    box(ax, (.15, 1.64), 2.1, .93, 'Training', 'fit scaling + model')
    box(ax, (3.05, 1.64), 2.1, .93, 'Validation', 'compare and select', fill='#f5f8fb')
    box(ax, (5.95, 1.64), 2.1, .93, 'Locked choice', 'model + rule', fill='#f5f8fb')
    box(ax, (8.85, 1.64), 2.8, .93, 'Held-out test', 'report final performance', fill='#fbf6ef')
    for a, b in [(2.25, 3.05), (5.15, 5.95), (8.05, 8.85)]:
        arrow(ax, (a, 2.105), (b, 2.105))
    ax.text(1.2, .79, 'Fit parameters', ha='center', fontsize=10, color=MUTED)
    ax.text(4.1, .79, 'Tune choices', ha='center', fontsize=10, color=MUTED)
    ax.text(7, .79, 'Freeze', ha='center', fontsize=10, color=MUTED)
    ax.text(10.25, .79, 'Evaluate', ha='center', fontsize=10, color=MUTED)
    export(fig, 'evaluation-workflow')


if __name__ == '__main__':
    roadmap()
    experiment()

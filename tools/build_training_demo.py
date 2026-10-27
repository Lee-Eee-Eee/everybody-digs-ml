"""Create an honest, reproducible XOR training animation for the README.

A two-layer NumPy network trains on fixed synthetic points using full-batch
binary cross-entropy. Each displayed boundary and loss is from the same step.
Run: uv run python tools/build_training_demo.py
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets' / 'training-demo.gif'
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': '#fbfcfb', 'axes.facecolor': '#ffffff',
})


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def main():
    rng = np.random.default_rng(8)
    centers = np.array([[-.8, -.8], [-.8, .8], [.8, -.8], [.8, .8]])
    x = np.concatenate([rng.normal(center, .22, size=(35, 2)) for center in centers])
    y = np.repeat([0, 1, 1, 0], 35)[:, None]
    w1 = rng.normal(0, .7, (2, 8)); b1 = np.zeros((1, 8))
    w2 = rng.normal(0, .5, (8, 1)); b2 = np.zeros((1, 1))
    grid = np.linspace(-1.55, 1.55, 100)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    snapshots, loss_history = [], []
    steps = 360
    for step in range(steps + 1):
        hidden = np.tanh(x @ w1 + b1)
        p = sigmoid(hidden @ w2 + b2)
        loss = float(-np.mean(y*np.log(p + 1e-12) + (1-y)*np.log(1-p + 1e-12)))
        loss_history.append(loss)
        if step % 10 == 0:
            scores = sigmoid(np.tanh(points @ w1 + b1) @ w2 + b2).reshape(xx.shape)
            snapshots.append((step, scores.copy(), float(np.mean((p >= .5) == y))))
        if step == steps:
            break
        dlogit = (p - y) / len(y)
        dw2 = hidden.T @ dlogit; db2 = dlogit.sum(axis=0, keepdims=True)
        dh = (dlogit @ w2.T) * (1 - hidden**2)
        dw1 = x.T @ dh; db1 = dh.sum(axis=0, keepdims=True)
        w1 -= .8*dw1; b1 -= .8*db1; w2 -= .8*dw2; b2 -= .8*db2
    assert loss_history[-1] < loss_history[0] and snapshots[-1][2] > .95
    cmap = LinearSegmentedColormap.from_list('decision', ['#dcecf5', '#fffdf9', '#fae5db'])
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.1, 4.0), dpi=100,
                                      gridspec_kw={'width_ratios': [1.13, 1]})
    fig.subplots_adjust(left=.07, right=.97, bottom=.19, top=.79, wspace=.33)
    fig.text(.07, .93, 'Can a network learn XOR?', fontsize=20, fontweight='bold', color='#17314a')
    fig.text(.07, .865, 'One tiny network  /  actual training  /  synthetic data', fontsize=10.5, color='#55717d')
    view = left.imshow(snapshots[0][1], origin='lower', extent=(-1.55,1.55,-1.55,1.55),
                       cmap=cmap, vmin=0, vmax=1, interpolation='bilinear', aspect='equal')
    for label, color, marker in [(0, '#24627c', 'o'), (1, '#ad4f32', 's')]:
        subset = x[y[:, 0] == label]
        left.scatter(subset[:, 0], subset[:, 1], c=color, marker=marker, s=15,
                     edgecolors='white', linewidths=.3, alpha=.83, label=f'Class {label}')
    left.set(xlabel='Feature 1', ylabel='Feature 2', title='Decision region', xlim=(-1.55,1.55), ylim=(-1.55,1.55))
    left.legend(loc='upper left', frameon=True, framealpha=.94, fontsize=8)
    right.set(xlim=(0,steps), ylim=(0,max(.8,loss_history[0]*1.1)), xlabel='Training update',
              ylabel='Binary cross-entropy', title='Training loss')
    right.grid(axis='y', color='#dce5e6', linewidth=.7)
    curve, = right.plot([], [], color='#137d79', linewidth=2.3)
    dot, = right.plot([], [], marker='o', color='#137d79', markersize=5)
    status = fig.text(.76, .045, '', ha='center', color='#263e4d', fontsize=9.5)
    fig.text(.07, .045, '2 inputs  →  8 tanh neurons  →  1 sigmoid output', color='#55717d', fontsize=9.5)
    for ax in (left,right):
        for label in ax.get_xticklabels()+ax.get_yticklabels(): label.set_fontfamily('Times New Roman')
    fig.canvas.draw()
    frames = []
    for step, scores, acc in snapshots:
        view.set_data(scores)
        curve.set_data(np.arange(step + 1), loss_history[:step + 1])
        dot.set_data([step], [loss_history[step]])
        status.set_text(f'Update {step:3d}   ·   loss {loss_history[step]:.3f}   ·   train accuracy {acc:.0%}')
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(image.copy()).convert('RGB').convert('P', palette=Image.Palette.ADAPTIVE, colors=128))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    durations = [120] * len(frames); durations[0] = 700; durations[-1] = 1500
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    plt.close(fig)
    print(f'{OUT}: {len(frames)} frames; loss {loss_history[0]:.4f} -> {loss_history[-1]:.4f}; '
          f'train accuracy {snapshots[0][2]:.3f} -> {snapshots[-1][2]:.3f}')


if __name__ == '__main__':
    main()

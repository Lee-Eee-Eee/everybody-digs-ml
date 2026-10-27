"""Export selected saved notebook figures for the README preview."""
from pathlib import Path
import base64
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets'
FIGURES = {
    'preview-knn.png': ('lessons/classical_ml/20_classical_classification.ipynb', 'toy_distances ='),
    'preview-resnet.png': ('lessons/computer_vision/23_resnet_from_blocks.ipynb', "ax.set_title('Small post-activation residual block'"),
    'preview-diffusion.png': ('lessons/other_networks/22_diffusion_from_noise.ipynb', 'image_noise = torch.randn_like(image)'),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (path, marker) in FIGURES.items():
        notebook = nbformat.read(ROOT / path, 4)
        cells = [cell for cell in notebook.cells if cell.cell_type == 'code' and marker in cell.source]
        if len(cells) != 1:
            raise ValueError(f'Expected exactly one cell containing {marker!r} in {path}')
        images = [out.data['image/png'] for out in cells[0].outputs if 'image/png' in out.get('data', {})]
        if len(images) != 1:
            raise ValueError(f'Expected exactly one saved PNG in {path}: {marker!r}')
        (OUT / name).write_bytes(base64.b64decode(images[0]))
        print(name, '<-', path)


if __name__ == '__main__':
    main()

import numpy as np


def plot_imagine(p, mapping, file):
    p = (p >= 0.5).astype(np.uint8)
    p = p.reshape(-1, 28, 28)
    for mk in mapping.argsort():
        print(f"class: {int(mapping[mk])}", file=file)
        image = p[mk]
        print(*[str(row)[1:-1] for row in image], sep='\n', file=file)
        print(file=file)
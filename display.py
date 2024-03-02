# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from typing import List, Tuple


def plot(*series: List[Tuple[float, float]]):

    for serie in series:
        xs = [x[0] for x in serie]
        ys = [x[1] for x in serie]
        plt.plot(xs, ys)

    plt.show()


def display(images: List[npt.ArrayLike], legends=None):
    nb_plots = len(images) // 5
    fig, axs = plt.subplots(nb_plots, 5, figsize=(12, 12))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < len(images):
            ax.imshow(
                images[i].astype(np.float64),
                cmap='gray',
                norm=NoNorm()
            )

            if legends and len(legends) > i:
                ax.set_title(legends[i])

    plt.tight_layout()
    plt.show()


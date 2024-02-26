# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

def display(images, legends=None):
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


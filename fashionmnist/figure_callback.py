from typing import List
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Layer
from skimage.transform import rescale, resize
from anisotropic_diffusion import anisotropic_diffusion_


class FigureCallback(Callback):
    """callback that generates plt figures and saves in the logdir

    Args:
        layer_names (str): sub_dir to put the figures in
        path (Path): log directory to save figures under
    """

    def __init__(self, layer_names: List[str], path):
        self.layer_names = layer_names
        self.path = path
        self.filter_test_image = False

    def on_epoch_end(self, epoch: int, logs: dict):
        """at the end of each epoch, create and write the figure

        Args:
            epoch (int): epoch number
            logs (dict): log dict with loss, original, and reconstructed batches.
        """
        loss = logs["loss"]
        original = logs["original"]
        reconstructed = logs["reconstructed"]

        col_count = min(original.shape[0], 10)
        figure, axes = plt.subplots(3, col_count, figsize=(15, 5))

        # these are the layers to show histograms
        for bins_col in range(len(self.layer_names)):

            # get the layer
            layer = self.model.get_layer(self.layer_names[bins_col])

            # set the title for the plot
            axes[0][bins_col].set_title(layer.name)

            # extract the kernels weights
            kernel_weights = [
                kernel
                for kernel in layer.trainable_variables
                if "kernel" in kernel.name
            ][0]

            axes[0][bins_col].hist(kernel_weights.numpy().reshape(-1))

        quantiles = [0.05, 0.95]
        for n in range(col_count):
            q_test = np.quantile(original[n], quantiles)
            axes[1][n].imshow(original[n], cmap="gray", vmin=q_test[0], vmax=q_test[1])

            reshape_test_image = np.reshape(reconstructed[n], (64, 64))
            reshape_test_image = resize(
                reshape_test_image, (256, 256), order=3, anti_aliasing=True
            )

            q_re = np.quantile(reshape_test_image, quantiles)

            if self.filter_test_image:
                reshape_test_image -= q_re[0]
                reshape_test_image /= q_re[1] - q_re[0]

                filtered_test_image = anisotropic_diffusion_(
                    reshape_test_image, niter=20, kappa=0.1, option=3, gradient_scale=8
                )
                axes[2][n].imshow(filtered_test_image, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                axes[2][n].imshow(
                    reshape_test_image, cmap="gray", vmin=q_re[0], vmax=q_re[1]
                )

        figure.suptitle(f"Epoch {epoch}: loss={loss:0.2f}")

        # make the save path, if needed
        self.path.mkdir(parents=True, exist_ok=True)

        # save and close
        figure.savefig(self.path / f"{epoch}-figure.png")
        plt.close(figure)

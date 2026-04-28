"""implements the OrientedPowerMap2D keras layer
"""
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


def complex_exp(
    x_grid: np.ndarray, y_grid: np.ndarray, freq: float, angle_rad: float
) -> np.ndarray:
    """compute complex exponential on the grid for given frequency and angle

    Args:
        x_grid (np.ndarray): the x coordinates on the grid
        y_grid (np.ndarray): the y coordinates on the grid
        freq (float): frequency of the exponential
        angle_rad (float): direction of the exponential, in radians

    Returns:
        np.ndarray: complex exponential at the grid points
    """
    return np.exp(
        freq * (x_grid * np.sin(angle_rad) + y_grid * np.cos(angle_rad)) * 1.0j
    )


def gauss(x_grid: np.ndarray, y_grid: np.ndarray, sigma: float) -> np.ndarray:
    """computed the gaussian on the given grid

    Args:
        x_grid (np.ndarray): the x coordinates on the grid
        y_grid (np.ndarray): the y coordinates on the grid
        sigma (float): gaussian sigma

    Returns:
        np.ndarray: gaussian at the grid points
    """
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -(x_grid * x_grid + y_grid * y_grid) / (2.0 * sigma * sigma)
    )


def make_meshgrid(size: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """makes a mesh centered at 0,0 of the given size

    Args:
        size (int, optional): size of the grid. Defaults to 9.

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple of x- and y- grids
    """
    return np.meshgrid(
        np.linspace(-(size // 2), size // 2, size),
        np.linspace(-(size // 2), size // 2, size),
    )


def kernels2tensor(
    kernels: List[np.ndarray], channels: int, dtype=tf.float32
) -> tf.Tensor:
    """turns list of numpy arrays to a tensor of given typeS

    Args:
        kernels (List[np.ndarray]): list of kernels to be turned to tensor
        channels (int): input channels, supported by repeating
        dtype ([type], optional): type of output tensor. Defaults to tf.float32.

    Returns:
        tf.Tensor: the tensor formed from the kernels.  axis for kernels is last
    """

    kernels = np.array(kernels)
    kernels = np.expand_dims(kernels, axis=-1)
    kernels = np.repeat(kernels, channels, axis=0)
    kernels = np.repeat(kernels, channels, axis=-1)
    kernels = np.moveaxis(kernels, 0, -1)
    return tf.constant(kernels, dtype=dtype)


def make_gabor_kernels(
    x_grid, y_grid, in_channels, directions=3, freqs=[2.0, 1.0]
) -> tf.Tensor:
    """makes a bank of gabor kernels as a complex tensor

    Args:
        x_grid ([type]): [description]
        y_grid ([type]): [description]
        in_channels (int):
        directions (int, optional): [description]. Defaults to 3.
        freqs (list, optional): [description]. Defaults to [2.0, 1.0].

    Returns:
        tf.Tensor: complex tensor with a kernel on each channel
    """

    angles_rad = [n * np.pi / float(directions) for n in range(directions)]
    sine_kernels = kernels2tensor(
        [
            complex_exp(x_grid, y_grid, freq, angle_rad)
            for freq in freqs
            for angle_rad in angles_rad
        ],
        channels=in_channels,
    )
    sigmas = [2.0 / freq for freq in freqs]
    gauss_kernels = kernels2tensor(
        [gauss(x_grid, y_grid, sigma) for sigma in sigmas], channels=in_channels
    )
    gauss_kernels = np.repeat(
        gauss_kernels, sine_kernels.shape[-1] // gauss_kernels.shape[-1], axis=-1
    )

    bank = gauss_kernels * sine_kernels
    g0 = kernels2tensor([gauss(x_grid, y_grid, 4.0 / freqs[-1])], channels=in_channels)
    return tf.concat([bank, g0], -1)


class OrientedPowerMap2D(Layer):
    """creates a stacked gabor filter bank that is non-trainable

    Args:
        directions (int, optional): [description]. Defaults to 3.
        freqs (list, optional): [description]. Defaults to [2.0, 1.0].
        size (int, optional): [description]. Defaults to 13.
    """

    def __init__(self, directions=3, freqs=[2.0, 1.0], size=13, **kwargs):

        super().__init__(trainable=False, activity_regularizer=None, **kwargs)
        self.directions = directions
        self.freqs = freqs
        self.size = size

    def build(self, input_shape):
        """[summary]

        Args:
            input_shape ([type]): [description]
        """

        # computer gabor filter bank
        x_grid, y_grid = make_meshgrid(size=self.size)
        kernels = make_gabor_kernels(
            x_grid,
            y_grid,
            in_channels=input_shape[-1],
            directions=self.directions,
            freqs=self.freqs,
        )
        self._real_kernels = tf.math.real(kernels)
        self._imag_kernels = tf.math.imag(kernels)

    def call(self, inputs):
        """[summary]

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        response = (
            tf.nn.conv2d(inputs, self._real_kernels, strides=1, padding="SAME") ** 2
            + tf.nn.conv2d(inputs, self._imag_kernels, strides=1, padding="SAME") ** 2
        )
        return response


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for_dir = 5
    for_freq = [2.0, 1.0]
    _x_grid, _y_grid = make_meshgrid(size=9)
    gabor_kernels = make_gabor_kernels(
        _x_grid, _y_grid, 6, directions=for_dir, freqs=for_freq
    )

    _, axs = plt.subplots(
        len(for_freq) + 1, for_dir, figsize=(for_dir * 3, len(for_freq) * 3)
    )
    for n in range(for_dir):
        for m in range(len(for_freq)):
            img = tf.squeeze(gabor_kernels[..., 0, m * for_dir + n])
            axs[m][n].imshow(tf.math.real(img), cmap="plasma")

    _g0 = tf.squeeze(gabor_kernels[..., 0, -1])
    for n in range(for_dir):
        axs[len(for_freq)][n].imshow(tf.squeeze(_g0))

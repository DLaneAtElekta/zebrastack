"""implementation of SoftMax pooling layer
"""
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K
from tensor_utils import (
    normalize_data_format,
    normalize_padding,
    normalize_tuple,
    convert_data_format,
    conv_output_length,
)


def logsumexp_pool(
    value,
    ksize,
    strides=[1, 2, 2, 1],
    padding="SAME",
    data_format="NHWC",
    scale_up=1e2,
    name=None,
):
    """function to calculate the log(sum(exp(x))) function,
    which is a continuous approximation to the max function

    Args:
        value ([type]): [description]
        ksize ([type]): [description]
        strides (list, optional): [description]. Defaults to [1, 2, 2, 1].
        padding (str, optional): [description]. Defaults to "SAME".
        data_format (str, optional): [description]. Defaults to "NHWC".
        scale_up ([type], optional): [description]. Defaults to 1e2.
        name ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    assert len(value.shape) == 4
    value = tf.transpose(value, perm=[3, 1, 2, 0])
    scaled = tf.scalar_mul(scale_up, value)
    patched_response = tf.image.extract_patches(
        scaled,
        sizes=[1, 2, 2, 1],  # TODO: pass these in???
        strides=strides,
        rates=[1, 1, 1, 1],
        padding=padding,
    )
    logsumexp_result = tf.math.reduce_logsumexp(patched_response, axis=-1)
    descaled = tf.scalar_mul(1.0 / scale_up, logsumexp_result)
    return tf.transpose(tf.expand_dims(descaled, axis=-1), perm=[3, 1, 2, 0])


class LogSumExpPooling2D(Layer):
    """SoftMax pooling layer for performing continuous approximation of max

    Args:
      pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
        specifying the size of the pooling window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the pooling operation.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.
      name: A string, the name of the layer.
    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        data_format=None,
        name=None,
        **kwargs
    ):
        super(LogSumExpPooling2D, self).__init__(name=name, **kwargs)
        if data_format is None:
            data_format = K.image_data_format()
        if strides is None:
            strides = pool_size
        self.pool_size = normalize_tuple(pool_size, 2, "pool_size")
        self.strides = normalize_tuple(strides, 2, "strides")
        self.padding = normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """evalute the layer for given input

        Args:
            inputs (tf.Tensor):
                input tensor to be evaluated
                expected 4-dimensional

        Returns:
            tf.Tensor: result of applying the softmax
        """
        if len(inputs.shape) != 4:
            raise ValueError("expecting 4-dimensional tensor")

        if self.data_format == "channels_last":
            pool_shape = (1,) + self.pool_size + (1,)
            strides = (1,) + self.strides + (1,)
        else:
            pool_shape = (1, 1) + self.pool_size
            strides = (1, 1) + self.strides
        outputs = logsumexp_pool(
            inputs,
            ksize=pool_shape,
            strides=strides,
            padding=self.padding.upper(),
            data_format=convert_data_format(self.data_format, 4),
        )
        return outputs

    def compute_output_shape(self, input_shape):
        """[summary]

        Args:
            input_shape ([type]): [description]

        Returns:
            [type]: [description]
        """
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_first":
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_output_length(
            rows, self.pool_size[0], self.padding, self.strides[0]
        )
        cols = conv_output_length(
            cols, self.pool_size[1], self.padding, self.strides[1]
        )
        if self.data_format == "channels_first":
            return tf.TensorShape([input_shape[0], input_shape[1], rows, cols])
        return tf.TensorShape([input_shape[0], rows, cols, input_shape[3]])

    def get_config(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        config = {
            "pool_size": self.pool_size,
            "padding": self.padding,
            "strides": self.strides,
            "data_format": self.data_format,
        }
        base_config = super(LogSumExpPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    logging.basicConfig()

    input_tensor = tf.constant(np.zeros((1, 4, 4, 1)))

    softmax_layer = LogSumExpPooling2D()
    result = softmax_layer(input_tensor)

    logging.info(result)

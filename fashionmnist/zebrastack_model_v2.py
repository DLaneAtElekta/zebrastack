"""zebrastack_model_v2
"""
import logging
from time import time
from pathlib import Path
from typing import Optional

from autologging import logged
import numpy as np
from skimage.transform import rescale
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    SpatialDropout2D,
    ZeroPadding2D,
    LocallyConnected2D,
    Dense,
    Flatten,
    Reshape,
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Sequential

from oriented_powermap_2d import OrientedPowerMap2D
from logsumexp_pooling_2d import LogSumExpPooling2D
from figure_callback import FigureCallback
from tensor_utils import configure_logger, generate_batches


@logged
def prepare_images(img_array: np.ndarray, size=64):
    """prepares images for training: whitening and reshaping

    Args:
        img_array (np.ndarray): the input image as a numpy array
        sz (int, optional): [description]. Defaults to 64.

    Returns:
        np.ndarray: updated 4-d array
    """
    assert len(img_array.shape) == 3
    img_array = (img_array / 255.0).astype(np.float32)
    # prepare_images._log.info(f"img_array: {img_array.shape} {img_array.dtype}")
    img_array = [
        rescale(img_array[n], size / img_array.shape[-1], order=3)
        for n in range(img_array.shape[0])
    ]
    img_array = np.reshape(img_array, (len(img_array), size, size, 1))
    # prepare_images._log.info(f"img_array: {img_array.shape} {img_array.dtype}")
    return img_array


def create_encoder_v1(
    size=64,
    latent_dim=8,
    locally_connected_channels=2,
    act_func="softplus",
):
    """creates the encoder side of the autoencoder, mapping to latent_dim gaussian

    Args:
        size (int): size x size input
        latent_dim (int): gaussian dimensions
        locally_connected_channels = 2

        size (int, optional):
            input is size x size. Defaults to 64.
        latent_dim (int, optional):
            dimension of gaussian blob. Defaults to 8.
        locally_connected_channels (int, optional):
            channels on locally connected layer. Defaults to 2.
        act_func (str, optional):
            activation function for most layers. Defaults to "softplus".

    Returns:
        Sequential: encoder model
    """

    return Sequential(
        [
            Input(shape=(size, size, 1), name="retina_{}".format(size)),
            ####
            #### V1 layers
            Conv2D(16, (5, 5), name="v1_conv2d", activation=act_func, padding="same"),
            MaxPooling2D((2, 2), name="v1_maxpool", padding="same"),
            SpatialDropout2D(0.1, name="v1_dropout"),
            ####
            #### V2 layers
            Conv2D(16, (3, 3), name="v2_conv2d", activation=act_func, padding="same"),
            MaxPooling2D((2, 2), name="v2_maxpool", padding="same"),
            ####
            #### V4 layers
            Conv2D(32, (3, 3), name="v4_conv2d", activation=act_func, padding="same"),
            MaxPooling2D((2, 2), name="v4_maxpool", padding="same"),
            ####
            #### IT Layers
            Conv2D(32, (3, 3), name="pit_conv2d", activation=act_func, padding="same"),
            Conv2D(64, (3, 3), name="cit_conv2d", activation=act_func, padding="same"),
            LocallyConnected2D(
                locally_connected_channels,
                (3, 3),
                name="ait_local",
                activation=act_func,
                kernel_regularizer=l1_l2(0.5, 0.5),
            ),
            ####
            #### VLPFC
            # generate latent vector Q(z|X)
            Flatten(name="vlpfc_flatten"),
            Dense(latent_dim, name="vlpfc_dense", activation=act_func),
            Dense(latent_dim + latent_dim, name="z_mean_log_var"),
        ],
        name="v1_to_vlpfc_encoder",
    )


def create_encoder_v2(
    size=64,
    latent_dim=8,
    locally_connected_channels=3,
    act_func="softplus",
):
    """creates the encoder side of the autoencoder, mapping to latent_dim gaussian
    V2 replaces the Conv2D layers with gabor powermaps

    Args:
        size (int): size x size input
        latent_dim (int): gaussian dimensions
        locally_connected_channels = 2

        size (int, optional):
            input is size x size. Defaults to 64.
        latent_dim (int, optional):
            dimension of gaussian blob. Defaults to 8.
        locally_connected_channels (int, optional):
            channels on locally connected layer. Defaults to 2.
        act_func (str, optional):
            activation function for most layers. Defaults to "softplus".

    Returns:
        Sequential: encoder model
    """

    return Sequential(
        [
            Input(shape=(size, size, 1), name="retina_{}".format(size)),
            ####
            #### V1 layers
            OrientedPowerMap2D(
                directions=7,
                freqs=[2.0, 1.0, 0.5, 0.25],
                size=9,
                name="v1_powmap",
            ),
            MaxPooling2D((2, 2), name="v1_maxpool", padding="same"),
            ####
            #### V2 layers
            OrientedPowerMap2D(
                directions=7,
                freqs=[2.0, 1.0, 0.5, 0.25],
                size=9,
                name="v2_powmap",
            ),
            MaxPooling2D((2, 2), name="v2_maxpool", padding="same"),
            # dimensional reduction
            Conv2D(16, (1, 1), name="v2_reduce", activation=act_func, padding="same"),
            ####
            #### V4 layers
            OrientedPowerMap2D(
                directions=7,
                freqs=[2.0, 1.0, 0.5, 0.25],
                size=9,
                name="v4_powmap",
            ),
            MaxPooling2D((2, 2), name="v4_maxpool", padding="same"),
            Conv2D(16, (1, 1), name="v4_reduce", activation=act_func, padding="same"),
            ####
            #### IT Layers
            Conv2D(16, (3, 3), name="pit_conv2d", activation=act_func, padding="same"),
            Conv2D(8, (3, 3), name="cit_conv2d", activation=act_func, padding="same"),
            LocallyConnected2D(
                locally_connected_channels,
                (3, 3),
                name="ait_local",
                activation=act_func,
                kernel_regularizer=l1_l2(l1=0.05, l2=0.05),
            ),
            ####
            #### VLPFC
            # generate latent vector Q(z|X)
            Flatten(name="vlpfc_flatten"),
            Dense(latent_dim, name="vlpfc_dense", activation=act_func),
            Dense(latent_dim + latent_dim, name="z_mean_log_var"),
        ],
        name="v1_to_vlpfc_encoder",
    )


def create_decoder(
    dense_shape, latent_dim=8, locally_connected_channels=3, act_func="softplus"
):
    """creates the decoder side of the autoencoder, given the input shape
    Args:
        dense_shape (tuple): shape to be used for dense layer
        latent_dim (int, optional):
            latent dimension of input. Defaults to 8.
        locally_connected_channels (int, optional):
            number of channels for locally connected layer. Defaults to 2.
        act_func (str, optional):
            activation function to be used. Defaults to 'softplus'.

    Returns:
        Sequential: keras model for decoder
    """
    return Sequential(
        [
            Input(shape=(latent_dim,), name="z_sampling"),
            Dense(
                dense_shape[1] * dense_shape[2] * dense_shape[3],
                name="vlpfc_dense_back",
                activation=act_func,
            ),
            Reshape(
                (dense_shape[1], dense_shape[2], dense_shape[3]),
                name="vlpfc_antiflatten",
            ),
            ####
            #### IT retro Layers
            ZeroPadding2D(padding=(1, 1), name="ait_padding_back"),
            LocallyConnected2D(
                locally_connected_channels,
                (3, 3),
                name="ait_local_back",
                activation=act_func,
                kernel_regularizer=l1_l2(0.5, 0.5),
            ),
            ZeroPadding2D(padding=(1, 1), name="cit_padding_back"),
            Conv2DTranspose(
                8, (3, 3), name="cit_conv2d_trans", activation=act_func, padding="same"
            ),
            Conv2DTranspose(
                16, (3, 3), name="pit_conv2d_trans", activation=act_func, padding="same"
            ),
            ####
            #### V4 retro layers
            Conv2DTranspose(
                16, (3, 3), name="v4_conv2d_trans", activation=act_func, padding="same"
            ),
            UpSampling2D((2, 2), name="v4_upsample_back"),
            ####
            #### V2 retro layers
            Conv2DTranspose(
                16, (3, 3), name="v2_conv2d_trans", activation=act_func, padding="same"
            ),
            UpSampling2D((2, 2), name="v2_upsample_back"),
            ####
            #### V1 retro layers
            Conv2D(
                1,
                (5, 5),
                name="v1_conv2d_5x5_back",
                # activation='sigmoid', no sigmoid == return logits
                padding="same",
            ),
            UpSampling2D((2, 2), name="v1_upsample_back"),
        ],
        name="pulvinar_to_v1_decoder",
    )


# @traced
def log_normal_pdf(sample, mean, logvar, raxis=1):
    """[summary]

    Args:
        sample ([type]): [description]
        mean ([type]): [description]
        logvar ([type]): [description]
        raxis (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


# @traced
def reparameterize(mean: tf.Tensor, logvar: tf.Tensor):
    """parameter trick allows training of the VAE

    Args:
        mean ([tf.Tensor]): gaussian means
        logvar ([tf.Tensor]): gaussian log variance

    Returns:
        tf.Tensor: sampled vector
    """
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean


# @traced
def compute_loss(encoder: Sequential, decoder: Sequential, value: tf.Tensor):
    """compute the VAE loss function

    Args:
        encoder ([type]): [description]
        decoder ([type]): [description]
        value ([type]): [description]

    Returns:
        [type]: [description]
    """
    encoded_value = encoder(value)
    mean, logvar = tf.split(encoded_value, 2, 1)
    z_value = reparameterize(mean, logvar)
    x_logit = decoder(z_value)
    # tf.print(f"x_logit, x.shape = {x_logit}, {x.shape}")
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=value)
    # tf.print(f"cross_ent.shape = {cross_ent.shape}")
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z_value, 0.0, 0.0)
    logqz_x = log_normal_pdf(z_value, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


# @traced
def compute_loss_via_channels(
    encoder: Sequential,
    decoder: Sequential,
    encoder_channels_extractor: tf.keras.models.Model,
    decoder_channels_extractor: tf.keras.models.Model,
    value: tf.Tensor,
):
    """compute the VAE loss function using encoded channels

    Args:
        encoder ([type]): [description]
        decoder ([type]): [description]
        value ([type]): [description]

    Returns:
        [type]: [description]
    """
    encoded_value = encoder(value)
    mean, logvar = tf.split(encoded_value, 2, 1)
    z_value = reparameterize(mean, logvar)
    # x_logit = decoder(z_value)

    encoded_channels_value = encoder_channels_extractor(value)
    encoded_channels_value = tf.sigmoid(encoded_channels_value)
    decoded_channels_value = decoder_channels_extractor(z_value)

    # tf.print(f"x_logit, x.shape = {x_logit}, {x.shape}")
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=decoded_channels_value, labels=encoded_channels_value
    )
    # tf.print(f"cross_ent.shape = {cross_ent.shape}")
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z_value, 0.0, 0.0)
    logqz_x = log_normal_pdf(z_value, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(
    encoder: tf.keras.models.Model,
    decoder: tf.keras.models.Model,
    encoder_channels_extractor: tf.keras.models.Model,
    decoder_channels_extractor: tf.keras.models.Model,
    x_samples: tf.Tensor,
    optimizer: tf.optimizers.Optimizer,
):
    """Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.

    Args:
        encoder ([type]): [description]
        decoder ([type]): [description]
        x_samples ([type]): [description]
        optimizer ([type]): [description]
    """

    all_trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    with tf.GradientTape() as tape:
        loss = compute_loss(encoder, decoder, x_samples)

        # now add loss due to channel matching
        # channel_weight = 0.1
        # loss += channel_weight * compute_loss_via_channels(
        #     encoder,
        #     decoder,
        #     encoder_channels_extractor,
        #     decoder_channels_extractor,
        #     x_samples,
        # )

        # tf.print(f"loss value = {loss}")
    gradients = tape.gradient(loss, all_trainable_variables)
    optimizer.apply_gradients(zip(gradients, all_trainable_variables))


class ZebraStackModel:
    """encapsulates the zebrastack VAE model

    Args:
        latent_dim (int, optional): [description]. Defaults to 8.
        use_v2 (bool, optional): [description]. Defaults to False.
    """

    def __init__(self, latent_dim=8, use_v2=False):
        self.latent_dim = latent_dim
        if use_v2:
            self.encoder = create_encoder_v2(latent_dim=latent_dim)
        else:
            self.encoder = create_encoder_v1(latent_dim=latent_dim)
        dense_shape = self.encoder.get_layer("ait_local").output_shape
        self.decoder = create_decoder(dense_shape, latent_dim=latent_dim)
        self.encoder_channels_extractor = tf.keras.Model(
            inputs=self.encoder.input,
            outputs=self.encoder.get_layer("v2_maxpool").output,
        )
        self.decoder_channels_extractor = tf.keras.Model(
            inputs=self.decoder.input,
            outputs=self.decoder.get_layer("v2_conv2d_trans").output,
        )

    def train(
        self,
        train_images: tf.Tensor,
        test_images: tf.Tensor,
        batch_size: int = 16,
        epoch_count: int = 10,
        callback: Optional[tf.keras.callbacks.Callback] = None,
    ):
        """perform training on train images; update with result of test images

        Args:
            train_images (tf.Tensor):
                train images in the form of a tensor
            test_images (tf.Tensor):
                test images
            batch_size (int, optional):
                batch size for training. Defaults to 16.
            epoch_count (int, optional):
                number of epochs. Defaults to 10.
            callback (Optional[tf.keras.callbacks.Callback], optional):
                training callback. Defaults to None.
        """

        # this will carry intermediate results to the callback
        logs = {"loss": None, "reconstructed": None}

        if callback:
            callback.set_model(self.encoder)
            callback.on_train_begin(logs=logs)

        optimizer = tf.keras.optimizers.Adam(1e-4)
        loss = tf.keras.metrics.Mean()
        for epoch in range(1, epoch_count):
            if callback:
                callback.on_epoch_begin(epoch, logs=logs)

            # perform training
            train_batches = generate_batches(train_images, batch_size)
            start_time = time()
            for step, input_batch in train_batches:
                if not step % 10000:
                    logging.info(f"{step}: {input_batch.shape}")

                train_step(
                    self.encoder,
                    self.decoder,
                    self.encoder_channels_extractor,
                    self.decoder_channels_extractor,
                    input_batch,
                    optimizer,
                )

            end_time = time()
            time_elapsed = end_time - start_time

            # calculate test results
            test_batches = generate_batches(test_images, batch_size)
            loss.reset_states()
            for step, test_batch in test_batches:
                if step == 0:
                    logs["original"] = test_batch
                    logs["reconstructed"] = self.generate(self.recognize(test_batch))
                loss(compute_loss(self.encoder, self.decoder, test_batch))

            elbo = -loss.result().numpy()
            logs["loss"] = elbo

            if callback:
                callback.on_epoch_end(epoch, logs=logs)

            logging.info(
                f"Epoch: {epoch}, test ELBO: {elbo}, time elapsed: {time_elapsed}"
            )

        if callback:
            callback.on_train_end(logs=logs)

    def recognize(self, image: tf.Tensor) -> tf.Tensor:
        """processes an image (4-d tensor, so can be a batch)
        returns latent variable, but only mean part

        Args:
            image (tf.Tensor):
                input tensor to be recognized

        Returns:
            tf.Tensor:
                tensor of latent_dim size, representing the encodings of the input tensors
        """
        encoder_output = self.encoder(image)
        return encoder_output[..., 0 : self.latent_dim]

    def generate(self, latent: tf.Tensor) -> tf.Tensor:
        """generates a tensor from the given latent tensor

        Args:
            latent (tf.Tensor):
                the latent tensor(s) to be regenerated

        Returns:
            tf.Tensor:
                the regenerated image(s)
        """
        assert latent.shape[-1] == self.latent_dim

        sigmoid = lambda x: np.exp(-np.logaddexp(0, -x))
        sigmoid_generated = sigmoid(self.decoder(latent))

        return sigmoid_generated

    def save_model(self, path: Path):
        """save the encoder and decoder as saved_models

        Args:
            path (Path): the path to save the models
        """
        self.encoder.save(path / "encoder")
        self.decoder.save(path / "decoder")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or run inference using zebrastack model."
    )
    parser.add_argument(
        "--encoder-ver",
        metavar="encoder version",
        type=int,
        default=2,
        help="version of the encoder: 1 or 2",
    )
    parser.add_argument(
        "--latent-dim",
        metavar="latent dimension",
        type=int,
        default=8,
        help="latent dimension for the autoencoder",
    )
    parser.add_argument(
        "--locally-connected",
        metavar="locally connected channels",
        type=int,
        default=2,
        help="locally connected channels for the autoencoder",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="the number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size for training",
    )
    parser.add_argument(
        "--infer",
        type=str,
        default=None,
        help="run inference on the specified image file",
    )
    parser.add_argument(
        "--generate",
        type=str,
        default=None,
        help="generate from string latent variable",
    )

    args = parser.parse_args()

    log_dir = configure_logger("figures")
    logging.info(f"tensorflow version = {tf.version.VERSION}")

    # create the model and write it out
    model = ZebraStackModel(latent_dim=args.latent_dim, use_v2=(args.encoder_ver == 2))

    # write model summaries to the log location
    model.encoder.summary(print_fn=logging.info)
    model.decoder.summary(print_fn=logging.info)

    # load the fashion mnist dataset
    (_train_images, _), (
        _test_images,
        _,
    ) = tf.keras.datasets.fashion_mnist.load_data()

    # select a subset
    _train_images, _test_images = (
        _train_images[: len(_train_images) // 1, ...],
        _test_images[: len(_test_images) // 1, ...],
    )

    # prepare the images for training
    _train_images = prepare_images(_train_images)
    _test_images = prepare_images(_test_images)
    logging.info(f"train_images: {_train_images.shape} {_train_images.dtype}")

    # set up figure callback for V2 layers
    if args.encoder_ver == 2:
        layer_names = [
            "v2_reduce",
            "v4_reduce",
            "pit_conv2d",
            "cit_conv2d",
            "ait_local",
        ]
    else:
        layer_names = ["pit_conv2d", "cit_conv2d", "ait_local"]
    nb_callback = FigureCallback(layer_names, log_dir)

    # now do training
    model.train(
        _train_images,
        _test_images,
        batch_size=args.batch_size,
        epoch_count=args.epochs,
        callback=nb_callback,
    )

    # now save the model to the log location
    model.save_model(log_dir)

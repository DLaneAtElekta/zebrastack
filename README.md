# zebrastack

add a new zebrastack.sln, to replace LeastSquares.sln

https://blog.keras.io/building-autoencoders-in-keras.html

Implementations of the network are at:
- anat0mixer/zebrastack_covidnet.ipynb
- keras_autoencoder/vae.py

Prepping the data at
- anat0mixer/prep_nih - a powershell notebook
- anat0mixer/imagegen.ipynb - tf generator
- dynamic_routing_encoder.ipynb
- anat0mixer.ipynb

Other (non Cxr8) data
- fashion_mnist.ipynb

## FashionMNIST VAE (merged from zebrastack_fashionmnist)

Continuation of VAE work adapted for FashionMNIST, including:
- `fashionmnist/zebrastack_model_v2.py` — V2 encoder model (V1>V2>V4 hierarchy)
- `fashionmnist/oriented_powermap_2d.py` — oriented power map layers
- `fashionmnist/logsumexp_pooling_2d.py` — logsumexp pooling
- `fashionmnist/anisotropic_diffusion.py` — anisotropic diffusion filter
- `fashionmnist/fashionmnist_train.ipynb` — training notebook
- `fashionmnist/FashionMnistVaeTfNet/` — C# TensorFlow.NET implementation

## ResNet18-VAE / Herringstack (merged from resnet18-vae)

Herring stack visual system model using ResNet as a base, including:
- ResNet encoder/decoder with squeeze-excitation modules
- Oriented power maps, CLAHE transform, filter utilities
- CXR8, NLST, Oxford Flowers dataset loaders
- HerringstackApi — C# ASP.NET Core API for image latent vectors
- Training notebooks for ResNet34/50 VAE variants

## PyTorch CIFAR10 (merged from pytorch-cifar)

CIFAR10 classification experiments with PyTorch, including model zoo:
VGG, ResNet, DenseNet, DPN, DLA, MobileNet, EfficientNet, and more.
Best accuracy: 95.47% (DLA).

keras_autoencoder\build_autoencoder.py:from keras.models import Model
keras_autoencoder\keras_autoencoder.py:from keras.models import Model
anat0mixer\dynamic_routing_encoder.ipynb:    "from keras.models import Model\n",
anat0mixer\zebrastack_v0_covidnet.ipynb:    "from keras.models import Model\n",
anat0mixer\zebrastack_v0_covidnet.ipynb:    "    encoder = keras.models.load_model('data\\zebrastack_v0_covidnet_encoder_model')\n",
anat0mixer\zebrastack_v0_covidnet.ipynb:    "    decoder = keras.models.load_model('data\\zebrastack_v0_covidnet_decoder_model')"
littlecarmash\model_vae_3stage.py:from tensorflow.keras.models import Model
littlecarmash\LittleCarVae-cardb.ipynb:    "from keras.models import Model\n",
littlecarmash\LittleCarVae.ipynb:    "from keras.models import Model\n",

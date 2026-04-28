import datetime
import logging
import os
from typing import Callable, Tuple
import matplotlib as mpl
# from matplotlib import colors, colormaps, image
import torch
import torch.nn.functional as F
from torchinfo import summary
from torchvision import transforms
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
import SimpleITK as sitk
from PIL import Image

from vae import load_model, train_vae

## these are some of the flags in this file
TODO_FIX_USE_MATCH = True


def match_histograms(fixed, moving):
    fixed = sitk.GetImageFromArray(fixed)
    moving = sitk.GetImageFromArray(moving)

    matcher = sitk.HistogramMatchingImageFilter()
    if fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    moving = sitk.GetArrayFromImage(moving)

    return moving

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

class Cxr8Dataset(Dataset):
    """class that represents the CXR8 chest x-ray dataset, with some pre-processing"""

    def __init__(
        self,
        root_path: str,
        transform: Callable[[torch.Tensor], torch.Tensor],
        sz: int | Tuple[int, int],
    ):
        """initialize the dataset with given root_path and transform

        Args:
            root_path (string): Directory with all the images.
            transform (Callable[[torch.Tensor], torch.Tensor]):  Optional transform to be applied
                on a sample.
            sz (int): size of input image.

            Actual input channels will be original, clahe_1, clahe_2, and
            oriented pyramids of the same, reduced to fixed channels
            TODO: include first oriented map as part of pre processing
        """
        self.root_path = root_path if root_path is Path else Path(root_path)

        csv_filename = self.root_path / "Data_Entry_2017_v2020.csv"
        self.data_entry_df = pd.read_csv(csv_filename)

        self.transform = transform
        if isinstance(sz, int):
            self.input_size = (sz, sz)
        else:
            self.input_size = sz

        clip_limit = 10

        self.clahe_filters = [
            cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
            )
            for tile_grid_size in [16, 8, 4]
        ]
        self.cache = {}


    def read_img_file(self, img_name: str):
        img_name = self.root_path / "images" / img_name
        img_name = str(img_name)
        # print(f"img_name {img_name}")
        if img_name in self.cache:
            return self.cache[img_name]
        
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.input_size)
        image_avg, image_std = (
            np.average(image),
            np.std(image),
        )
        # image = 0.5 + (image - image_avg) / (3.0 * image_std)
        image = image - image_avg
        image = image / (0.66 * image_std)
        image = 1.0 / (1.0 + np.exp(-image))
        logging.debug(f"image pre-clahe sigmoid: min={np.min(image)}, max={np.max(image)}")

        channel_list = [
            self.apply_clahe(filter, image, angles=[0,30,60]) for filter in self.clahe_filters
        ]

        image_result = np.stack(
            channel_list + [image],
            axis=-1,
        )
        image_result = image_result.astype(np.float32)

        # self.cache[img_name] = image_result
        return image_result

    def apply_clahe(self, clahe_filter, image, angles):
        # image = 1.0/(1.0*np.exp(-image))
        image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)
    
        # image = adjust_gamma(image, gamma=1.15)

        # add padding and rotate
        pad_thick = 64
        image = np.pad(image, pad_width=[(pad_thick,pad_thick),(pad_thick,pad_thick)], mode="reflect")
        image_center = tuple(np.array(image.shape[1::-1]) / 2)

        acc_image = None
        for angle in angles:
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            rot_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

            rot_image = clahe_filter.apply(rot_image)
            image_min, image_max, image_avg, image_std = (
                np.min(rot_image),
                np.max(rot_image),
                np.average(rot_image),
                np.std(rot_image),
            )
            logging.debug(
                f"image min/max/avg = {image_min}, {image_max}, {image_avg:.4f}, {image_std:.4f}"
            )
            # normalize to 4*std
            norm = "min_max"
            if norm == "min_max":
                rot_image = (rot_image - image_min) / (image_max - image_min)
            elif norm == "sigmoid":
                rot_image = rot_image - image_avg
                rot_image = rot_image / (0.96 * image_std)
                rot_image = 1.0 / (1.0 + np.exp(-rot_image))
                logging.debug(
                    f"image post-clahe sigmoid: min={np.min(rot_image)}, max={np.max(rot_image)}"
                )
            elif norm == "stddev":
                rot_image = 0.5 + (rot_image - image_avg) / (3.0 * image_std)

            rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
            rot_image = cv2.warpAffine(rot_image, rot_mat, rot_image.shape[1::-1], flags=cv2.INTER_LINEAR)
            rot_image = rot_image[pad_thick:-pad_thick,pad_thick:-pad_thick]

            rot_image = rot_image.astype(np.float32)
            if acc_image is None:
                acc_image = rot_image
            else:
                acc_image += rot_image

        return acc_image * (1.0/float(len(angles)))

    def __len__(self):
        return len(self.data_entry_df)

    def __str__(self):
        return f"{type(self)}: Dataset at {self.root_path} with {len(self)} items."

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if TODO_FIX_USE_MATCH:
            match self.data_entry_df["Image Index"].iloc[idx]:
                case pd.Series() as series:
                    images = series.apply(self.read_img_file)
                case str() as strName:
                    images = self.read_img_file(strName)

        image_index = self.data_entry_df["Image Index"].iloc[idx]
        if isinstance(image_index, pd.Series):
            images = image_index.apply(self.read_img_file)
            fns = image_index.values
        elif isinstance(image_index, str):
            images = self.read_img_file(image_index)
            fns = [image_index]
        else:
            raise ("unknown type")

        if self.transform:
            images = self.transform(images)

        finding_labels = self.data_entry_df["Finding Labels"].iloc[idx]
        if isinstance(finding_labels, pd.Series):
            finding_labels = finding_labels.str  # .split("|")
        elif isinstance(finding_labels, str):
            finding_labels = finding_labels  # .split("|")
        else:
            raise ("unknown type")

        samples = {"image": images, "labels": finding_labels, "filenames": fns}

        return samples


####
#########
############################
##########################################
#############################################################################################################


def infer_vae(device, input_size: Tuple[int, int, int], source_dir: str):
    """_summary_

    Args:
        device (_type_): _description_
        input_size (_type_): _description_
        source_dir (_type_): _description_
    """
    print(f"inferring images in {source_dir}")

    model, _, _ = load_model(
        (4, input_size, input_size),
        device,
        kernel_size=9,
        directions=7,
        latent_dim=32 * 32,  # 16 * 16,
        train_stn=False,
    )
    # model.eval()
    model.requires_grad_(False)

    print(
        summary(
            model,
            input_size=(37, 4, input_size, input_size),
            depth=10,
            col_names=[
                "input_size",
                "kernel_size",
                # "mult_adds",
                # "num_params",
                "output_size",
                "trainable",
            ],
        )
    )

    # TODO: use bokeh to create a figure for inferences
    logging.warn("TODO: use bokeh to create a figure for inferences")
    # for fn in Path(source_dir).glob("**/*.png"):
    #     print(fn)


    # TODO: move dataset preparation to cxr8_dataset.py
    data_temp_path = os.environ["DATA_TEMP"]
    root_path = Path(data_temp_path) / "cxr8"

    infer_dataset = Cxr8Dataset(
        root_path,
        sz=input_size,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    bone_cmap = mpl.colormaps["bone"]
    bone_cmap = mpl.colors.LinearSegmentedColormap(
        "bone_gm", mpl._cm.datad["bone"], 256, gamma=1.5
    )

    gamma = mpl.colors.PowerNorm(gamma=2.1, vmin=0.0, vmax=1.1)

    for n in range(0, len(infer_dataset)):
        sample = infer_dataset[n]
        image = sample["image"].to(device)
        image = torch.unsqueeze(image, 0)
        assert isinstance(image, torch.Tensor)
        # print(image)

        recon_image = model(image)
        recon_image = recon_image.detach()
        recon_image = recon_image.cpu()
        recon_image = recon_image.numpy()
        print(recon_image.shape)

        for ch in range(recon_image.shape[1]):
            slice_image = recon_image[0, ch, :, :]

            # slice_image = (slice_image - np.min(slice_image)) / (
            #     np.max(slice_image) - np.min(slice_image)
            # )
            # slice_image = bone_cmap(slice_image, gamma=1.0)
            # # slice_image = gamma(slice_image)
            # slice_image *= 255.0
            # slice_image = slice_image.astype(np.uint8)
            # slice_image = Image.fromarray(
            #     slice_image, "RGBA"
            # )  # 'L' mode for (8-bit grayscale pixels)

            # Save the image
            [fn] = sample["filenames"]
            fn = fn.split(".")[0]
            # slice_image.save(f"reconst_cxr8\{fn}_{ch}.png")
            # TODO: implement gamma
            # TODO: move to show_utils
            mpl.image.imsave(
                f"reconst_cxr8\{fn}_{ch}.png",
                np.around(120.0 * slice_image, decimals=0),
                vmin=0.0,
                vmax=120.0,
                cmap=bone_cmap, # "bone",
            )

        # TODO: implement bokeh shifter front end
        # ori_image = model(image) # torch.randn(1, 4, 512, 512) + 0.5
        # ori_latent, _ = model.encoder(image)
        # # ori_image = torch.clamp(ori_image, 0, 1)
        # ori_image = ori_image.to(device)
        # ori_image.requires_grad_(True)

        # opt = torch.optim.Adam([ori_image], lr=1e-3)

        # # Training:
        # for i in range(10000):
        #     opt.zero_grad()            
        #     latent, _ = model.encoder(ori_image)
        #     loss = F.mse_loss(ori_latent, latent)
        #     loss.backward()
        #     opt.step()
        #     if i % 10 == 0:
        #         print("Iteration %d, Loss=%f" % (
        #             i, float(loss)))

        # ori_image = ori_image.detach()
        # ori_image = ori_image.cpu()
        # ori_image = ori_image.numpy()        
        # for ch in range(ori_image.shape[1]):
        #     slice_image = ori_image[0, ch, :, :]

        #     slice_image = (slice_image - np.min(slice_image)) / (
        #         np.max(slice_image) - np.min(slice_image)
        #     )
        #     slice_image = gamma(slice_image)
        #     slice_image *= 255.0

        #     slice_image = slice_image.astype(np.uint8)
        #     slice_image = Image.fromarray(
        #         slice_image, "L"
        #     )  # 'L' mode for (8-bit grayscale pixels)

        #     # Save the image
        #     slice_image.save(f"ori_image_{ch}.png")            

if "__main__" == __name__:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        help="perform a single epoch of training with l1_weights/stn flags",
        # default="b10s/b60s/b10/b60"
    )
    parser.add_argument(
        "--infer",
        type=str,
        help="performance inference on images in specified directory",
        default="..\data",
    )
    args = parser.parse_args()

    # TODO: log config via yaml
    logging.warn("TODO: switch to log config via yaml")
    log_base = datetime.date.today().strftime("%Y%m%d")
    logging.basicConfig(
        filename=f"runs/{log_base}_vae_main.log",
        format="%(asctime)s|%(levelname)s|%(module)s|%(funcName)s|%(message)s",
        level=logging.DEBUG,
    )

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    logging.info(f"torch operations on {device} device")

    if args.train:
        train_vae(
            device,
            epic_cycles=1,
            protocol=args.train,
        )

    if args.infer:
        infer_vae(device, input_size=256, source_dir=args.infer)

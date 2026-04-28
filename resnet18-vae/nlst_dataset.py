from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset
import pydicom
import pandas as pd


class NltsDataset(Dataset):
    def __init__(self, root_path: Union[Path, str], sz: int = 512):
        if isinstance(root_path, str):
            root_path = Path(root_path)
        self.root_path = root_path
        self.sz = sz

        self.items = pd.DataFrame(
            [],
            columns=[
                "PatientID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "FrameOfReferenceUID",
                "SOPInstanceUID",
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "fn",
            ],
        )
        for dcm_fn in root_path.glob("**/*.dcm"):
            ds = pydicom.dcmread(dcm_fn, stop_before_pixels=True)
            assert ds.Modality == "CT"
            match ds.ImageType:
                case [("ORIGINAL"|"DERIVED"), ("PRIMARY"|"SECONDARY"), "AXIAL", *_]:
                    if ds.Rows != 512 or ds.Columns != 512:
                        print(f"Skipping images with {ds.Rows} x {ds.Columns}")
                        continue
                    self.items.loc[-1] = [
                        ds.PatientID,
                        ds.StudyInstanceUID,
                        ds.SeriesInstanceUID,
                        ds.FrameOfReferenceUID,
                        ds.SOPInstanceUID,
                        ds.ImagePositionPatient,
                        ds.ImageOrientationPatient,
                        str(dcm_fn),
                    ]
                    self.items.index = self.items.index + 1
                case [_, _, "LOCALIZER", *_]:
                    print(ds.PatientID, "Localizer")
                    continue
                case _:
                    raise (Exception("invalid type"))

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return f"{type(self)}: Dataset at {self.root_path} with {len(self)} items."

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # TODO: normalize images

        finding_labels = self.items[idx]["ImagePositionPatient"]
        return {"image": self.items[idx], "labels": finding_labels}


if __name__ == "__main__":
    root_path = Path("e:/") / "Shared" / "data" / "manifest-NLST_allCT" / "NLST"
    ds = NltsDataset(root_path=root_path)
    print(ds)

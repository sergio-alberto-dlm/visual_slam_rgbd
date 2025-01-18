import glob
import os

import numpy as np
import torch
from PIL import Image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, pose


class KITIParser:
    def __init__(self, input_folder, sequence):
        self.input_folder = input_folder
        self.sequence = sequence
        self.load_poses(self.input_folder, sequence)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def load_poses(self, datapath, sequence):
        # set directory paths 
        images_path = os.path.join(datapath, "gray_images", "sequences", sequence, "image_0")
        poses_path = os.path.join(datapath, "dataset_poses", "poses", f"{sequence}.txt")
        
        # get list of image files sorted numerically
        assert os.path.isdir(images_path), f"poses did not found in {images_path}"
        self.color_paths = sorted(glob.glob(os.path.join(images_path, "*.png")))
        
        # load poses 
        assert os.path.isfile(poses_path), f"poses did not found in {poses_path}"
        poses_data = self.parse_list(poses_path)
        self.poses = [np.array(pose.reshape(3, 4), dtype=np.float32) for pose in poses_data]


class KITIDataset(MonocularDataset):
    def __init__(self, config):
        super().__init__(config)
        dataset_path = config["Dataset"]["dataset_path"]
        self.sequence = config["Dataset"]["sequence"]
        parser = KITIParser(input_folder=dataset_path, sequence=self.sequence)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.poses = parser.poses


def load_dataset(config):
    if config["Dataset"]["type"] == "kitti":
        return KITIDataset(config)
    else:
        raise ValueError("Unknown dataset type")

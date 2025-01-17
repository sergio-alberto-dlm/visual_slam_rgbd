import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
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
        super().__init__(args, path, config)
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
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose


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
        # Check for poses file
        poses_path = os.path.join(datapath, "dataset_poses", "poses", f"{sequence}.txt")
        images_path = os.path.join(datapath, "gray_images", "sequences", sequence, "image_0")
        
        # Get list of image files sorted numerically
        self.color_paths = sorted(glob.glob(os.path.join(images_path, "*.png")))
        
        # Load poses if file exists
        if os.path.isfile(poses_path):
            poses_data = self.parse_list(poses_path)
            self.poses = [np.array(pose.reshape(3, 4), dtype=np.float32) for pose in poses_data]
        else:
            print(f"Warning: No poses file found at {poses_path}")
            self.poses = None

        # Set depth paths to None since KITTI raw dataset doesn't include depth
        self.depth_paths = None


class KITIDataset(MonocularDataset):
    def __init__(self, config):
        super().__init__(config)
        dataset_path = config["Dataset"]["dataset_path"]
        self.sequence = config["Dataset"]["sequence"]
        parser = KITIParser(input_folder=dataset_path, sequence=self.sequence)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


def load_dataset(config):
    if config["Dataset"]["type"] == "kitti":
        return KITIDataset(config)
    else:
        raise ValueError("Unknown dataset type")

import torch
from torch import nn
import cv2 
import numpy as np 
from torchvision import transforms 
import torchvision.transforms.functional as F 

from utils.vo_utils import getWorld2View2
from utils.vo_utils import image_gradient, image_gradient_mask
from submodules.MiDaS.midas.model_loader import default_models, load_model


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        image_height,
        image_width,
        model,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color.unsqueeze(0) if color.dim() == 2 else color 
        self.depth = None 
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_height = image_height
        self.image_width = image_width

        self.model = model  

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix, model):
        gt_color, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.height,
            dataset.width,
            model=model, 
            device=dataset.device,
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        median_img_grad_intensity = img_grad_intensity.median()
        self.grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edge_threshold
        )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

    @torch.no_grad()
    def compute_depth(self):
        resize_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the model's expected input size
            transforms.ToTensor(),          # Convert to tensor
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # Ensure 3 channels
        ])
        # convert to PIL image 
        image = F.to_pil_image(self.original_image)
        # resize to fit the model 
        image = resize_transform(image)
        # add a batch dimension
        image = image.unsqueeze(0)
        # Perform the prediction
        prediction = self.model.forward(image)
        self.depth = (
            torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=self.original_image.shape[1:], 
            mode="bicubic", 
            align_corners=False,
            )
        ).squeeze()
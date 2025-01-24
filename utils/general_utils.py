import yaml
import torch 
from submodules.MiDaS.midas.model_loader import default_models, load_model

def load_config(path):
    """
    Loads config file.
    """
    # load configuration from per scene/dataset cfg.
    with open(path, "r") as f:
        cfg = yaml.full_load(f)

    return cfg 

def init_model(model_type="dpt_swin2_tiny_256"):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimize=False
    side=False 
    height=None 
    square=False 
    grayscale=False 
    model_type=model_type
    model_path="submodules/MiDaS/weights/dpt_swin2_tiny_256.pt" 
    model, transform, net_w, net_h = load_model(device=device, model_path=model_path, model_type=model_type, optimize=optimize, height=height, square=square)
    return model
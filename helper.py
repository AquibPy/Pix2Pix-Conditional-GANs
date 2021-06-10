import torch
import torch.optim as optim
from utils import load_checkpoint
import numpy as np
import secrets
from PIL import Image
import config
from generator import Generator
from torchvision.utils import save_image
 # Loading Model
gen = Generator(in_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(),lr= config.LEARNING_RATE,betas=(0.5,0.999))
load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)

def generate_image(image_path: str):
    input_image = np.array(Image.open(image_path))
    augmentation = config.test_only(image=input_image)["image"]
    augmentation = torch.unsqueeze(augmentation,0).to(config.DEVICE)

    gen.eval()
    with torch.no_grad():
        y_fake = gen(augmentation)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        hashed = secrets.token_hex(8)
        filename = f"{hashed}.png"
        save_image(y_fake, f"images/output/{filename}")

    return filename
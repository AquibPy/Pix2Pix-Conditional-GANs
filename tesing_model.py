import torch
import torch.optim as optim
from utils import load_checkpoint
import numpy as np
from PIL import Image
import config
from generator import Generator
from torchvision.utils import save_image
 # Loading Model
gen = Generator(in_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(),lr= config.LEARNING_RATE,betas=(0.5,0.999))
load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)

# Input Image
image_path = "assets/1.jpg"
input_image = np.array(Image.open(image_path))
augmentation = config.test_only(image=input_image)["image"]
augmentation = torch.unsqueeze(augmentation,0).to(config.DEVICE)

#generating output
gen.eval()
with torch.no_grad():
    y_fake = gen(augmentation)
    y_fake = y_fake * 0.5 + 0.5  # remove normalization#
    save_image(y_fake,"output.png")
    save_image(augmentation * 0.5 + 0.5,"input.png")
gen.train()
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import save_some_examples,load_checkpoint
import config
from dataset import MapDataset
from generator import Generator
from torchvision.utils import save_image

val_dataset = MapDataset(config.VAL_DIR)
val_loader = DataLoader(val_dataset,batch_size=4,shuffle=False)
gen = Generator(in_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(),lr= config.LEARNING_RATE,betas=(0.5,0.999))
load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)

def save_some_examples(gen, val_loader, folder):
    for idx ,(x,y) in enumerate(val_loader):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            save_image(y_fake, folder + f"/y_gen_{idx}.png")
            save_image(x * 0.5 + 0.5, folder + f"/input_{idx}.png")
        gen.train()


if __name__=="__main__":
    save_some_examples(gen,val_loader,folder='evaluation')
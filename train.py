import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_checkpoint,save_some_examples,load_checkpoint
import config
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator

def train_fn(disc,gen,loader,opt_disc,opt_gen,l1,bce,g_scaler,d_scaler):
    loop = tqdm(loader,leave=True)
    for idx , (x,y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        #train discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x,y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real,torch.ones_like(D_real))
            D_fake_loss = bce(D_fake,torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
        
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        # train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake,torch.ones_like(D_fake))
            L1 = l1(y_fake,y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr= config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(),lr= config.LEARNING_RATE,betas=(0.5,0.999))
    BCE =nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC,disc,opt_disc,config.LEARNING_RATE)
    
    train_dataset = MapDataset(config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(config.VAL_DIR)
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE,g_scaler,d_scaler)

        if config.SAVE_MODEL and epoch % 5 ==0:
            save_checkpoint(disc,opt_disc,filename=config.CHECKPOINT_DISC)
            save_checkpoint(gen,opt_gen,filename= config.CHECKPOINT_GEN)
        
        save_some_examples(gen,val_loader,epoch,folder='evaluation')

if __name__ == "__main__":
    main()
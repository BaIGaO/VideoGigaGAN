# -*- coding: utf-8 -*-
#
# This script provides a complete training pipeline for the VideoGigaGAN model.
# It includes a mock dataset, model and optimizer initialization, and a training loop
# that alternates between training the discriminator and the generator.


import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from pathlib import Path
from PIL import Image
import random
import time

# Import AMP utilities
from torch.cuda.amp import GradScaler, autocast # Added for AMP

# Import from our custom modules
from models import VideoGigaGAN_Generator, VideoDiscriminator
from losses import GANLoss, CharbonnierLoss, r1_regularization, LPIPSLoss

# --- Configuration ---
class Config:
    # Paths
    DATASET_PATH = './data/mock_dataset' # Path to mock dataset
    CHECKPOINT_PATH = './output'

    # Training
    NUM_EPOCHS = 100
    BATCH_SIZE = 1
    NUM_FRAMES = 4 # Number of frames per video clip
    LR_G = 5e-5
    LR_D = 5e-5
    
    # Image size
    HR_SIZE = 256
    UPSCALE_FACTOR = 2
    LR_SIZE = HR_SIZE // UPSCALE_FACTOR
    
    # Loss weights from paper
    MU_GAN = 0.05
    MU_CHAR = 10.0
    MU_LPIPS = 5.0
    MU_R1 = 0.2048 / 2 # R1 is applied every other step, so we average the weight

    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 10
    R1_REG_FREQUENCY = 16 # Adjusted R1 regularization frequency

# --- Mock Dataset ---
class MockVideoDataset(Dataset):
    def __init__(self, root_dir, num_frames, hr_size, upscale_factor, num_videos=100):
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.hr_size = hr_size
        self.lr_size = hr_size // upscale_factor
        
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            print(f"Creating mock dataset at {self.root_dir}...")
            for v_idx in range(self.num_videos):
                video_path = self.root_dir / f"video_{v_idx:03d}"
                video_path.mkdir(parents=True, exist_ok=True)
                for f_idx in range(num_frames * 2): 
                    img = Image.fromarray((torch.rand(hr_size, hr_size, 3) * 255).byte().numpy())
                    img.save(video_path / f"frame_{f_idx:04d}.png")
        
        self.video_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        frame_paths = sorted(video_folder.glob("*.png"))
        
        start_frame_idx = random.randint(0, len(frame_paths) - self.num_frames)
        
        hr_frames = [TF.to_tensor(Image.open(frame_paths[start_frame_idx + i]).convert("RGB")) for i in range(self.num_frames)]
        hr_video = torch.stack(hr_frames)
        
        lr_video = F.interpolate(hr_video, size=(self.lr_size, self.lr_size), mode='bicubic', align_corners=False)
        
        return lr_video, hr_video

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MockVideoDataset(cfg.DATASET_PATH, cfg.NUM_FRAMES, cfg.HR_SIZE, cfg.UPSCALE_FACTOR)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    
    net_g = VideoGigaGAN_Generator(spynet_pretrained='./ckpt/spynet_20210409-c6c1bd09.pth',scale =cfg.UPSCALE_FACTOR).to(device)
    net_d = VideoDiscriminator().to(device)

    optimizer_g = optim.Adam(net_g.parameters(), lr=cfg.LR_G, betas=(0.9, 0.99))
    optimizer_d = optim.Adam(net_d.parameters(), lr=cfg.LR_D, betas=(0.9, 0.99))

    criterion_gan = GANLoss(gan_type='vanilla').to(device)
    criterion_char = CharbonnierLoss().to(device)
    criterion_lpips = LPIPSLoss(net='alex').to(device).eval()

    # --- Initialize GradScaler for AMP ---
    scaler = GradScaler() # Added for AMP

    total_steps = 0
    for epoch in range(cfg.NUM_EPOCHS):
        for i, (lr_video, hr_video) in enumerate(dataloader):
            total_steps += 1
            start_time = time.time()
            
            lr_video = lr_video.to(device)
            hr_video = hr_video.to(device)
    
            # --- Train Discriminator with AMP ---
            net_d.train()
            optimizer_d.zero_grad()
            
            with torch.no_grad(): # No AMP needed here as no gradients are computed
                fake_hr_video = net_g(lr_video)
            
            hr_video.requires_grad = True
            # Use autocast for the forward pass of the discriminator
            with autocast(): # Added for AMP
                real_pred = net_d(hr_video)
                loss_d_real = criterion_gan(real_pred, True, is_disc=True)
                fake_pred = net_d(fake_hr_video.detach())
                loss_d_fake = criterion_gan(fake_pred, False, is_disc=True)
                loss_d = (loss_d_real + loss_d_fake) / 2
                
                # Apply R1 Regularization periodically
                if total_steps % cfg.R1_REG_FREQUENCY == 0:
                    loss_r1 = r1_regularization(real_pred, hr_video)
                    loss_d += cfg.MU_R1 * loss_r1
                else:
                    loss_r1 = torch.tensor(0.0)

            # Scale the loss and call backward
            scaler.scale(loss_d).backward() # Modified for AMP
            # Step the optimizer using the scaled gradients
            scaler.step(optimizer_d) # Modified for AMP
            # Update the scaler for next iteration
            scaler.update() # Modified for AMP
            
            # --- Train Generator with AMP ---
            net_g.train()
            optimizer_g.zero_grad()
            
            # Use autocast for the forward pass of the generator and discriminator
            with autocast(): # Added for AMP
                fake_hr_video = net_g(lr_video)
                fake_pred_g = net_d(fake_hr_video)
                loss_g_adv = criterion_gan(fake_pred_g, True, is_disc=False)
                
                n, t, c, h, w = fake_hr_video.shape
                # 使用-1来自动推断维度大小
                fake_hr_video_reshaped = fake_hr_video.view(-1, c, h, w)
                hr_video_reshaped = hr_video.view(-1, c, h, w)
        
                loss_g_char = criterion_char(fake_hr_video_reshaped, hr_video_reshaped)
                loss_g_lpips = criterion_lpips(fake_hr_video_reshaped, hr_video_reshaped)
                
                loss_g = cfg.MU_GAN * loss_g_adv + cfg.MU_CHAR * loss_g_char + cfg.MU_LPIPS * loss_g_lpips

            # Scale the loss and call backward
            scaler.scale(loss_g).backward() # Modified for AMP
            # Step the optimizer using the scaled gradients
            scaler.step(optimizer_g) # Modified for AMP
            # Update the scaler for next iteration
            scaler.update() # Modified for AMP
    
            if total_steps % cfg.LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(f"[Epoch {epoch+1}/{cfg.NUM_EPOCHS}] [Step {total_steps}] [Time: {elapsed:.2f}s] "
                      f"[D Loss: {loss_d.item():.4f}] [G Loss: {loss_g.item():.4f}] "
                      f"[R1: {loss_r1.item():.4f}]")
                
            if total_steps % cfg.SAVE_INTERVAL == 0:
                Path(cfg.CHECKPOINT_PATH).mkdir(exist_ok=True)
                g_path = Path(cfg.CHECKPOINT_PATH) / f"net_g_step_{total_steps}.pth"
                d_path = Path(cfg.CHECKPOINT_PATH) / f"net_d_step_{total_steps}.pth"
                torch.save(net_g.state_dict(), g_path)
                torch.save(net_d.state_dict(), d_path)
                print(f"Saved checkpoints to {cfg.CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()




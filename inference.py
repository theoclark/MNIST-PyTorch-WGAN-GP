import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('num_images', type=int, default=50,
                    help='Number of images to generate')

def get_noise(n):
  return torch.rand(n,8,4,4).to(device)

class Upsample_block(nn.Module):
  """
  Increases the size of the "image" by a factor of 2
  """
  def __init__(self,chan_in,chan_out):
    super().__init__()
    self.model = nn.Sequential(
        nn.ConvTranspose2d(chan_in,chan_out,3,stride=2,padding=1,output_padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(chan_out),
        nn.Conv2d(chan_out,chan_out,3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(chan_out),
        nn.Conv2d(chan_out,chan_out,3,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(chan_out)
    )
  def forward(self,X):
    return self.model(X)

class Generator(nn.Module):
  def __init__(self, upsample_block):
    super().__init__()
    self.model = nn.Sequential(
      upsample_block(8,64), # 4 --> 8
      upsample_block(64,32), # 8 --> 16
    )
    self.final_block = nn.Sequential(
      nn.ConvTranspose2d(32,8,3,stride=2,padding=3,output_padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(8),
      nn.Conv2d(8,1,3,padding=1),
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self,Z):
    out = self.model(Z)
    out = self.final_block(out)
    out = self.sigmoid(out)
    return out

class Downsample_block(nn.Module):
  """
  Decreases the size of the "image" by a factor of 2
  """
  def __init__(self, chan_in, chan_out):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(chan_in, chan_out,3,stride=2,padding=1),
        nn.LeakyReLU()
    )
  def forward(self,X):
    return self.model(X)

class Discriminator(nn.Module):
  def __init__(self, downsample_block):
    super().__init__()
    self.model = nn.Sequential(
      downsample_block(1,16), # 28 --> 14
      downsample_block(16,32), # 14 --> 7
      downsample_block(32,16), # 7 --> 4
      downsample_block(16,8), # 4 --> 2
      downsample_block(8,1), # 2 --> 1
      nn.Sigmoid()
    )

  def forward(self,X):
    out = self.model(X)
    return out.squeeze()

def generate_fakes(number):
  Z = get_noise(n=number)
  fakes = G(Z).cpu()
  return fakes

def save_fake_sample(images):
  os.chdir("./Saved_Images")
  for i in tqdm(range(len(images))):
    img = images[i].squeeze(0).detach()
    img_name = f'fake-image-{i+1}.png'
    save_image(img,img_name)

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Running on {device}')

    args = parser.parse_args()
    number_of_fakes = args.num_images

    G = Generator(Upsample_block).to(device)
    D = Discriminator(Downsample_block).to(device)

    G = torch.load('Generator Weights.zip', map_location=torch.device(device))
    D = torch.load('Discriminator Weights.zip', map_location=torch.device(device))
    print('Model loaded')

    fakes = generate_fakes(number_of_fakes)
    print('Images generated')
    save_fake_sample(fakes)
    print('Images saved')

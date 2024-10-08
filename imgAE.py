import os 
#import wsi_preprocessing as pp
import torch
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import Upsample
from torch.nn import ConvTranspose2d
from PIL import Image
import torch.nn.functional as F
from torch import flatten
from math import sqrt
from torch.optim import SGD 
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.nn import BCELoss
from torch.optim import Adam
from torch.nn import Parameter
from torch.nn import ParameterList
from torch.utils.data import Dataset, DataLoader
import glob 
import torchvision.transforms as T
from matplotlib import pyplot as plt
import numpy as np 
import PIL
from torchvision.io import read_image
#import lightning.pytorch as pl
import torchvision
#import cifar10
import pandas as pd 
from torch import nn, optim, utils, Tensor
import HelperFunctions as HF



class CCA(torch.nn.Module):
    def __init__(self,n_channels_cnn):
        """
            patch_batch --> WSI patches
        """
        super(CCA,self).__init__()

 
        self.n_channels_cnn = n_channels_cnn
        self.model_parameter = ParameterList()
     #   self.encoder = nn.Sequential(
     #                    nn.Conv2d(in_channels= 3, out_channels= 10, kernel_size= (5,5)),
     #                    nn.ReLU(),
     #                    nn.MaxPool2d(kernel_size=(2,2)),
     #                    nn.Conv2d(in_channels= 10, out_channels = 2, kernel_size= (5,5)),
     #                    nn.ReLU(),
     #                    nn.MaxPool2d(kernel_size=(2,2)),
     #                    nn.Flatten(start_dim= 1),
     #                    nn.Linear(2 * 53 * 53, 1500),
     #                    nn.ReLU(),
     #                    nn.Linear(1500,128),
      #                   nn.ReLU(),
      #                   nn.Linear(128,64),
     #                    nn.Linear(64,2),
              
      #        )

        self.decoder_1 = nn.Sequential(
                      
                        nn.Linear(32,64),
                        nn.Linear(64,128),
                        nn.ReLU(),
                        nn.Linear(128,1500),
                        nn.ReLU(),
                        nn.Linear(1500, 2 * 53 * 53))
        
        self.decoder_2 = nn.Sequential(
                        nn.Unflatten(dim = 1, unflattened_size= torch.Size([2,53,53])),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_channels= 2, out_channels= 10, kernel_size= (5,5)),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size = (5,5))
                        )
        
        self.encoder_conv_to_flat = nn.Sequential(nn.Conv2d(in_channels= 3, out_channels= 10, kernel_size= (5,5)),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=(2,2)),
                         nn.Conv2d(in_channels= 10, out_channels = 2, kernel_size= (5,5)),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=(2,2)),
                         nn.Flatten(start_dim= 1))
        
        self.encoder_flat_to_canonical_var = nn.Sequential(nn.Linear(2 * 53 * 53, 1500),
                         nn.ReLU(),
                         nn.Linear(1500,128),
                         nn.ReLU(),
                         nn.Linear(128,64),
                         nn.Linear(64,32)
                         
                    )



    def forward(self,x):
        x_original = x
        flat = self.encoder_conv_to_flat(x_original)
        canonical = self.encoder_flat_to_canonical_var(flat)
        x_hat_prev = self.decoder_1(canonical)
        x_hat = self.decoder_2(x_hat_prev)

        return x_hat,x_hat_prev, flat, canonical 


    def training_step(self, batch):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch
        # normalize
        x_original = x / 255
        # size : batch x channels x width x height

      #  x_original = x.view(x.size(0), -1)
        flat = self.encoder_conv_to_flat(x_original)
        canonical = self.encoder_flat_to_canonical_var(flat)
        x_hat = self.decoder(canonical)

        loss = nn.functional.mse_loss(x_hat, x_original)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss 


 #   def configure_optimizers(self):
 #       optimizer = optim.Adam(self.parameters(), lr=0.001)
 #       return optimizer
        

"""
class PatchesDatasetTesting(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.img_paths = ['Slides/slide_data/sample1/p1_slide.svs_mpp-0.5_crop47-90_files/1',
                    'Slides/slide_data/sample2/p2_slide.svs_mpp-0.5_crop47-13_files/1',
                    'Slides/slide_data/sample3/p3_slide.svs_mpp-0.5_crop21-98_files/1']
        
        self.images = [[] for i in range(len(self.img_paths))]

        for c,img_path in enumerate(self.img_paths):
            directory = os.fsencode(img_path)
            for patch in os.listdir(directory):
                absolute_path = img_path + '/' + os.fsdecode(patch)
                img_tensor = read_image(path=absolute_path)

                self.images[c].append(img_tensor)
        
    def __len__(self):
        # Return amount of samples
        return len(self.images)
    
    def __getitem__(self,idx):

        return torch.stack((self.images[0][idx], self.images[1][idx], self.images[2][idx]), dim=0)
            

class PatchesDataset(Dataset):
    def __init__(self):
        self.img_path = 'Slides/slide_data/sample2/p2_slide.svs_mpp-0.5_crop47-13_files/1'
        # Store imgs here
        self.data = []
        directory = os.fsencode(self.img_path)
        for patch in os.listdir(directory):
            absolute_path = self.img_path + '/' + os.fsdecode(patch)


            img_tensor = read_image(path=absolute_path)
            # Can show tensor with this directly
   #         img_show = T.ToPILImage()(img_tensor)
   #         img_show.show()


            self.data.append(img_tensor)

    def __len__(self):
        # Amount of patches
        return len(self.data)
    

    def __getitem__(self,idx):
        img = self.data[idx]
        return img


class cifar10dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = []
        for c,data in enumerate(cifar10.data_batch_generator()):
            if c == 500: # Testing with 500 from cifar10 ; 32x32 images
                break
            else:
                img_tensor = torch.tensor(data[0]).permute(2,0,1) # Just taking images, no labels
                self.data.append(img_tensor)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        return img 
"""
    



def training_loop(batch_size,epochs):
    train_loss = []
    EPOCHS = epochs
    MODEL = CCA(n_channels_cnn=3)
    LOSS_FUNC = MSELoss()
    #Optimizer
    OPTIMIZER = Adam(MODEL.parameters(), lr=0.0001)      
    
   # dataset = cifar10dataset()
   # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 #   dataset = PatchesDatasetTesting()
 #   data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True) 


    # These two (imgs/2 for img 0 & 1, with just 1 patch ) with just a single patch work on NN (2*53*53,1500)-> ReLU -> (1500->128) -> ReLU -> (128,64) -> (64,32), lr = 0.0001
    # After about 1200 epochs ; also works for all pictures ! (0 to 18 imgs with 100 patches each, after 400/500 epochs)
    # Using just the first 3 Steps in NN works too, but takes longer (so reducing dimension helps)
    img_paths = ['0to128/0/imgs/1',
                          '0to128/1/imgs/1',
                          '0to128/2/imgs/1',
                          '0to128/5/imgs/1',
                          '0to128/6/imgs/1',
                          '0to128/7/imgs/1',
                          '0to128/8/imgs/1',
                          '0to128/9/imgs/1',
                          '0to128/10/imgs/1',
                          '0to128/11/imgs/1',
                          '0to128/12/imgs/1',
                          '0to128/13/imgs/1',
                          '0to128/14/imgs/1',
                          '0to128/15/imgs/1',
                          '0to128/16/imgs/1',
                          '0to128/18/imgs/1']
           
    


    
    images = [[] for i in range(len(img_paths))]

    for c,img_path in enumerate(img_paths):
        directory = os.fsencode(img_path)
        for patch in os.listdir(directory):
            absolute_path = img_path + '/' + os.fsdecode(patch)
            try:
                img_tensor = read_image(path=absolute_path)
            except RuntimeError:  # finds hidden mac file .ds_store 
                pass
            images[c].append(img_tensor)



    
    for i in range(EPOCHS):
      #  for c, data in enumerate(data_loader):
        for n_patch in range(len(images[0])):
            # for first 3 images
            data = torch.stack((images[0][n_patch],images[1][n_patch]))
          #  data = torch.stack((images[0][n_patch],images[1][n_patch],images[2][n_patch]),dim=0)

        
            # normalize images
            data = data/255
            data = data.type(torch.FloatTensor)
            
            x_hat,x_hat_prev, flat, canonical = MODEL.forward(data)
            loss = LOSS_FUNC(data,x_hat)
           # loss = loss.type(torch.FloatTensor)
            train_loss.append(float(loss))

            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

         #   print(f'epoch {i} step {c} loss {loss}')
            print(f'epoch {i} step {n_patch} loss {loss}')
             # Plot final loss after last epoch
    
            if i == EPOCHS - 1:
              #  for idx, data in enumerate(data_loader):
                amount_pics = 2
                og_images = [(data[idx, : ]) * 255 for idx in range(amount_pics)]
                    # Take first batch for testing purposes
              #      og_images = list(data.unbind(dim=0))
              #      data = data / 255
                #  data = data.view(data.size(0),-1)
                recon_images = MODEL.decoder(MODEL.encoder_flat_to_canonical_var(MODEL.encoder_conv_to_flat(data)))
                recon_images = [(recon_images[idx, :]) * 255 for idx in range(amount_pics)]
            # recon_images = [(recon_images[idx, :].reshape(3,32,32)) * 255 for idx in range(batch_size)]
                img_grid = torchvision.utils.make_grid(torch.stack(og_images + recon_images, dim=0), nrow=3, normalize=True, pad_value=0.5)

                img_grid = img_grid.permute(1, 2, 0)
                plt.figure(figsize=(3,3))
                plt.title("Reconstruction examples on MNIST")
                plt.imshow(img_grid)
                plt.axis("off")
              #  plt.show()
                plt.savefig('CCA-Project/img_only.png')
                plt.clf() # close plot so we don't save the same plot for the train loss (see below)
            #  plt.close()
                break

        train_loss = HF.avg_per_epoch(train_loss,i)

    plt.plot(train_loss, label='train_loss')
    plt.savefig('CCA-Project/training_img_only')
    plt.legend()
    plt.clf()
   # plt.show()



            
def tensor_to_image(tensor):
    tensor = (tensor*255)
  #  if np.ndim(tensor)>3:
  #      assert tensor.shape[0] == 1
  #      tensor = tensor[0]
    # Rearrange
    patch_visualization = T.ToPILImage()(tensor)
    patch_visualization.show()
   
 


if __name__ == '__main__':


    training_loop(1,2000)

    


    







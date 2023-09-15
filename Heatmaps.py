import torch
from torch.nn import Linear
from torch.nn import Upsample
from torch.nn import ConvTranspose2d
from math import sqrt
import torch.nn.functional as F
import pandas as pd 
import numpy as np 
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision.io import read_image
import imgAE
from torch.nn import ParameterList
import torchvision



# We only want the decoder part which reconstructs the image :
# We create the model (the decoder) and then set its weights from the pre-trained model







class decodernumAEtest(torch.nn.Module):
    def __init__(self, n_channels_cnn, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_channels_cnn = n_channels_cnn
        # IMPORTANT : in order to work, we can't just put everything in self.part_of_decoder() ; 
        # it needs to have the SAME structure as the imgAE !
        self.decoder = nn.Sequential(
                      
                        nn.Linear(32,64),
                        nn.Linear(64,128),
                        nn.ReLU(),
                        nn.Linear(128,1500),
                        nn.ReLU(),
                        nn.Linear(1500, 2 * 53 * 53),
                        nn.Unflatten(dim = 1, unflattened_size= torch.Size([2,53,53])),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_channels= 2, out_channels= 10, kernel_size= (5,5)),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size = (5,5))
                        )

        self.encoder_flat_to_canonical_var = nn.Sequential(nn.Linear(2 * 53 * 53, 1500),
                         nn.ReLU(),
                         nn.Linear(1500,128),
                         nn.ReLU(),
                         nn.Linear(128,64),
                         nn.Linear(64,32))
        

    def forward(self,x, att_weights):

        x = self.encoder_flat_to_canonical_var(x)
        x = self.decoder(x)

        att_weights = self.encoder_flat_to_canonical_var(att_weights)
        att_weights = self.decoder(att_weights)

        return x, att_weights



# TODO : for this to work, wrapperAE needs to be rewritten so that vector_patch in wrapperAE is the vector_patch
# in the decoding stage (that already went through encoding) ; right now, vector_patch is the one before the 
# full encdoding is done (just the one after CNN in flattened form)
class decodernumAE(torch.nn.Module):
    def __init__(self, n_channels_cnn, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_channels_cnn = n_channels_cnn

        self.part_of_decoder = nn.Sequential(
                        nn.Unflatten(dim = 1, unflattened_size= torch.Size([2,53,53])),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_channels= 2, out_channels= 10, kernel_size= (5,5)),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.ReLU(),
                        nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size = (5,5))
        )

    
    def forward(self,x,att_weights):


        x = self.part_of_decoder(x)
        att_weights = self.part_of_decoder(att_weights)

        return x, att_weights



def multi_rna_feats(rna_feats,sample, patch, version):
   
    """ Take a list of idx of rna features and calculate attention mapping based on all these rna features 
        by averaging over these."""
    
    patches_storage = []
    for counter,idx_rna in enumerate(rna_feats):
        att_mapping_patch = torch.unsqueeze(attention_weights(sample,idx_rna,patch,version), dim = 0).type(torch.FloatTensor)
        patches_storage.append(att_mapping_patch)
        if counter > 0:
            # Add all rna feats (there att mapping values) together
            patches_storage = [torch.add(patches_storage[0], patches_storage[-1])]

    # Take the average
    final_att_mapping = patches_storage[0] / len(rna_feats)


    return final_att_mapping
    



def attention_weights(sample_idx, rna_feature_idx, patch_idx,version):
    """
    Calculate attention weights for specific sample,rna feature and patch
    """
    # First, load data
    # TODO : make sample specific 
    coco_img = pd.read_csv("CCA-Project/coco_vals_{}/img_patches_sample_batch_0".format(version))
    coco_rna = pd.read_csv("CCA-Project/coco_vals_{}/rna_sample_batch_0".format(version))
    correlations = pd.read_csv("CCA-Project/coco_vals_{}/correlations_sample_batch_0".format(version))

    # Calculate the attention weight mapping
    att_mapping = torch.empty(coco_img.shape[0])
    # Specific patch
    patch = pd.Series(coco_img.iloc[:,patch_idx])
    # With according correlation
    correlation = correlations.iloc[patch_idx].values[1] # 1 bc first column is index
    # RNA value
    rna_value = coco_rna.iloc[rna_feature_idx].values[1] # 1 bc first column is index

    for idx,patch_value in patch.items():
        att_mapping[idx] = rna_value * correlation * patch_value


    return att_mapping



def tensor_to_image(tensor, type='patch'):

    if type.lower() == 'patch':
        # For heatmap, we keep values between 0/1
        #tensor = tensor*255 # TODO : plt seems to do clipping automatically, so this not needed 
        pass


    # Rearrange
    tensor = tensor.squeeze(dim=0).permute(1,2,0).detach()

    if type.lower() == 'patch':
        plt.imshow(tensor, interpolation='nearest')
        plt.savefig('CCA-Project/img_patch_s0_rna0_p4.png')
    elif type.lower() == 'heat':
        # We need to scale the heatmap values up, since they are really small (most between 0 and 0.1 --> we set them to a range
        # between 0 and 1)
        # TODO : scale up for each color (R/G/B) itself or scale for all three simultaneously ?
        # Scaling for RGB simultaneously 
     #   min_val = torch.min(tensor)
     #   max_val = torch.max(tensor)
     #   tensor = (1-0)/(max_val-min_val) * (tensor - max_val) + 1
        plt.imshow(tensor, cmap='hot', interpolation='nearest')
        plt.savefig('CCA-Project/attention_heatmap_patch_s0_rna0_p4.png')
  #  plt.show()
    

if __name__ == '__main__':


    PATH  = 'CCA-Project/cnn_ae_model/cnn_model.pt'
    PATH_2 = 'CCA-Project/cnn_ae_model/cnn_model_new.pt'

    pre_model = torch.load(PATH)
    pre_model_2 = torch.load(PATH_2)
    # Pretrained state dict (weights & biases)
    pretrained_dict = pre_model
    pretrained_dict_2 = pre_model_2

 

    # Model for which we want to overwrite weights
    model = decodernumAEtest(n_channels_cnn=3)
  #  model_2 = decodernumAE(n_channels_cnn=3)
    model_dict = model.state_dict()
   # model_dict_2 = model_2.state_dict()
    # Filter only necessary weights & biases
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
   # pretrained_dict_2 = {k: v for k, v in pretrained_dict_2.items() if k in model_dict_2}
    # update
    model_dict.update(pretrained_dict)
  #  model_dict_2.update(pretrained_dict_2)

    model.load_state_dict(model_dict)
  #  model_2.load_state_dict(model_dict_2)

    SAMPLE = 0
    RNA_FEATURE = 0
    PATCH = 4

    multiple_rna_feats = True

    img_patches = []
    att_patches = []

    VERSION = 1

    for PATCH in range(1,2):
        vector_patch = pd.read_csv("CCA-Project/coco_vals_{}/patches/vector_patches_sample_batch_0_image_patch_{}".format(VERSION,PATCH))
        vector_patch_decoder_stage = pd.read_csv("CCA-Project/coco_vals_{}/patches/vector_patches_sample_batch_0_image_patch_{}".format(VERSION,PATCH))

        vector_patch_sample = torch.unsqueeze(torch.tensor(vector_patch.iloc[SAMPLE,1:]), dim = 0).type(torch.FloatTensor)# 1 : .. since we have index column I think
        vector_patch_decoder_stage_sample = torch.unsqueeze(torch.tensor(vector_patch_decoder_stage.iloc[SAMPLE,1:]), dim = 0).type(torch.FloatTensor)
        if multiple_rna_feats == False:
            att_mapping_patch = torch.unsqueeze(attention_weights(SAMPLE,RNA_FEATURE,PATCH, VERSION), dim = 0).type(torch.FloatTensor)
        else:
            rna_feats_idx = [i for i in range(100)] 
            att_mapping_patch = multi_rna_feats(rna_feats_idx,SAMPLE,PATCH,VERSION)




        img_patch,att_patch = model.forward(vector_patch_sample, att_mapping_patch)

        img_patch = torch.squeeze(img_patch * 255,dim=0)

        att_patch = torch.squeeze(att_patch * 255,dim=0)
    
        img_patches.append(img_patch)
        att_patches.append(att_patch)

    img_patches = torch.stack(img_patches, dim=0)
    att_patches = torch.stack(att_patches, dim=0)
    patches = [img_patches, att_patches]
    patches = torch.cat(patches, dim=0)
   # att_patches = torch.stack(att_patches, dim=0)

    # TODO: attention images somehow have values over 300 after the NN, thus when normalizing everything together
    # in the img_grid , we get results that make no sense --> scale img_patches / att_patches again before ?
    img_grid = torchvision.utils.make_grid(patches, nrow=3, normalize=True, pad_value=0.5)
    img_grid = img_grid.permute(1, 2, 0)
    plt.figure(figsize=(3,2))
    plt.title("Reconstructed patches and attention weights")
    plt.imshow(img_grid)
    plt.axis("off")
    #  plt.show()
    plt.savefig('CCA-Project/img_and_att_2.png')
    plt.clf() # close plot so we don't save the same plot for the train loss (see below)

     #   tensor_to_image(img_patch, type='patch')
     #   tensor_to_image(att_patch, type='heat')












    #  img_example_path = '0to128/0/imgs/1/96_15.png'
  #  img_example_tensor = read_image(img_example_path)
  #  img_example_tensor = img_example_tensor.type(torch.FloatTensor) / 255
  #  tensor_to_image(img_example_tensor, type='patch')
  #  img_example_tensor = torch.unsqueeze(img_example_tensor,dim=0)
  #  model_2 = imgAE.CCA(n_channels_cnn=3)
  #  model_2.load_state_dict(pre_model)
  #  x,y,_ = model_2.forward(img_example_tensor) 
  #  x = torch.squeeze(x,dim=0)  


    



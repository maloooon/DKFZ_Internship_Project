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



# We only want the decoder part which reconstructs the image :
# We create the model (the decoder) and then set its weights from the pre-trained model


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

    
    def forward(self,x, att_weights):


        x = self.part_of_decoder(x)
        att_weights = self.part_of_decoder(att_weights)

        return x, att_weights




def attention_weights(sample_idx, rna_feature_idx, patch_idx):
    """
    Calculate attention weights for specific sample,rna feature and patch
    """
    # First, load data
    # TODO : make sample specific 
    coco_img = pd.read_csv("CCA-Project/coco_vals/img_patches_sample_batch_0")
    coco_rna = pd.read_csv("CCA-Project/coco_vals/rna_sample_batch_0")
    correlations = pd.read_csv("CCA-Project/coco_vals/correlations_sample_batch_0")

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
        #tensor = tensor*255 # TODO : plt seems to do clipping automatically, so this not needed ? 
        pass


    # Rearrange
    tensor = tensor.squeeze(dim=0).permute(1,2,0).detach()

    if type.lower() == 'patch':
        plt.imshow(tensor, interpolation='nearest')
        plt.savefig('CCA-Project/img_patch.png')
    elif type.lower() == 'heat':
        # We need to scale the heatmap values up, since they are really small (most between 0 and 0.1 --> we set them to a range
        # between 0 and 1)
        # TODO : scale up for each color (R/G/B) itself or scale for all three simultaneously ?
        # Scaling for RGB simultaneously 
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        tensor = (1-0)/(max_val-min_val) * (tensor - max_val) + 1
        plt.imshow(tensor, cmap='hot', interpolation='nearest')
        plt.savefig('CCA-Project/attention_heatmap_patch.png')
  #  plt.show()
    
   


if __name__ == '__main__':

    SAMPLE = 0
    RNA_FEATURE = 2
    PATCH = 2
    vector_patch = pd.read_csv("CCA-Project/coco_vals/patches/vector_patches_sample_batch_0_image_patch_{}".format(PATCH))

    vector_patch_sample = torch.unsqueeze(torch.tensor(vector_patch.iloc[SAMPLE,1:]), dim = 0).to(torch.float32)# 1 : .. since we have index column I think
    

    att_mapping_patch = torch.unsqueeze(attention_weights(SAMPLE,RNA_FEATURE,PATCH), dim = 0).to(torch.float32)

    PATH  = 'CCA-Project/cnn_ae_model/cnn_model.pt'

    pre_model = torch.load(PATH)
    # Pretrained state dict (weights & biases)
    pretrained_dict = pre_model


    # Model for which we want to overwrite weights
    model = decodernumAE(n_channels_cnn=3)
    model_dict = model.state_dict()
    # Filter only necessary weights & biases
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # update
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    img_patch, attention_patch = model.forward(vector_patch_sample, att_mapping_patch)

    tensor_to_image(img_patch, type='patch')
    tensor_to_image(attention_patch, type='heat')

    """
    patch_attention_weights = attention_weights(rna_feature_idx= 0, 
                      rna_loadings= 'FactorLoadings/test_batch_3/RNA/rna_fl_patch_44',
                      correlations= 'FactorLoadings/test_batch_3/Correlations/correlation_values_patches',
                      patches_loadings= ['FactorLoadings/test_batch_3/Patches/fl_patch_0']) # Testing with single patch


    PATH  = 'cnn_ae_model/cnn_model.pt'

    flattened_vectors_from_patches_df = pd.read_csv('FactorLoadings/test_batch_3/FlattenedImageVectorsPerPatch/flattened_vectors')
    # take first patch (first column)
    # Get into right structure (batch_size, features)
    flattened_vector_patch_0 = torch.tensor(flattened_vectors_from_patches_df['patch_0']).unsqueeze(dim=0)
    attention_weights_patch_0 = patch_attention_weights[0]

    pre_model = torch.load(PATH)
    # Pretrained state dict (weights & biases)
    pretrained_dict = pre_model

    # Model for which we want to overwrite weights
    model = decodernumAE(n_channels_cnn=3)
    model_dict = model.state_dict()
    # Filter only necessary weights & biases
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # update
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    img_patch, attention_patch = model.forward(flattened_vector_patch_0, attention_weights_patch_0)

    img_patch_show = tensor_to_image(attention_patch)

    print("h")
    """
   


    



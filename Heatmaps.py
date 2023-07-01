import torch
from torch.nn import Linear
from torch.nn import Upsample
from torch.nn import ConvTranspose2d
from math import sqrt
import torch.nn.functional as F
import pandas as pd 



# We only want the decoder part which reconstructs the image :
# We create the model (the decoder) and then set its weights from the pre-trained model


class decodernumAE(torch.nn.Module):
    def __init__(self, n_channels_cnn, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_channels_cnn = n_channels_cnn

        # Input will be vector of size 2809

        self.maxpool2_d = Upsample(scale_factor=2,mode='bilinear')
        self.conv2_d = ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=(5,5), stride=1)

        self.maxpool1_d = Upsample(scale_factor=2,mode='bilinear')
        self.conv1_d = ConvTranspose2d(in_channels=5, out_channels=n_channels_cnn, kernel_size=(5,5),stride=1)

    def nn_forward_pass(self,x, attention_weights):
        """
        x : first flattened vector (after last pooling operation) in imgAE
        
        """
        x_size = int(sqrt(x.size(dim=1))) 

        # Reshape into img form

        # Get size of curr batch
        curr_batch_size = x.size(dim=0)
        x = x.reshape([curr_batch_size,1,x_size,x_size])



        x = self.maxpool2_d(x)

        x = F.relu(self.conv2_d(x))

        x = self.maxpool1_d(x)
        x = F.relu(self.conv1_d(x))

        assert(x.size(dim=1) == self.n_channels_cnn 
               and x.size(dim=2) == 224 and x.size(dim=3) == 224) , "wrong sizes"

        return x



def attention_weights(rna_feature_idx,rna_loadings, correlations, patches_loadings):
    """
    Calculate attention values (loading rna feature * correlation * loading patch TODO: more sophisticated formula?) for a specific rna feature
    (TODO : or later maybe rna feature set rather?? that has something in common, bagging done like in MCAT)
    We receive attention weights for each patch corresponding to specific RNA feature
    :param rna_feature_idx : specific rna feature (right now given via index) , type : int
    :param rna_loadings : rna loadings , type : csv file
    :param correlations : correlation values for each patch , type : csv file
    :param patches : list of the all the patches loadings for an image , type : list of csv files
    """

    rna_loadings_df = pd.read_csv(rna_loadings)
    rna_loadings_tensor = torch.tensor(rna_loadings_df['0'].values)

    rna_feature_loading = rna_loadings_tensor[rna_feature_idx]

    correlations_df = pd.read_csv(correlations)
    correlations_tensor = torch.tensor(correlations_df['0'].values)

    patches_loadings_tensors = []
    for patch in patches_loadings:
        patches_loadings_df = pd.read_csv(patch)
        patches_loadings_tensors.append(torch.tensor(patches_loadings_df['0'].values))

    # Store attention weights for each patch
    patch_attention_weights = [torch.empty(size=(patches_loadings_tensors[0].size(dim=0),1)) for _ in range(len(patches_loadings_tensors))]
    
  
    for patch_idx, patch in enumerate(patches_loadings_tensors):
        for patch_loading_idx in range(patch.size(dim=0)):
            attention_w = rna_feature_loading * correlations_tensor[patch_idx] * patches_loadings_tensors[patch_idx][patch_loading_idx]
            patch_attention_weights[patch_idx][patch_loading_idx] = attention_w

    return patch_attention_weights
    








if __name__ == '__main__':

    patch_attention_weights = attention_weights(rna_feature_idx= 0, 
                      rna_loadings= 'FactorLoadings/test_batch_3/RNA/rna_fl_patch_44',
                      correlations= 'FactorLoadings/test_batch_3/Correlations/correlation_values_patches',
                      patches_loadings= ['FactorLoadings/test_batch_3/Patches/fl_patch_0']) # Testing with single patch


    PATH  = 'cnn_ae_model/cnn_model.pt'
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

   


    



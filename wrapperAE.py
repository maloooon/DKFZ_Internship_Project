from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
import pandas as pd 
import imgAE
import numAE
from torch.optim import SGD 
from torch.nn import MSELoss
from scipy.stats import pearsonr  
import itertools
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# TODO : ein Dataset/Dataloader , über Index dann an img / num AE geben
# TODO : own loss function (maybe as class like in SuMO) to wrap reconstruction loss and maximizing correlation



class fullDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.img_paths = ['Slides/slide_data/sample1/p1_slide.svs_mpp-0.5_crop47-90_files/1',
                          'Slides/slide_data/sample2/p2_slide.svs_mpp-0.5_crop47-13_files/1',
                          'Slides/slide_data/sample3/p3_slide.svs_mpp-0.5_crop21-98_files/1']

        self.num_path = 'Slides/data/rnaseq.csv' # TODO : just for RNAseq right now
        # Will store tuples of (img,rnaseq) for each sample ; note that img conists of 
        # multiple patches with the same rnaseq --> leads to problems for canonical variable calculation (std of same numerical data = 0, will lead
        # to correlation of NaN)
        
        # new idea : store tuples for different samples (RNA data, random patch (of that patient))

        # NUMERICAL DATA
        num_df = pd.read_csv(self.num_path)
        self.num_data = torch.from_numpy(num_df.values).float()

        self.data = []
        self.images = [[] for i in range(len(self.img_paths))]

        # IMAGING DATA
        for c,img_path in enumerate(self.img_paths):
            directory = os.fsencode(img_path)
            for patch in os.listdir(directory):
                absolute_path = img_path + '/' + os.fsdecode(patch)
                pixel_vals = torch.tensor(Image.open(absolute_path).getdata())

                # Assign RGB channels
                channel_R = pixel_vals[:,0]/255
                channel_G = pixel_vals[:,1]/255
                channel_B = pixel_vals[:,2]/255

                    
                img = torch.stack((channel_R.reshape(224,224),channel_G.reshape(224,224),channel_B.reshape(224,224)),dim=0)

                self.images[c].append(img)


        self.numerical_data = [self.num_data[0,:], self.num_data[1,:],self.num_data[2,:]]

        


    def __len__(self):
        # Return amount of patches
        return len(self.images)

    def __numfeats__(self):
        # Get number of features
        return self.numerical_data[0].size(dim=0)
    
    def __getitem__(self,idx):
       
        

        # but still based on same RNA data
        #           vector               list of patches
        return self.numerical_data[idx], self.images[idx]
        

def training_loop(batch_size,epochs):

    # Load data
    dataset = fullDataset()
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    EPOCHS = epochs

    # Models to train simultaneously
    MODEL_1 = imgAE.CCA(n_channels_cnn=3)
    MODEL_2 = numAE.numAE(in_size=dataset.__numfeats__())

    # optimizer
    OPTIMIZER_1 = SGD(MODEL_1.parameters(), lr=0.001, momentum=0.9)
    OPTIMIZER_2 = SGD(MODEL_2.parameters(), lr=0.001, momentum=0.9)

    # Loss function(s) : combination correlation maximization and reconstruction loss
    LOSS_FUNC_RECONSTRUCTION = MSELoss()
    

    for epoch in range(EPOCHS):
        for step, data in enumerate(data_loader):
            for counter,specific in enumerate(data):
                if counter == 0:
                    data[counter] = data[counter].type(torch.FloatTensor)
                else:
                    for counter2, _ in enumerate(specific):
                        data[counter][counter2] = data[counter][counter2].type(torch.FloatTensor)
            


            # Sample image patches (in one epoch we want to go through all patches for respective sample batch size)
            # e.g. batch_size = 3 gives 3 patches for 3 samples

            amount_patches = len(data[1])
            correlations = torch.empty(amount_patches,1)
            vector_patches_sample_0 = [] # TODO : testing for just one sample
            for substep, patch_sample in enumerate(data[1]):
            
                decoded_output_num, bottleneck_value_num = MODEL_2.nn_forward_pass(data[0])
                decoded_output_img, vector_patch, bottleneck_value_img = MODEL_1.nn_forward_pass(patch_sample)
                
                loss_reconstruction_num = LOSS_FUNC_RECONSTRUCTION(data[0], decoded_output_num)
                loss_reconstruction_img = LOSS_FUNC_RECONSTRUCTION(patch_sample, decoded_output_img)


                # Problem : we use patches of images with the same numerical data (as different patches refer to same patient)
                # Thus for the numerical data, we have the same values in a batch --> when calculating standard deviation
                # across batch where all values the same --> 0 and in correlation calculation divided by 0 --> returns NaN


                # bottleneck --> canonical variables
                bottleneck_value_num_list = list(itertools.chain.from_iterable(bottleneck_value_num.detach()))
                for c,i in enumerate(bottleneck_value_num_list):
                    bottleneck_value_num_list[c] = bottleneck_value_num_list[c].item()

                bottleneck_value_img_list = list(itertools.chain.from_iterable(bottleneck_value_img.detach()))
                for c,i in enumerate(bottleneck_value_img_list):
                    bottleneck_value_img_list[c] = bottleneck_value_img_list[c].item()


                # Maximize correlation --> minimize negative correlation
                loss_correlation = torch.tensor(pearsonr(bottleneck_value_num_list, bottleneck_value_img_list)[0],
                                                            requires_grad = True)
                
                correlations[substep] = loss_correlation

                alpha = 0.25
                beta = 0.25
                gamma = 0.5
                full_loss = alpha * loss_reconstruction_num + beta * loss_reconstruction_img + gamma * - loss_correlation




                if epoch == EPOCHS - 1:
                    # Factor loadings (or correlation coefficients) : Calculate correlation between original data features and bottlenecks
                    # For RNA, we calculate to the given corresponding rna features, for the image to the first resulting vector in CNN
                    amount_rna_features = data[0].size(dim=1)
                    amount_patch_features = vector_patch.size(dim=1)
                    amount_patches = len(data[1])
                    factor_loadings_rna = torch.empty(amount_rna_features,1)
                    factor_loadings_image_patch = torch.empty(amount_patch_features,1) # dim = 3 also possible (width x height)
      
                    # These two loops take long time (especially the one for RNA since 17000 values)

                    # For RNA data
                    for i in range(amount_rna_features):
                        current_rna_feature_list = data[0][:,i].tolist()
                        factor_loadings_rna[i] = pearsonr(bottleneck_value_num_list, current_rna_feature_list)[0]

                    # For patch vector data
                    for i in range(amount_patch_features):
                        current_patch_feature_list = vector_patch[:,i].tolist()
                        factor_loadings_image_patch[i] = pearsonr(bottleneck_value_img_list, current_patch_feature_list)[0]

                    


                    if substep == len(data[1]) - 1:
                        # Only needed for last step since we have the same rna data for the substeps
                        temp_loading_rna = pd.DataFrame(factor_loadings_rna.numpy())
                        temp_loading_rna.to_csv('FactorLoadings/test_batch_3/RNA/rna_fl_patch_{}'.format(substep))
                        # Also save all the correlation values just once
                        temp_correlations = pd.DataFrame(correlations.detach().numpy()) # TODO : ist detach global ? muss man hiernach wieder attachen ??
                        temp_correlations.to_csv('FactorLoadings/test_batch_3/Correlations/correlation_values_patches')
                    # TODO : zsm in eine csv file (rows : features, column patches)
                    temp_loading_patch = pd.DataFrame(factor_loadings_image_patch.numpy())
                    temp_loading_patch.to_csv('FactorLoadings/test_batch_3/Patches/fl_patch_{}'.format(substep))

                
                    vector_patches_sample_0.append(vector_patch[0,:].detach().tolist()) # TODO : ist detach global ? muss man hiernach wieder attachen ?? ; TODO : testing for just one sample
                    

                    if substep == 0:
                        decoded_patch_0 = decoded_output_img[0,:,:,:]
                        tensor_to_image(decoded_patch_0)
        



                OPTIMIZER_1.zero_grad()
                OPTIMIZER_2.zero_grad()
                full_loss.backward()
                OPTIMIZER_1.step()
                OPTIMIZER_2.step()
            
            

                print(f'epoch {epoch} step {step} substep {substep} loss reconstruction num {loss_reconstruction_num} loss reconstruction img {loss_reconstruction_img} loss correlation {loss_correlation}')

            names = ['patch_{}'.format(i) for i in range(len(data[1]))]
            vector_patches_df = pd.DataFrame.from_dict(dict(zip(names, vector_patches_sample_0))) # TODO : testing for just one sample
            vector_patches_df.to_csv('FactorLoadings/test_batch_3/FlattenedImageVectorsPerPatch/flattened_vectors')



            

    # save CNN model
    torch.save(MODEL_1.state_dict(), 'cnn_ae_model/cnn_model.pt')

    return decoded_patch_0


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor.detach(), dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    # Rearrange
    tensor = tensor.reshape(224,224,3)
    plt.imshow(tensor, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    patch_0 = training_loop(3,1)
    tensor_to_image(patch_0)




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
# TODO : ein Dataset/Dataloader , über Index dann an img / num AE geben
# TODO : own loss function (maybe as class like in SuMO) to wrap reconstruction loss and maximizing correlation



class fullDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.img_path = 'TCGA-05-4249-01Z-00-DX1.9fce0297-cc19-4c04-872c-4466ee4024db.svs_mpp-0.5_crop47-90_files/1' # TODO : change to 0 for real program
        self.num_path = 'Slides/data/rnaseq.csv' # TODO : just for RNAseq right now
        # Will store tuples of (img,rnaseq) for each sample ; note that img conists of 
        # multiple patches with the same rnaseq

        # NUMERICAL DATA
        num_df = pd.read_csv(self.num_path)
        self.num_data = torch.from_numpy(num_df.values).float()

        self.data = []
        self.curr_image = []

        # IMAGING DATA
        directory = os.fsencode(self.img_path)
        for patch in os.listdir(directory):
            absolute_path = self.img_path + '/' + os.fsdecode(patch)
            pixel_vals = torch.tensor(Image.open(absolute_path).getdata())

            # Assign RGB channels
            channel_R = pixel_vals[:,0]
            channel_G = pixel_vals[:,1]
            channel_B = pixel_vals[:,2]

                
            img = torch.stack((channel_R.reshape(224,224),channel_G.reshape(224,224),channel_B.reshape(224,224)),dim=0)

            self.curr_image.append(img)


        self.numerical_data = self.num_data[0,:]

        


    def __len__(self):
        # Return amount of patches
        return len(self.curr_image)

    def __numfeats__(self):
        # Get number of features
        return self.numerical_data.size(dim=0)
    
    def __getitem__(self,idx):
        # Get different patches
        img = self.curr_image[idx]

        # but still based on same RNA data
        return self.numerical_data, img
        

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
    

    for i in range(EPOCHS):
        for c, data in enumerate(data_loader):
            # data is list with [numerical features, patches]
            for counter,specific in enumerate(data):
                data[counter] = data[counter].type(torch.FloatTensor)
            
            decoded_output_num, bottleneck_value_num = MODEL_2.nn_forward_pass(data[0])
            decoded_output_img, bottleneck_value_img = MODEL_1.nn_forward_pass(data[1])
            
            loss_reconstruction_num = LOSS_FUNC_RECONSTRUCTION(data[0], decoded_output_num)
            loss_reconstruction_img = LOSS_FUNC_RECONSTRUCTION(data[1], decoded_output_img)


            # Problem : we use patches of images with the same numerical data (as different patches refer to same patient)
            # Thus for the numerical data, we have the same values in a batch --> when calculating standard deviation
            # across batch where all values the same --> 0 and in correlation calculation divided by 0 --> returns NaN

            # Calculate single correlations (in the end we have correlation between each patch and corresponding rna feature)
            correlation_losses = []
            curr_batch_size = bottleneck_value_img.size(dim=0)
            for batch_idx in range(curr_batch_size):
                loss_correlation = torch.tensor(pearsonr(bottleneck_value_num.detach().numpy()[batch_idx], bottleneck_value_img.detach().numpy()[batch_idx])[0], requires_grad = True)
                correlation_losses.append(loss_correlation)
            # negative, since we want to maximize correlation so we minimize negative correlation
            loss_correlation = - (sum(correlation_losses) / batch_size)
            alpha = 0.25
            beta = 0.25
            gamma = 0.5
            full_loss = alpha * loss_reconstruction_num + beta * loss_reconstruction_img + gamma * loss_correlation

            OPTIMIZER_1.zero_grad()
            OPTIMIZER_2.zero_grad()
            full_loss.backward()
            OPTIMIZER_1.step()
            OPTIMIZER_2.step()

            print(f'epoch {i} step {c} loss {full_loss}')



if __name__ == '__main__':
    training_loop(5,10)


  #  for c, numerical in enumerate(data_loader):


        
        # batch size x feature size (the same 5 rna sequences (same patient))
       # print(numerical[0].shape)
        # batch size x num channels x width x height (5 different patches, same patient)
       # print(numerical[1].shape)
       # break


        


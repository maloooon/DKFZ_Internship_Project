import torch
from torch.utils.data import Dataset, DataLoader
import os
#from PIL import Image
import pandas as pd 
import imgAE
import numAE
from torch.optim import SGD 
from torch.nn import MSELoss
#from scipy.stats import pearsonr  
import itertools
import numpy as np
#from PIL import Image
from matplotlib import pyplot as plt
#from torchmetrics import PearsonCorrCoef
import math 
from torch.optim import Adam
from torchvision.io import read_image
import HelperFunctions as HF
from torch.nn import CosineSimilarity #cosine embedding loss 
from torch.optim.lr_scheduler import StepLR


def get_free_gpu_idx():
    """Get the index of the GPU with current lowest memory usage."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used > tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmin(memory_available)


try:
    gpu_idx = get_free_gpu_idx()
    print("Using GPU #%s" % gpu_idx)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
except Exception as e:
    print(e)

    
class fullDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.img_paths = ['0to128/0/imgs/1',
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
                try:
                    img_tensor = read_image(path=absolute_path)
                except RuntimeError: # finds hidden mac file .ds_store 
                    pass

                self.images[c].append(img_tensor)
        

        self.numerical_data = [self.num_data[0,:], self.num_data[1,:],self.num_data[2,:],
                               self.num_data[5,:],self.num_data[6,:],self.num_data[7,:],
                               self.num_data[8,:],self.num_data[9,:],self.num_data[10,:],
                               self.num_data[11,:],self.num_data[12,:],self.num_data[13,:],
                               self.num_data[14,:],self.num_data[15,:],self.num_data[16,:],
                               self.num_data[18,:]]

        


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
    

class test_dataset(Dataset):
    """ Using only numerical data"""
    def __init__(self) -> None:
        super().__init__()
        num_path_dna = 'Slides/data/tests/DNA.csv'

        num_path_mRNA = 'Slides/data/tests/mRNA.csv'

        data_path_list = [num_path_dna,num_path_mRNA]

        self.data = []

        for data_path in data_path_list:
            num_df = pd.read_csv(data_path)
            num_data = torch.from_numpy(num_df.values).float()
            self.data.append(num_data)
        

        # keeps index for some reason
        for c,_ in enumerate(self.data):
            self.data[c] = self.data[c][:,1:]

   
    def __len__(self):
        # num samples
        return self.data[0].size(dim=0)

    def __numfeats__(self):
        # Get number of features
        return self.data[0].size(dim=1)

    def __getitem__(self,idx):
        return self.data[0][idx], self.data[1][idx]


def train_old_data(batch_size,epochs):
    """ Using only numerical data ; here, correlation optimization works ; trained on LUAD mRNA & DNA data """
    dataset = test_dataset()
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)


    EPOCHS = epochs

    MODEL_1 = numAE.numAE(in_size = dataset.__numfeats__())
    MODEL_2 = numAE.numAE(in_size = dataset.__numfeats__())

  #  OPTIMIZER_1 = SGD(MODEL_1.parameters(), lr=0.0001, momentum=0.9)
  #  OPTIMIZER_2 = SGD(MODEL_2.parameters(), lr=0.01, momentum=0.9)
    OPTIMIZER_1 = Adam(MODEL_1.parameters(), lr = 0.001)
    OPTIMIZER_2 = Adam(MODEL_2.parameters(), lr = 0.0001)

    LOSS_FUNC_RECONSTRUCTION = MSELoss()

    train_loss_reconstruction_dna = []
    train_loss_reconstruction_mRNA = []
    train_loss_correlation = []
    train_loss = []



    for epoch in range(EPOCHS):
        for c, data in enumerate(data_loader):
            mRNA_data = data[0].type(torch.FloatTensor)
            DNA_data = data[1].type(torch.FloatTensor)


            
            decoded_output_mRNA, bottleneck_value_mRNA = MODEL_1.forward(mRNA_data)
            decoded_output_DNA, bottleneck_value_DNA = MODEL_2.forward(DNA_data)

            loss_mRNA = LOSS_FUNC_RECONSTRUCTION(mRNA_data, decoded_output_mRNA)
            loss_DNA = LOSS_FUNC_RECONSTRUCTION(DNA_data, decoded_output_DNA)

            pearson = 1#PearsonCorrCoef()

            loss_correlation = pearson(bottleneck_value_mRNA.clone().detach().squeeze(dim=1), bottleneck_value_DNA.clone().detach().squeeze(dim=1))

            alpha = 0.3
            beta = 0.3
            gamma = 0.4

            full_loss = alpha * loss_mRNA + beta * loss_DNA + gamma * - loss_correlation
            
            OPTIMIZER_1.zero_grad()
            OPTIMIZER_2.zero_grad()
            full_loss.backward()
            OPTIMIZER_1.step()
            OPTIMIZER_2.step()

            train_loss_reconstruction_dna.append(float(loss_DNA.clone().detach()))
            train_loss_reconstruction_mRNA.append(float(loss_mRNA.clone().detach()))
            train_loss.append(float(full_loss.clone().detach()))
            train_loss_correlation.append(float(loss_correlation.clone().detach()))

            print(f'epoch {epoch} step {c} full loss {full_loss} dna {loss_DNA} mrna {loss_mRNA}')
        
 

        train_loss = HF.avg_per_epoch(train_loss,epoch)

        train_loss_reconstruction_dna = HF.avg_per_epoch(train_loss_reconstruction_dna,epoch)

        train_loss_reconstruction_mRNA = HF.avg_per_epoch(train_loss_reconstruction_mRNA,epoch)

        train_loss_correlation = HF.avg_per_epoch(train_loss_correlation,epoch)

    plt.plot(train_loss, label='loss overall', color= 'blue')
    plt.plot(train_loss_reconstruction_dna, label='dna loss', color='green')
    plt.plot(train_loss_reconstruction_mRNA, label ='mrna loss', color='purple')
    plt.plot(train_loss_correlation, label='correlation', color='red')
    plt.legend(loc='upper center')
    plt.show()
    
        



   # img_paths = ['Slides/slide_data/sample1/p1_slide.svs_mpp-0.5_crop47-90_files/1',
  #                      'Slides/slide_data/sample2/p2_slide.svs_mpp-0.5_crop47-13_files/1',
  #                      'Slides/slide_data/sample3/p3_slide.svs_mpp-0.5_crop21-98_files/1']
    
    img_paths = ['Slides/slide_data/sample_n_6/p6_slide_imgs/1',
            'Slides/slide_data/sample_n_7/p7_slide_imgs/1',
            'Slides/slide_data/sample_n_8/p8_slide_imgs/1',
            'Slides/slide_data/sample_n_9/p9_slide_imgs/1',
            'Slides/slide_data/sample_n_10/p10_slide_imgs/1']

    num_path = 'Slides/data/rnaseq.csv' # TODO : just for RNAseq right now
    # Will store tuples of (img,rnaseq) for each sample ; note that img conists of 
    # multiple patches with the same rnaseq --> leads to problems for canonical variable calculation (std of same numerical data = 0, will lead
    # to correlation of NaN)
    
    # new idea : store tuples for different samples (RNA data, random patch (of that patient))

    # NUMERICAL DATA
    num_df = pd.read_csv(num_path)
    num_data = torch.from_numpy(num_df.values).float()

    data = []
    images = [[] for i in range(len(img_paths))]

    # IMAGING DATA
    for c,img_path in enumerate(img_paths):
        directory = os.fsencode(img_path)
        for patch in os.listdir(directory):
            absolute_path = img_path + '/' + os.fsdecode(patch)
            img_tensor = read_image(path=absolute_path)

            images[c].append(img_tensor)


 #   numerical_data = [num_data[0,:], num_data[1,:],num_data[2,:]]
 #   data_num = torch.stack((num_data[0,:], num_data[1,:],num_data[2,:]), dim=0)

    numerical_data = [num_data[6,:], num_data[7,:],num_data[8,:],num_data[9,:],num_data[10,:]]
    data_num = torch.stack((num_data[6,:], num_data[7,:],num_data[8,:],num_data[9,:],num_data[10,:]), dim=0)


    EPOCHS = epochs

    MODEL_1 = imgAE.CCA(n_channels_cnn=3)
    MODEL_2 = numAE.numAE(in_size = num_data[0,:].size(dim=0))

    # optimizer
    OPTIMIZER_1 = Adam(MODEL_2.parameters(), lr=0.0001)
    OPTIMIZER_2 = SGD(MODEL_1.parameters(), lr=0.0001)

    # Loss function(s) : combination correlation maximization and reconstruction loss
    LOSS_FUNC_RECONSTRUCTION = MSELoss()
    
    train_loss = []
    train_loss_reconstruction_rna = []
    train_loss_reconstruction_img = []
    train_loss_correlation = []

    for epoch in range(EPOCHS):
        correlations = []
        for n_patch in range(len(images[0])):
         #   data_img = torch.stack((images[0][n_patch],images[1][n_patch],images[2][n_patch]),dim=0)
            data_img = torch.stack((images[0][n_patch],images[1][n_patch],images[2][n_patch],images[3][n_patch],images[4][n_patch]),dim=0)
            data_img = data_img/255
            data_img = data_img.type(torch.FloatTensor)
            data_num = data_num.type(torch.FloatTensor)

            amount_patches = len(images[0])
    
           # correlations = torch.empty(amount_patches,1,requires_grad=False) # Just save correlations, no gradient needed for that list

            decoded_output_img, vector_patch, bottleneck_value_img = MODEL_1.forward(data_img)
            decoded_output_num, bottleneck_value_num = MODEL_2.forward(data_num)
            
                
            loss_reconstruction_img = LOSS_FUNC_RECONSTRUCTION(data_img, decoded_output_img)
            loss_reconstruction_num = LOSS_FUNC_RECONSTRUCTION(data_num, decoded_output_num)


            
            pearson = 1#PearsonCorrCoef()
            n_canonical_variables = bottleneck_value_num.size(dim=1)
            # corr_matrix (row : rna values, column : img values)
            corr_matrix = torch.empty(size=(n_canonical_variables,n_canonical_variables))
    
            # TODO : high bottleneck
            # Go through all canonical variables (per batch)
         #   for i in range(n_canonical_variables):
         #       for j in range(n_canonical_variables):
         #           corr_matrix[i,j] = pearson(bottleneck_value_num[:,i].clone().detach(),bottleneck_value_img[:,j].clone().detach())
                    # Need to clone & detach bc. otherwise the bottlneck values always go NaN after the first iteration (TODO: why ????)



            
            # This calculation gives slightly different values (ab 5/6. nachkommastelle)
            counter_img = 0
            idx = 0

            if n_canonical_variables > 1:
                for idx_pair, pair in enumerate(itertools.product(bottleneck_value_img.clone().detach().unbind(dim=1), bottleneck_value_num.clone().detach().unbind(dim=1))):
                    corr_matrix[idx,counter_img] = pearson(pair[0].clone().detach(),pair[1].clone().detach())
                    if idx % (n_canonical_variables - 1) == 0 and idx != 0:
                        idx = 0
                        counter_img += 1
                    else:
                        idx += 1
                    


                non_nan_rows = []
                # keep information of rows which have no NaN values
                for row in range(corr_matrix.size(dim=0)):
                    if ~torch.any(torch.isnan(corr_matrix[row,:])):
                        non_nan_rows.append(row)


                # Remove rows with NaN values
                corr_matrix = corr_matrix[~torch.isnan(corr_matrix).any(axis=1)]

                # TODO : smth more sophisticated ?
                loss_correlation = torch.mean(corr_matrix)
            else:
                loss_correlation = pearson(bottleneck_value_img.clone().detach().squeeze(dim=1), bottleneck_value_num.clone().detach().squeeze(dim=1))

            correlations.append(loss_correlation.clone().detach())

            loss_correlation_d_copy = loss_correlation.clone().detach()
         #   correlations[n_patch] = loss_correlation_d_copy

            

            alpha = 0.3 # same samples during substeps --> learns fast
            beta = 0.3
            gamma = 0.4
            full_loss = alpha * loss_reconstruction_num + beta * loss_reconstruction_img + gamma * - loss_correlation
            train_loss.append(float(full_loss))
            train_loss_reconstruction_rna.append(float(loss_reconstruction_num))
            train_loss_reconstruction_img.append(float(loss_reconstruction_img))
            train_loss_correlation.append(float(loss_correlation))

            """
            if epoch == EPOCHS - 1:
                # Factor loadings (or correlation coefficients) : Calculate correlation between original data features and bottlenecks
                # For RNA, we calculate to the given corresponding rna features, for the image to the first resulting vector in CNN
                amount_rna_features = data_num.size(dim=1)
                amount_patch_features = vector_patch.size(dim=1)
                factor_loadings_rna = torch.empty(amount_rna_features, n_canonical_variables,requires_grad=False)
                factor_loadings_image_patch = torch.empty(amount_patch_features , n_canonical_variables, requires_grad=False) # dim = 3 also possible (width x height)
    
                # These two loops take long time (especially the one for RNA since 17000 values)

                # For RNA data
                canonical_variables_num = bottleneck_value_num.clone().detach().unbind(dim=1)
                data_num_d = data_num.clone().detach().unbind(dim=1)[1:] # first one 0,1,2 ; think from pandas somehow indexing ? TODO : check


                idx = 0
                counter_canonical_variables = 0
                for idx_pair, pair in enumerate(itertools.product(data_num_d, canonical_variables_num)):
                    factor_loadings_rna[idx, counter_canonical_variables] = pearson(pair[0], pair[1])
                    if idx % (amount_rna_features - 1) == 0 and idx != 0:
                        idx = 0
                        counter_canonical_variables += 1
                    else:
                        idx += 1

                # Remove features (rows) with nan values TODO : what if in other batches (sample groups), these are not nan values ? how to handle?
                non_nan_features_rna = []
                # keep information of rows which have not only NaN values
                for row in range(factor_loadings_rna.size(dim=0)):
                    if ~torch.all(torch.isnan(factor_loadings_rna[row,:])):
                        non_nan_features_rna.append(row)
                non_nan_canonical_rna = []
                for column in range(factor_loadings_rna.size(dim=1)):
                    if ~torch.all(torch.isnan(factor_loadings_rna[:,column])):
                        non_nan_canonical_rna.append(column)
                    


                # TODO : kann sein, dass man hier erst feature selection machen muss , deswegen komisch verteilte NaN values

                # Delete canonical variables (columns) which ONLY consist out of NaN values
                factor_loadings_rna = factor_loadings_rna[:,~torch.isnan(factor_loadings_rna).all(axis=0)] 
                # Delete features (rows) which ONLY consist out of NaN values
                factor_loadings_rna = factor_loadings_rna[~torch.isnan(factor_loadings_rna).all(axis=1)] 

                # For image data
                canonical_variables_img = bottleneck_value_img.clone().detach().unbind(dim=1)
                # Take first vector of images after passing it through conv/pooling layers
                data_img_vector_d = vector_patch.clone().detach().unbind(dim=1)
                idx = 0
                counter_canonical_variables = 0
                for idx_pair, pair in enumerate(itertools.product(data_img_vector_d, canonical_variables_img)):
                    factor_loadings_image_patch[idx, counter_canonical_variables] = pearson(pair[0], pair[1])
                    if idx % (amount_patch_features - 1) == 0 and idx != 0:
                        idx = 0
                        counter_canonical_variables += 1
                    else:
                        idx += 1


                non_nan_features_patch = []
                # keep information of rows which have not only NaN values
                for row in range(factor_loadings_image_patch.size(dim=0)):
                    if ~torch.all(torch.isnan(factor_loadings_image_patch[row,:])):
                        non_nan_features_patch.append(row)
                non_nan_canonical_patch = []
                for column in range(factor_loadings_image_patch.size(dim=1)):
                    if ~torch.all(torch.isnan(factor_loadings_image_patch[:,column])):
                        non_nan_canonical_patch.append(column)
                
                # Delete canonical variables (columns) which ONLY consist out of NaN values
                factor_loadings_image_patch = factor_loadings_image_patch[:,~torch.isnan(factor_loadings_image_patch).all(axis=0)]
                # Delete features (rows) which ONLY consist out of NaN values
                factor_loadings_image_patch = factor_loadings_image_patch[~torch.isnan(factor_loadings_image_patch).all(axis=1)]



            
                if n_patch == amount_patches - 1:
                    # Only needed for last patch since we have the same rna data for all the patches in one sample group
                    temp_loading_rna = pd.DataFrame(factor_loadings_rna.numpy())
                    temp_loading_rna.to_csv('FactorLoadings/test_batch_3/RNA/rna_fl_patch_{}'.format(n_patch))
                    # Also save all the correlation values just once
                    temp_correlations = pd.DataFrame(correlations.numpy()) 
                    temp_correlations.to_csv('FactorLoadings/test_batch_3/Correlations/correlation_values_patches')
                # TODO : zsm in eine csv file (rows : features, column patches)
                temp_loading_patch = pd.DataFrame(factor_loadings_image_patch.numpy())
                temp_loading_patch.to_csv('FactorLoadings/test_batch_3/Patches/fl_patch_{}'.format(n_patch))

              #  vector_patch_0_d_copy = vector_patch[0,:].clone().detach()
              #  vector_patches_sample_0.append(vector_patch_0_d_copy.tolist()) # TODO : test for single sample 0
        
            """

            OPTIMIZER_1.zero_grad()
            OPTIMIZER_2.zero_grad()
            full_loss.backward()
            OPTIMIZER_1.step()
            OPTIMIZER_2.step()


            print(f'epoch {epoch} substep {n_patch} loss reconstruction num {loss_reconstruction_num} loss reconstruction img {loss_reconstruction_img} correlation {loss_correlation}')
            """
            names = ['patch_{}'.format(i) for i in range(len(data[1]))]
            vector_patches_df = pd.DataFrame.from_dict(dict(zip(names, vector_patches_sample_0))) # TODO : testing for just one sample
            vector_patches_df.to_csv('FactorLoadings/test_batch_3/FlattenedImageVectorsPerPatch/flattened_vectors')
            """
        train_loss = HF.avg_per_epoch(train_loss,epoch)

        train_loss_reconstruction_rna = HF.avg_per_epoch(train_loss_reconstruction_rna,epoch)

        train_loss_reconstruction_img = HF.avg_per_epoch(train_loss_reconstruction_img,epoch)

        train_loss_correlation = HF.avg_per_epoch(train_loss_correlation,epoch)

    # Plot final loss after last epoch


    plt.plot(train_loss, label='train_loss', color= 'blue')
    plt.plot(train_loss_reconstruction_rna, label='rna loss', color='green')
    plt.plot(train_loss_reconstruction_img, label ='img loss', color='purple')
    plt.plot(train_loss_correlation, label='correlation', color='red')
    plt.legend(loc='upper center')
    plt.show()
            

    # save CNN model
    torch.save(MODEL_1.state_dict(), 'cnn_ae_model/cnn_model.pt')


def training_loop(batch_size,epochs):

    # Load data
    dataset = fullDataset()
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    EPOCHS = epochs

    # Models to train simultaneously
    MODEL_1 = imgAE.CCA(n_channels_cnn=3)
    MODEL_2 = numAE.numAE(in_size=dataset.__numfeats__())
    

    # optimizer
    OPTIMIZER_1 = Adam(MODEL_1.parameters(), lr=0.0001) # img
    OPTIMIZER_2 = Adam(MODEL_2.parameters(), lr=0.001) # rna
    

    # Loss function(s) : combination correlation maximization and reconstruction loss
    LOSS_FUNC_RECONSTRUCTION = MSELoss()
    
    train_loss = []
    train_loss_reconstruction_rna = []
    train_loss_reconstruction_img = []
    train_loss_correlation = []

  #  scheduler_1 = StepLR(optimizer=OPTIMIZER_1, step_size=250, gamma=0.5) # every 50 epochs, calculate current lr with 0.5
  #  scheduler_2 = StepLR(optimizer=OPTIMIZER_2, step_size=250, gamma=0.5)


    
    for epoch in range(EPOCHS):
        
        for step, data in enumerate(data_loader):
            for counter,specific in enumerate(data):
                if counter == 0:
                    # At position 0 , we have RNA data (batch_size, features)
                    data[counter] = data[counter].type(torch.FloatTensor)
                else:
                    # At position 1 , we have the patches (patches, (batch_size, channels, width, height))
                    for counter2, _ in enumerate(specific):
                        data[counter][counter2] = data[counter][counter2].type(torch.FloatTensor) / 255
                        
            
        
            # Sample image patches (in one epoch we want to go through all patches for respective sample batch size)
            # e.g. batch_size = 3 gives 3 patches for 3 samples

            amount_patches = len(data[1])
            
            correlations_storage = []
            vector_patches_storage = []


            coco_img_storage = [] # need flexible data structure to store coco of img patches (tensor doesn't work as we don't know size of vector_patch yet)
            for substep, patch_sample in enumerate(data[1]):
            
                decoded_output_num, bottleneck_value_num = MODEL_2.forward(data[0])
                decoded_output_img, decoded_output_img_prev, vector_patch, bottleneck_value_img = MODEL_1.forward(patch_sample)
                
                loss_reconstruction_num = LOSS_FUNC_RECONSTRUCTION(data[0], decoded_output_num)
                loss_reconstruction_img = LOSS_FUNC_RECONSTRUCTION(patch_sample, decoded_output_img)

             #   pearson = PearsonCorrCoef()
                n_canonical_variables_img = bottleneck_value_img.size(dim=1)
                n_canonical_variables_num = bottleneck_value_num.size(dim=1)
                # corr_matrix (row : rna values, column : img values)
                corr_matrix = torch.empty(size=(n_canonical_variables_img,n_canonical_variables_num))
                # Replacement for pearson
                cos = CosineSimilarity(dim=0, eps=1e-6)



                # This calculation gives slightly different values (ab 5/6. nachkommastelle)
                counter_img = 0
                idx = 0
                
                if n_canonical_variables_num > 1 or n_canonical_variables_img > 1:
                    for idx_pair, pair in enumerate(itertools.product(bottleneck_value_img.clone().detach().unbind(dim=1), bottleneck_value_num.clone().detach().unbind(dim=1))):
                        img_c = pair[0].clone().detach()
                        num_c = pair[1].clone().detach()
                        corr_matrix[idx, counter_img] = abs(cos(img_c - img_c.mean(dim=0,keepdim=True), num_c - num_c.mean(dim=0,keepdim=True)))
                       # corr_matrix[idx,counter_img] = pearson(pair[0].clone().detach(),pair[1].clone().detach())
                        if idx % (n_canonical_variables_img - 1) == 0 and idx != 0:
                            idx = 0
                            counter_img += 1
                        else:
                            idx += 1
                        


                    non_nan_rows = []
                    # keep information of rows which have no NaN values
                    for row in range(corr_matrix.size(dim=0)):
                        if ~torch.any(torch.isnan(corr_matrix[row,:])):
                            non_nan_rows.append(row)


                    # Remove rows with NaN values
                    corr_matrix = corr_matrix[~torch.isnan(corr_matrix).any(axis=1)]

                   
                    loss_correlation = torch.mean(corr_matrix)
                else:
                    img_c = bottleneck_value_img.clone().detach().squeeze(dim=1)
                    num_c = bottleneck_value_num.clone().detach().squeeze(dim=1)
                    
                    loss_correlation = abs(cos(img_c - img_c.mean(dim=0,keepdim=True), num_c - num_c.mean(dim=0,keepdim=True)))
                  #  loss_correlation = pearson(bottleneck_value_img.clone().detach().squeeze(dim=1), bottleneck_value_num.clone().detach().squeeze(dim=1))


                alpha = 0.05
                beta = 0.4
                gamma = 0.55
                # Maximize correlation --> minimize negative correlation
                full_loss = alpha * loss_reconstruction_num + beta * loss_reconstruction_img + gamma * - loss_correlation 
                train_loss.append(float(full_loss))
                train_loss_reconstruction_rna.append(float(loss_reconstruction_num))
                train_loss_reconstruction_img.append(float(loss_reconstruction_img))
                train_loss_correlation.append(float(loss_correlation))



                if epoch == EPOCHS - 1:
                    # Calculate absoulute mean of ..
                    # Canonical variables for RNA features
                    ca_mean_rna = ((torch.mean(torch.abs(bottleneck_value_num), dim = 1)))
                    # Canonical variables for patch features
                    ca_mean_img = ((torch.mean(torch.abs(bottleneck_value_img), dim = 1)))

                    # Rename to names I used in architecture picture
                    feat_img = vector_patch # TODO : I take the vector patch in the encoding stage, maybe take the one in the decoding stage?
                    feat_img_trial = decoded_output_img_prev
                    feat_rna = data[0]

                    batch_size = feat_img.size(dim=0)

    
                    # Current patch in substep
                    coco_img = torch.empty(feat_img_trial.size(dim=1)) # amount features

                    # Current rna samples
                    coco_rna = torch.empty(feat_rna.size(dim=1)) # amount features

                    # Reshape feat_img so that we can iterate through amount of features to calculate each correlation coefficient

                    # This stays the same in the current SUBSTEP
                    feat_img_reshaped = torch.transpose(feat_img_trial,0,1) # amount features x batch size
                    for i, x in enumerate(feat_img_reshaped):
                        # correlation coefficient (loading) for each patch 
                        corr_coeff = abs(cos(x - x.mean(dim=0,keepdim=True), ca_mean_img - ca_mean_img.mean(dim=0,keepdim=True)))
                        corr_coeff_d = corr_coeff.clone().detach()
                        coco_img[i] = corr_coeff_d
                   
                    coco_img_storage.append(coco_img)
                     # Also store overall corrleation between RNA & image patch
                    correlations_storage.append(loss_correlation)
                    vector_patch_d = vector_patch.clone().detach()
                    feat_img_trial_d = feat_img_trial.clone().detach()

                   
                    pd.DataFrame(feat_img_trial_d.numpy()).to_csv("CCA-Project/coco_vals_2/patches/vector_patches_sample_batch_{}_image_patch_{}".format(step, substep))
                   

                
                    # This stays the same in the current STEP 
                        # We iterate over all patches (substeps) first, so this only needs to be calculated for each step ONCE
                    if substep == 99:
                        feat_rna_reshaped = torch.transpose(feat_rna,0,1)
                        for i, x in enumerate(feat_rna_reshaped):
                            # correlation coefficient (loading) for each rna feature 
                            corr_coeff = abs(cos(x - x.mean(dim=0,keepdim=True), ca_mean_rna - ca_mean_rna.mean(dim=0,keepdim=True)))
                            corr_coeff_d = corr_coeff.clone().detach()
                            coco_rna[i] = corr_coeff_d

                        # Also, in the last step for the current batch (last patch), we can store coco values

                        coco_img_total = torch.stack(coco_img_storage, dim = 1)
                        correlations_total = torch.stack(correlations_storage, dim = 0)
                       

                        # For the current batch of samples, we save the coco values, correlation overall aswell as patches stored in tensors
                        
                        pd.DataFrame(coco_rna.numpy()).to_csv("CCA-Project/coco_vals_2/rna_sample_batch_{}".format(step))
                        pd.DataFrame(coco_img_total.numpy()).to_csv("CCA-Project/coco_vals_2/img_patches_sample_batch_{}".format(step))
                        pd.DataFrame(correlations_total.numpy()).to_csv("CCA-Project/coco_vals_2/correlations_sample_batch_{}".format(step))
                       

                OPTIMIZER_1.zero_grad()
                OPTIMIZER_2.zero_grad()
                full_loss.backward()
                OPTIMIZER_1.step()
                OPTIMIZER_2.step()
            #    scheduler_1.step()
            #    scheduler_2.step()
            
            

                print(f'epoch {epoch} step {step} substep {substep} loss reconstruction num {loss_reconstruction_num} loss reconstruction img {loss_reconstruction_img} correlation {loss_correlation}')

        train_loss = HF.avg_per_epoch(train_loss,epoch)

        train_loss_reconstruction_rna = HF.avg_per_epoch(train_loss_reconstruction_rna,epoch)

        train_loss_reconstruction_img = HF.avg_per_epoch(train_loss_reconstruction_img,epoch)

        train_loss_correlation = HF.avg_per_epoch(train_loss_correlation,epoch)
                

            

    # Plot final loss after last epoch
    
    plt.plot(train_loss, label='train_loss', color= 'blue')
    plt.plot(train_loss_reconstruction_rna, label='rna loss', color='green')
    plt.plot(train_loss_reconstruction_img, label ='img loss', color='purple')
    plt.plot(train_loss_correlation, label='corr loss', color='red')
    plt.legend(loc='upper center')
  #  plt.show()
    plt.savefig('CCA-Project/training.png')

          
    # save CNN model
    torch.save(MODEL_1.state_dict(), 'CCA-Project/cnn_ae_model/cnn_model_new.pt')




if __name__ == '__main__':
    training_loop(8,600)






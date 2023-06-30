import os 
import wsi_preprocessing as pp
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
from torch.nn import Parameter
from torch.nn import ParameterList
from torch.utils.data import Dataset, DataLoader
import glob 


class CCA(torch.nn.Module):
    def __init__(self,n_channels_cnn):
        """
            patch_batch --> WSI patches
        """
        super(CCA,self).__init__()

 
        self.n_channels_cnn = n_channels_cnn
        self.model_parameter = ParameterList()

        # CNN structure
        # Encoder
        # img size 224x 224
        self.conv1 = Conv2d(in_channels=n_channels_cnn, out_channels=5,kernel_size=(5, 5),stride=1)
        
        # stride of 1 , thus 224-5+1 = 220 resulting size (220x220)
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=2)
        # 220:2 = 110 , resulting image (110x110)

        self.conv2 = Conv2d(in_channels=5, out_channels=1,kernel_size=(5, 5),stride=1)
        # 110-5+1 = 106 , 106x106
        self.maxpool2 = MaxPool2d(kernel_size=(2,2),stride=2)
        # 106:2 , 53x53


        # 1 x 53 x 53
        self.fc1 = Linear(in_features=2809, out_features=400)
        self.fc2 = Linear(in_features=400, out_features=100)

        # canonical variable
        self.fc3 = Linear(in_features=100, out_features=1)

        # Decoder
        self.fc3_d = Linear(in_features=1, out_features=100)
        self.fc2_d = Linear(in_features=100, out_features=400)
        self.fc1_d = Linear(in_features=400, out_features=2809)



        self.maxpool2_d = Upsample(scale_factor=2,mode='bilinear')
        self.conv2_d = ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=(5,5), stride=1)

        self.maxpool1_d = Upsample(scale_factor=2,mode='bilinear')
        self.conv1_d = ConvTranspose2d(in_channels=5, out_channels=n_channels_cnn, kernel_size=(5,5),stride=1)

       


        
        #TODO: pytorch only offers pseudo inverse of maxpooling (all non maximum values just set to 0 --> huge loss of data? 
        # ReLU activation won't affect these values anymore and they will stay 0) // use upsamling instead --> we can still have
        # learning effect here?
        #TODO : Later on, in the CCA structure, don't use encoder/decoder structure for the CNN-AE, use augmentation method CNN to find vector representation


    def nn_structure(self):
        """AE structures with reconstruction loss on decoder and 
           CCA loss (maximizing correlation between numerical & imaging)
           on bottleneck"""
        pass
        
    def nn_forward_pass(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        # Flatten every dimension but batch
        x, x_flattened = flatten(x,start_dim=1), flatten(x,start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # no activation
        x, x_bottleneck = self.fc3(x), self.fc3(x)

        x = self.fc3_d(x)
        x = F.relu(self.fc2_d(x))
        x = F.relu(self.fc1_d(x))
    

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

        return x, x_flattened, x_bottleneck
        


class PatchesDataset(Dataset):
    def __init__(self):
        self.img_path = 'TCGA-05-4249-01Z-00-DX1.9fce0297-cc19-4c04-872c-4466ee4024db.svs_mpp-0.5_crop47-90_files/1' # TODO : change to 0 for real program
        # Store imgs here
        self.data = []
        directory = os.fsencode(self.img_path)
        for patch in os.listdir(directory):
            absolute_path = self.img_path + '/' + os.fsdecode(patch)
            pixel_vals = torch.tensor(Image.open(absolute_path).getdata())

            # Assign RGB channels
            channel_R = pixel_vals[:,0]
            channel_G = pixel_vals[:,1]
            channel_B = pixel_vals[:,2]

                
            img = torch.stack((channel_R.reshape(224,224),channel_G.reshape(224,224),channel_B.reshape(224,224)),dim=0)

            self.data.append(img)

    def __len__(self):
        # Amount of patches
        return len(self.data)
    

    def __getitem__(self,idx):
        img = self.data[idx]
        return img




            



def training_loop(batch_size,epochs):
    train_loss = []
    EPOCHS = epochs
    MODEL = CCA(n_channels_cnn=3)
    OPTIMIZER = SGD(MODEL.parameters(), lr=0.001, momentum=0.9)
    LOSS_FUNC = MSELoss()
    

    dataset = PatchesDataset()
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    for i in range(EPOCHS):
        for c, data in enumerate(data_loader):

            data = data.type(torch.FloatTensor)
            
            decoded_output, bottleneck_value = MODEL.nn_forward_pass(data)
            loss = LOSS_FUNC(data,decoded_output)

            train_loss.append(float(loss))

            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            print(f'epoch {i} step {c} loss {loss}')



            






if __name__ == '__main__':
    # if, for instance, CMU-1.svs is in your current directory ("."):
  #  slides = pp.list_slides("./Slides")
  #  pp.save_slides_mpp_otsu(slides, "slides_mpp_otsu.csv")

    # this may take a few minutes (depending on your local machine, of course)
  #  pp.run_tiling("slides_mpp_otsu.csv", "tiles.csv")

  #  pp.calculate_filters("slides_mpp_otsu.csv", "", "tiles_filters.csv")


    """
    example_patch_1 = Image.open('TCGA-05-4249-01Z-00-DX1.9fce0297-cc19-4c04-872c-4466ee4024db.svs_mpp-0.5_crop47-90_files/0/17_32.png')
    example_patch_2 = Image.open('TCGA-05-4249-01Z-00-DX1.9fce0297-cc19-4c04-872c-4466ee4024db.svs_mpp-0.5_crop47-90_files/0/0_24.png')

    pixel_vals_1 = torch.tensor(example_patch_1.getdata())
    pixel_vals_2 = torch.tensor(example_patch_2.getdata())




    # Assign RGB channels
    channel_R_1 = pixel_vals_1[:,0]
    channel_G_1 = pixel_vals_1[:,1]
    channel_B_1 = pixel_vals_1[:,2]

    img_1 = torch.stack((channel_R_1.reshape(224,224),channel_G_1.reshape(224,224),channel_B_1.reshape(224,224)),dim=0)


    # Assign RGB channels
    channel_R_2 = pixel_vals_2[:,0]
    channel_G_2 = pixel_vals_2[:,1]
    channel_B_2 = pixel_vals_2[:,2]

    img_2 = torch.stack((channel_R_2.reshape(224,224),channel_G_2.reshape(224,224),channel_B_2.reshape(224,224)),dim=0)


    # Set as batch
    img = torch.stack((img_1,img_2), dim=0)

    """

    training_loop(5,10)

    


    







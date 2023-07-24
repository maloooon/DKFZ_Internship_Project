import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import SGD 
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os 
import pandas as pd 
from matplotlib import pyplot as plt
from torch import nn
import HelperFunctions as HF 



class numAE(torch.nn.Module):
    def __init__(self, in_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_size = in_size 


        self.encoder = nn.Sequential(
            nn.Linear(self.in_size,250),
            nn.ReLU(),
            nn.Linear(250,64),
          #  nn.ReLU(),
            nn.Linear(64,2),
           # nn.Linear(16,8)
        )


        self.decoder = nn.Sequential(
           # nn.Linear(8,16),
            nn.Linear(2,64),
           # nn.ReLU(),
            nn.Linear(64,250),
            nn.ReLU(),
            nn.Linear(250,self.in_size)
        )


    def forward(self,x):
        x = self.encoder(x)
        bottleneck = x.clone().detach()
        x = self.decoder(x)

        return x, bottleneck

    
   

class numDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data_path = 'Slides/data/rnaseq.csv' # TODO : just for rnaseq right now
        data_df = pd.read_csv(self.data_path)
        self.data = torch.from_numpy(data_df.values).float()
        
    
    def __len__(self):
        # Get amount of samples (needed for __getitem__) and features
        return self.data.size(dim=0)
    
    def __numfeats__(self):
        # Get number of features
        return self.data.size(dim=1)
    

    
    def __getitem__(self, index):
        return self.data[index,:]



def training_loop(epochs,batch_size):
    train_loss = []
    EPOCHS = epochs
    
    
    LOSS_FUNC = MSELoss()
    

    dataset = numDataset()
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)


    MODEL = numAE(in_size=dataset.__numfeats__())
    OPTIMIZER = Adam(MODEL.parameters(), lr= 0.001)
    for epoch in range(EPOCHS):
        for c, data in enumerate(data_loader):

            data = data.type(torch.FloatTensor)
            
            decoded_output, bottleneck_value = MODEL.forward(data)
            loss = LOSS_FUNC(data,decoded_output)

            train_loss.append(float(loss))

            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            print(f'epoch {epoch} step {c} loss {loss}')
    
    # Plot final loss after lass epoch

        train_loss = HF.avg_per_epoch(train_loss,epoch)
    
    plt.plot(train_loss, label='train_loss')
    plt.legend()
    plt.show()

    

if __name__ == '__main__':
    training_loop(200,8)







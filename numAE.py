import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import SGD 
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
import os 
import pandas as pd 
from matplotlib import pyplot as plt



class numAE(torch.nn.Module):
    def __init__(self, in_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_size = in_size 


        # Encoding
       # self.fc1 = Linear(self.in_size,1600)
        self.fc2 = Linear(self.in_size,500)
        self.fc3 = Linear(500,250)
        self.fc4 = Linear(250,64)
        self.fc5 = Linear(64,1)
       # self.fc6 = Linear(100,64)
       # self.fc7 = Linear(64,32)
       # self.fc8 = Linear(32,8)
       # self.fc9 = Linear(8,4)
        # canonical variable
       # self.fc10 = Linear(4,1)

        # Decoding
       # self.fc10_d = Linear(1,4)
       # self.fc9_d = Linear(4,8)
       # self.fc8_d = Linear(8,32)
       # self.fc7_d = Linear(32,64)
       # self.fc6_d = Linear(64,100)
        self.fc5_d = Linear(1,64)
        self.fc4_d = Linear(64,250)
        self.fc3_d = Linear(250,500)
        self.fc2_d = Linear(500,self.in_size)
     #   self.fc1_d = Linear(1600,self.in_size)



    
    def nn_forward_pass(self,x):
        
     #   x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
     #   x = F.relu(self.fc5(x))
     #   x = F.relu(self.fc6(x))
     #   x = F.relu(self.fc7(x))
     #   x = F.relu(self.fc8(x))
     #   x = F.relu(self.fc9(x))
        # no activation
        x, x_bottleneck = self.fc5(x), self.fc5(x)
      #  x, x_bottleneck = self.fc10(x), self.fc10(x)
      #  x = self.fc10_d(x)

      #  x = F.relu(self.fc9_d(x))
      #  x = F.relu(self.fc8_d(x))
      #  x = F.relu(self.fc7_d(x))
      #  x = F.relu(self.fc6_d(x))
        x = F.relu(self.fc5_d(x))
        x = F.relu(self.fc4_d(x))
        x = F.relu(self.fc3_d(x))
        x = F.relu(self.fc2_d(x))
     #   x = F.relu(self.fc1_d(x))


        return x , x_bottleneck
    

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
    OPTIMIZER = SGD(MODEL.parameters(), lr=0.0001, momentum=0.9)
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
    
    # Plot final loss after lass epoch
    
    plt.plot(train_loss, label='train_loss')
    plt.legend()
    plt.show()

    

if __name__ == '__main__':
    training_loop(200,64)







import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import SGD 
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
import os 
import pandas as pd 



class numAE(torch.nn.Module):
    def __init__(self, in_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_size = in_size 


        # Encoding
        self.fc1 = Linear(self.in_size,400)
        self.fc2 = Linear(400,200)
        self.fc3 = Linear(200,100)

        # Decoding
        self.fc3_d = Linear(100,200)
        self.fc2_d = Linear(200,400)
        self.fc1_d = Linear(400,self.in_size)



    
    def nn_forward_pass(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x, x_bottleneck = F.relu(self.fc3(x)), F.relu(self.fc3(x))
        
        x = F.relu(self.fc3_d(x))
        x = F.relu(self.fc2_d(x))
        x = F.relu(self.fc1_d(x))


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



def training_loop(epochs):
    train_loss = []
    EPOCHS = epochs
    
    
    LOSS_FUNC = MSELoss()
    

    dataset = numDataset()
    data_loader = DataLoader(dataset,batch_size=5,shuffle=True)


    MODEL = numAE(in_size=dataset.__numfeats__())
    OPTIMIZER = SGD(MODEL.parameters(), lr=0.001, momentum=0.9)
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
    training_loop(5)







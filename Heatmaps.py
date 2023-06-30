import torch
from torch.nn import Linear
from torch.nn import Upsample
from torch.nn import ConvTranspose2d
from math import sqrt
import torch.nn.functional as F



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

    def nn_forward_pass(self,x):
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




if __name__ == '__main__':
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
    #model.eval()
    model.load_state_dict(model_dict)


    



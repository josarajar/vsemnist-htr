from torch import nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,height,nlabels,prob): #Nlabels will be 47 in our case
        super().__init__()
        
        self.height = height
        self.nlabels = nlabels
        # convolutional layer (sees 28x28x1 image tensor)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, 
                               kernel_size=3, stride=1, padding=1)
        
        # convolutional layer (sees 12x12x16 tensor)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        
        self.conv3 = nn.Conv2d(16, 32, 7, stride=7, padding=0)
        
        # Max pool layer
        self.pool = nn.MaxPool2d(2, 2)

        # Linear layers
        self.linear1 = nn.Linear(32,256)
        
        self.linear2 = nn.Linear(256, nlabels)
    
        self.activation = nn.ReLU()
        
        self.logsoftmax = nn.LogSoftmax(dim=1) 

        self.BN_1 = nn.BatchNorm2d(num_features=6)

        self.BN_2 = nn.BatchNorm2d(num_features=16)
        
        self.BN_3 = nn.BatchNorm2d(num_features=32)

        self.BN_4 = nn.BatchNorm1d(256)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=prob)
        
    def forward(self, x):
        # Pass the input tensor through the CNN operations
        input_width = x.shape[3]
        output_width = int(input_width/self.height)
        x = self.conv1(x) 
        x = self.BN_1(x)
        x = self.activation(x) 
        x = self.pool(x)
        x = self.dropout(x) 
        x = self.conv2(x)
        x = self.BN_2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x) 
        x = self.conv3(x)
        x = self.BN_3(x)
        x = self.activation(x)
        x = self.dropout(x)
        # Pass the tensor through the Dense Layers
        x = x.squeeze(2)
        x = x.transpose(1,2)
        x = x.reshape(-1,32)
        x = self.linear1(x)
        x = self.BN_4(x)
        x = self.activation(x)
        x = self.dropout(x) 
        x = self.linear2(x)
        x = x.view(-1,output_width, self.nlabels)
        x = x.transpose(1,2)
        x = self.logsoftmax(x) 
        return x
    
    
class STN_CNN(nn.Module):
    def __init__(self,height,nlabels,prob): #Nlabels will be 47 in our case
        super().__init__()
        
        self.height = height
        self.nlabels = nlabels
        # convolutional layer (sees 28x28x1 image tensor)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, 
                               kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        
        self.conv3 = nn.Conv2d(16, 32, 7, stride=7, padding=0)
        
        # Max pool layer
        self.pool = nn.MaxPool2d(2, 2)

        # Linear layers
        self.linear1 = nn.Linear(32,256)
        
        self.linear2 = nn.Linear(256, nlabels)
    
        self.activation = nn.ReLU()
        
        self.logsoftmax = nn.LogSoftmax(dim=1) 

        self.BN_1 = nn.BatchNorm2d(num_features=6)

        self.BN_2 = nn.BatchNorm2d(num_features=16)
        
        self.BN_3 = nn.BatchNorm2d(num_features=32)

        self.BN_4 = nn.BatchNorm1d(256)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=prob)
        
        self.localization = [None]*4
        self.fc_loc = [None]*4
        for p_stn in range(4):
            # Spatial transformer localization-network
            self.localization[p_stn] = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=3, padding=1),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(10, 10, kernel_size=3, padding=1),
                nn.MaxPool2d(7 , stride=7),
                nn.ReLU(True),
            )

            # Regressor for the 3 * 2 affine matrix
            self.fc_loc[p_stn] = nn.Sequential(
                nn.Linear(10 * 1 * 10, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc[p_stn][2].weight.data.zero_()
            self.fc_loc[p_stn][2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        out=[None]*4
        for p_stn in range(4):
            xs = self.localization[p_stn](x)
            xs = xs.view(-1, 10 * 1 * 10)
            theta = self.fc_loc[p_stn](xs)
            theta = theta.view(-1, 2, 3)

            grid = F.affine_grid(theta, x.size())
            out[p_stn] = F.grid_sample(x, grid)

        return torch.cat(out,1)

        
    def forward(self, x):
        # Pass the input tensor through the CNN operations
        input_width = x.shape[3]
        output_width = int(input_width/self.height)
        x = self.stn(x) # STN layer
        x = self.conv1(x) 
        x = self.BN_1(x)
        x = self.activation(x) 
        x = self.pool(x)
        x = self.dropout(x) 
        x = self.conv2(x)
        x = self.BN_2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x) 
        x = self.conv3(x)
        x = self.BN_3(x)
        x = self.activation(x)
        x = self.dropout(x)
        # Pass the tensor through the Dense Layers
        x = x.squeeze(2)
        x = x.transpose(1,2)
        x = x.reshape(-1,32)
        x = self.linear1(x)
        x = self.BN_4(x)
        x = self.activation(x)
        x = self.dropout(x) 
        x = self.linear2(x)
        x = x.view(-1,output_width, self.nlabels)
        x = x.transpose(1,2)
        x = self.logsoftmax(x) 
        return x
    
class CNN_6(nn.Module):
    def __init__(self,height,nlabels,prob): #Nlabels will be 10 in our case
        super().__init__()
        
        self.height = height
        self.nlabels = nlabels
        # convolutional layer (sees 28x28x1 image tensor)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, 
                               kernel_size=5, stride=1, padding=2)
        
        # convolutional layer (sees 12x12x16 tensor)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        
        # Max pool layer
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.pool7 = nn.MaxPool2d(7, 7)

        # Linear layers
        self.linear1 = nn.Linear(128,64)
        
        self.linear2 = nn.Linear(64, nlabels)
    
        self.activation = nn.ReLU()
        
        self.logsoftmax = nn.LogSoftmax(dim=1) 

        self.BN_1 = nn.BatchNorm2d(num_features=16)

        self.BN_2 = nn.BatchNorm2d(num_features=64)
        
        self.BN_3 = nn.BatchNorm2d(num_features=128)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=prob)
        
    def forward(self, x):
        # Pass the input tensor through the CNN operations
        input_width = x.shape[3]
        output_width = int(input_width/self.height)
        
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.BN_1(x)
        x = self.activation(x) 
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.BN_2(x)
        x = self.activation(x) 
        x = self.pool2(x)
        
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        x = self.BN_3(x)
        x = self.activation(x) 
        x = self.pool7(x)
  
        # Pass the tensor through the Dense Layers
        x = x.squeeze(2)
        x = x.transpose(1,2)
        x = x.reshape(-1,128)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x) 
        x = self.linear2(x)
        x = x.view(-1,output_width, self.nlabels)
        x = x.transpose(1,2)
        x = self.logsoftmax(x) 
        return x

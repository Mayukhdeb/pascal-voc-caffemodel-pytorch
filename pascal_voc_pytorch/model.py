import torch.nn as nn

class PascalVoCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = nn.LocalResponseNorm(5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)
        self.norm2 = nn.LocalResponseNorm(5, alpha=9.999999747378752e-05, beta=0.75, k=1.0)    

        self.drop6 = nn.Dropout(p=0.5, inplace=False)    
        self.drop7 = nn.Dropout(p=0.5, inplace=False)

        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2)

        self.relu1 = nn.ReLU()  
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.fc6     = nn.Linear(in_features=9216, out_features=4096, bias=True)
        self.fc7     = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc8voc =  nn.Linear(in_features=4096, out_features=20, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.norm1(x)

        '''
        Saw a mudcrab the other day. Horrible creatures
        '''

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.pool5(x)

        x = x.reshape(x.size(0), -1)  ## flatten to (1, 9216)

        x = self.fc6(x)
        x = self.relu6(x)       
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.relu7(x)  
        x = self.drop7(x)

        x = self.fc8voc(x)

        return x
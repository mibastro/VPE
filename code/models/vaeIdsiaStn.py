# This code is modified from the repository
# https://github.com/bhpfelix/Variational-Autoencoder-PyTorch

from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
   
    def forward(self, x):
        numel = x.numel() / x.shape[0]
        return x.view(-1, int(numel)) 

def convNoutput(convs, input_size): # predict output size after conv layers
    input_size = int(input_size)
    input_channels = convs[0][0].weight.shape[1] # input channel
    output = torch.Tensor(1, input_channels, input_size, input_size)
    with torch.no_grad():
        for conv in convs:
            output = conv(output)
    return output.numel(), output.shape

class stn(nn.Module):
    def __init__(self, input_channels, input_size, params):
        super(stn, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
                    nn.Conv2d(input_channels, params[0], kernel_size=5, stride=1, padding=2),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(params[0], params[1], kernel_size=5, stride=1, padding=2),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )

        out_numel, out_size = convNoutput([self.conv1, self.conv2], input_size/2)
        # set fc layer based on predicted size
        self.fc = nn.Sequential(
                View(),
                nn.Linear(out_numel, params[2]),
                nn.ReLU()
                )
        self.classifier = classifier = nn.Sequential(
                View(),
                nn.Linear(params[2], 6) # affine transform has 6 parameters
                )
        # initialize stn parameters (affine transform)
        self.classifier[1].weight.data.fill_(0)
        self.classifier[1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def localization_network(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x


    def forward(self, x):
        theta = self.localization_network(x)
        theta = theta.view(-1,2,3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class VAEIdsia(nn.Module):
    def __init__(self, nc, input_size, latent_variable_size=300, cnn_chn=[100, 150, 250], param1=None, param2=None, param3=None):
        super(VAEIdsia, self).__init__()

        self.cnn_chn = cnn_chn
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

        self.input_size = input_size
        self.nc = nc
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, self.cnn_chn[0], 7, 2, 3) # inchn, outchn, kernel, stride, padding, dilation, groups
        self.bn1 = nn.BatchNorm2d(self.cnn_chn[0])

        self.e2 = nn.Conv2d(self.cnn_chn[0], self.cnn_chn[1], 4, 2, 1) # 1/4
        self.bn2 = nn.BatchNorm2d(self.cnn_chn[1])

        self.e3 = nn.Conv2d(self.cnn_chn[1], self.cnn_chn[2], 4, 2, 1) # 1/8
        self.bn3 = nn.BatchNorm2d(self.cnn_chn[2])

        self.fc1 = nn.Linear(int(input_size/8*input_size/8*self.cnn_chn[2]), latent_variable_size)
        self.fc2 = nn.Linear(int(input_size/8*input_size/8*self.cnn_chn[2]), latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, int(input_size/8*input_size/8*self.cnn_chn[2]))

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2) # 8 -> 16
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(self.cnn_chn[2], self.cnn_chn[1], 3, 1)
        self.bn6 = nn.BatchNorm2d(self.cnn_chn[1], 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2) # 16 -> 32
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(self.cnn_chn[1], self.cnn_chn[0], 3, 1)
        self.bn7 = nn.BatchNorm2d(self.cnn_chn[0], 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2) # 32 -> 64
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(self.cnn_chn[0], 3, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if param1 is not None:
            self.stn1 = stn(3, self.input_size, param1)
        if param2 is not None:
            self.stn2 = stn(self.cnn_chn[0], self.input_size/2, param2)
        if param3 is not None:
            self.stn3 = stn(self.cnn_chn[1], self.input_size/4, param3)


    def encode(self, x):
        if self.param1 is not None:
            x = self.stn1(x)

        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        if self.param2 is not None:
            h1 = self.stn2(h1)

        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        if self.param3 is not None:
            h2 = self.stn3(h2)

        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = h3.view(-1, int(self.input_size/8*self.input_size/8*self.cnn_chn[2]))

        return self.fc1(h4), self.fc2(h4), x

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        # h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h1 = h1.view(-1, self.cnn_chn[2], int(self.input_size/8), int(self.input_size/8))
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        return self.sigmoid(self.d4(self.pd3(self.up3(h3))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar, xstn = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar, xstn

    def init_params(self, net):
        print('Loading the model from the file...')
        net_dict = self.state_dict()
        if isinstance(net, dict):
            pre_dict = net
        else:
            pre_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pre_dict = {k: v for k, v in pre_dict.items() if (k in net_dict)} # for fs net
        net_dict.update(pre_dict)
        # 3. load the new state dict
        self.load_state_dict(net_dict)
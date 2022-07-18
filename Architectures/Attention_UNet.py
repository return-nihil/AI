import torch
import torch.nn as nn


class Conv_Block(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Conv_Block,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )


    def forward(self, x):

        out = self.conv(x)

        return out



class Up_Conv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Up_Conv,self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_chan, out_chan, kernel_size = 3, padding = 1),
		    nn.BatchNorm2d(out_chan),
			nn.ReLU()
        )


    def forward(self, x):

        out = self.up(x)

        return out



class Attention_Block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_Block,self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1),
            nn.BatchNorm2d(F_int)
        )

        self.relu = nn.ReLU()

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        out = x * psi

        return out



class Dense_Layer(nn.Module): 
    def __init__(self, in_chan, out_chan):
      super(Dense_Layer, self).__init__()

      self.dense = nn.Sequential(
        nn.Linear(in_chan, out_chan), 
        nn.BatchNorm1d(out_chan), 
        nn.ReLU(), 
        nn.Dropout(p = DROPOUT)
    ) 
      
    def forward(self, x):

        out = self.dense(x)

        return out



class AttU_Net(nn.Module):
    def __init__(self, in_chan, unet_out_chan):
        super(AttU_Net, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv1 = Conv_Block(in_chan, 64)
        self.conv2 = Conv_Block(64, 128)
        self.conv3 = Conv_Block(128, 256)
        self.conv4 = Conv_Block(256, 512)
        self.conv5 = Conv_Block(512, 1024)

        self.up5 = Up_Conv(1024, 512)
        self.att5 = Attention_Block(F_g = 512, F_l = 512, F_int = 256)
        self.up_conv5 = Conv_Block(1024, 512)

        self.up4 = Up_Conv(512, 256)
        self.att4 = Attention_Block(F_g = 256, F_l = 256, F_int = 128)
        self.up_conv4 = Conv_Block(512, 256)
        
        self.up3 = Up_Conv(256, 128)
        self.att3 = Attention_Block(F_g = 128, F_l = 128, F_int = 64)
        self.up_conv3 = Conv_Block(256, 128)
        
        self.up2 = Up_Conv(128, 64)
        self.att2 = Attention_Block(F_g = 64, F_l = 64, F_int = 32)
        self.up_conv2 = Conv_Block(128, 64)

        self.conv_1x1 = nn.Conv2d(64, unet_out_chan, kernel_size = 1)


    def forward(self, x):

        # ENCODING
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # DECODING + ATTENTION
        d5 = self.up5(x5)
        x4 = self.att5(d5, x4)
        d5 = torch.cat((x4, d5), dim = 1)        
        d5 = self.up_conv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(d4, x3)
        d4 = torch.cat((x3, d4), dim = 1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(d3, x2)
        d3 = torch.cat((x2, d3), dim = 1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(d2, x1)
        d2 = torch.cat((x1, d2), dim = 1)
        d2 = self.up_conv2(d2)

        out = self.conv_1x1(d2)

        return out
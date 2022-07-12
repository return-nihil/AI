import torch
import torch.nn as nn




class Conv_Block(nn.Module):
  def __init__(self, in_chan, out_chan, **kwargs):
    super(Conv_Block, self).__init__()

    self.conv = nn.Conv2d(in_chan, out_chan, **kwargs)
    self.relu = nn.ReLU()
    self.batchnorm = nn.BatchNorm2d(out_chan)


  def forward(self, x):

    x = self.conv(x)
    x = self.batchnorm(x)
    out = self.relu(x)

    return out



class Incept_Block(nn.Module):
  def __init__(self, in_chan, out_chan_1, reduct_3, out_chan_3, reduct_5, out_chan_5, out_chan_max):
    super(Incept_Block, self).__init__()

    self.module1 = Conv_Block(in_chan, out_chan_1, kernel_size = 1)

    self.module2 = nn.Sequential(
        Conv_Block(in_chan, reduct_3, kernel_size = 1),
        Conv_Block(reduct_3, out_chan_3, kernel_size = 3, padding = 1)
    )

    self.module3 = nn.Sequential(
        Conv_Block(in_chan, reduct_5, kernel_size = 1),
        Conv_Block(reduct_5, out_chan_5, kernel_size = 5, padding = 2)
    )

    self.module4 = nn.Sequential(
        nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
        Conv_Block(in_chan, out_chan_max, kernel_size = 1)
    )


  def forward(self, x):

    x1 = self.module1(x)
    x2 = self.module2(x)
    x3 = self.module3(x)
    x4 = self.module4(x)

    out = torch.cat([x1, x2, x3, x4], 1)

    return out



class InceptionNet(nn.Module):
  def __init__(self, in_chan, num_class):
    super(InceptionNet, self).__init__()

    self.conv1 = Conv_Block(in_chan, 64, kernel_size = 7, stride = 2, padding = 3)
    self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    self.conv2 = Conv_Block(64, 192, kernel_size = 3, stride = 1, padding = 1)
    self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    self.inception1 = Incept_Block(192, 64, 96, 128, 16, 32, 32)
    self.inception2 = Incept_Block(256, 128, 128, 192, 32, 96, 64)

    self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    self.inception3 = Incept_Block(480, 192, 96, 208, 16, 48, 64)
    self.inception4 = Incept_Block(512, 160, 112, 224, 24, 64, 64)
    self.inception5 = Incept_Block(512, 128, 128, 256, 24, 64, 64)
    self.inception6 = Incept_Block(512, 112, 144, 288, 32, 64, 64)
    self.inception7 = Incept_Block(528, 256, 160, 320, 32, 128, 128)

    self.maxpool4 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    self.inception8 = Incept_Block(832, 256, 160, 320, 32, 128, 128)
    self.inception9 = Incept_Block(832, 384, 192, 384, 48, 128, 128)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(p = DROPOUT)

    self.flat = nn.Flatten()
    self.fully1 = nn.Sequential(nn.Linear(1024, 128),
                                nn.ReLU())
    self.fully2 = nn.Linear(128, num_class)


  def forward(self, x):

    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.maxpool2(x)

    x = self.inception1(x)
    x = self.inception2(x)

    x = self.maxpool3(x)

    x = self.inception3(x)
    x = self.inception4(x)
    x = self.inception5(x)
    x = self.inception6(x)
    x = self.inception7(x)

    x = self.maxpool4(x)

    x = self.inception8(x)
    x = self.inception9(x)

    x = self.avgpool(x)
    x = self.dropout(x)

    x = self.flat(x)
    x = self.fully1(x)
    out = self.fully2(x)

    return out

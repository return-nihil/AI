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



class Lin_Block(nn.Module):
  def __init__(self, in_chan, out_chan):
    super(Lin_Block, self).__init__()

    self.lin = nn.Linear(in_chan, out_chan)
    self.relu = nn.ReLU()


  def forward(self, x):

    x = self.lin(x)
    out = self.relu(x)

    return out



class Classifier(nn.Module):
  def __init__(self, num_class):
    super(Classifier, self).__init__()

    self.flat = nn.Flatten()
    self.fully1 = Lin_Block(9216, 1024)
    self.fully2 = Lin_Block(1024, 128)
    self.out = nn.Linear(128, num_class)


  def forward(self, x):

    x = self.flat(x)
    x = self.fully1(x)
    x = self.fully2(x)
    out = self.out(x)

    return out



class AlexNet(nn.Module):

  def __init__(self, in_chan, num_class):
    super(AlexNet, self).__init__()

    self.conv1 = Conv_Block(in_chan, 96, kernel_size = 11, stride = 4)
    self.conv2 = Conv_Block(96, 256, kernel_size = 5, padding = 2)
    self.conv3 = Conv_Block(256, 384, kernel_size = 3, padding = 1)
    self.conv4 = Conv_Block(384, 384, kernel_size = 3, padding = 1)
    self.conv5 = Conv_Block(384, 256, kernel_size = 3, padding = 1)

    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2)

    self.avgpool = nn.AvgPool2d(kernel_size = 1)
    self.dropout = nn.Dropout(p = DROPOUT)
    
    self.classifier = Classifier(num_class)


  def forward(self, x):

    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.maxpool(x)
    x = self.conv3(x)
    x = self.conv4(x)

    x = self.conv5(x)
    x = self.maxpool(x)

    x = self.avgpool(x)
    x = self.dropout(x)

    out = self.classifier(x)

    return out

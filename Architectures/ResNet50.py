import torch
import torch.nn as nn



class Conv_Block(nn.Module):
  def __init__(self, in_chan, out_chan, **kwargs):
    super(Conv_Block, self).__init__()

    self.conv = nn.Conv2d(in_chan, out_chan, **kwargs)
    self.batchnorm = nn.BatchNorm2d(out_chan)


  def forward(self, x):

    x = self.conv(x)
    out = self.batchnorm(x)

    return out



class Res_Block(nn.Module):

  def __init__(self, in_chan, out_chan, identity_downsample = None, stride = 1):
    super(Res_Block, self).__init__()

    self.expansion = 4

    self.conv1 = Conv_Block(in_chan, out_chan, kernel_size = 1)
    self.conv2 = Conv_Block(out_chan, out_chan, kernel_size = 3, stride = stride, padding = 1)
    self.conv3 = Conv_Block(out_chan, out_chan * self.expansion, kernel_size = 1)

    self.relu = nn.ReLU()
    self.identity_downsample = identity_downsample


  def forward(self, x):

    identity = x

    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)

    if self.identity_downsample != None:
      identity = self.identity_downsample(identity)

    x += identity
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
    self.fully1 = Lin_Block(2048, 512)
    self.fully2 = Lin_Block(512, 64)
    self.out = nn.Linear(64, num_class)


  def forward(self, x):

    x = self.flat(x)
    x = self.fully1(x)
    x = self.fully2(x)
    out = self.out(x)

    return out



class ResNet50(nn.Module):

  def __init__(self, in_chan, num_class):
    super(ResNet50, self).__init__()

    self.chans = 64
    self.layers = [3, 4, 6, 3] 

    self.conv = Conv_Block(in_chan, 64, kernel_size = 7, stride = 2, padding = 3)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    self.layer1 = self._make_residual(Res_Block, self.layers[0], out_chan = 64, stride = 1)
    self.layer2 = self._make_residual(Res_Block, self.layers[1], out_chan = 128, stride = 2)
    self.layer3 = self._make_residual(Res_Block, self.layers[2], out_chan = 256, stride = 2)
    self.layer4 = self._make_residual(Res_Block, self.layers[3], out_chan = 512, stride = 2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(p = DROPOUT)

    self.classifier = Classifier(num_class)


  def _make_residual(self, block, num_res_blocks, out_chan, stride):
    identity_downsample = None
    layers = []

    if stride != 1 or self.chans != out_chan * 4:
      identity_downsample = Conv_Block(self.chans, out_chan * 4, kernel_size = 1, stride = stride)
      
    layers.append(block(self.chans, out_chan, identity_downsample, stride)) 
    self.chans = out_chan * 4

    for i in range(num_res_blocks -1):
      layers.append(block(self.chans, out_chan))

    return nn.Sequential(*layers)


  def forward(self, x):

    x = self.conv(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = self.dropout(x)

    out = self.classifier(x)

    return out

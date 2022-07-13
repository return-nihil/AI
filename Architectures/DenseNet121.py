import torch
import torch.nn as nn




class Bottleneck(nn.Module):
  def __init__(self, in_chan, bottleneck_size, growth_rate):
    super(Bottleneck, self).__init__()
    self.btgr = bottleneck_size * growth_rate

    self.conv1x1 = nn.Sequential(nn.BatchNorm2d(in_chan),
                                 nn.ReLU(),
                                 nn.Conv2d(in_chan, self.btgr, kernel_size = 1))

    self.conv3x3 = nn.Sequential(nn.BatchNorm2d(self.btgr),
                                 nn.ReLU(),
                                 nn.Conv2d(self.btgr, growth_rate, kernel_size = 3, padding = 1))


  def forward(self, input):

    x = self.conv1x1(input)
    x = self.conv3x3(x)
    out = torch.cat([x, input], dim = 1)

    return out



class Dense_Block(nn.Module):
  def __init__(self, in_chan, num_layers, bottleneck_size, growth_rate):
    super(Dense_Block, self).__init__()

    layers = []
    for layer_idx in range(num_layers):
        layers.append(Bottleneck(in_chan = in_chan + layer_idx * growth_rate,           
                                bottleneck_size = bottleneck_size,
                                growth_rate = growth_rate)
        )

    self.single_block = nn.Sequential(*layers)


  def forward(self, x):

    out = self.single_block(x)

    return out



class Transition(nn.Module):
  def __init__(self, in_chan, out_chan):
    super(Transition, self).__init__()

    self.conv1x1 = nn.Sequential(nn.BatchNorm2d(in_chan),
                                 nn.ReLU(),
                                 nn.Conv2d(in_chan, out_chan, kernel_size = 1))

    self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2) 


  def forward(self, x):

    x = self.conv1x1(x)
    out = self.avgpool(x)

    return out



class DenseNet121(nn.Module):
  def __init__(self, in_chans, num_class):
    super().__init__()

    self.layers = [6, 12, 24, 16]
    self.bottleneck_size = 2
    self.growth_rate = 32
    self.hidden_chans = self.growth_rate * self.bottleneck_size
       
    self.inconv = nn.Conv2d(in_chans, self.hidden_chans, kernel_size = 3, padding = 1)

    self.blocks = self._make_dense(Dense_Block, Transition)

    self.last_layer = nn.Sequential(nn.BatchNorm2d(self.hidden_chans),
                                   nn.ReLU(),
                                   nn.AdaptiveAvgPool2d((1,1)))
    
    self.dropout = nn.Dropout(p = DROPOUT)

    self.classifier = nn.Sequential(nn.Flatten(),
                                    nn.Linear(self.hidden_chans, 64),
                                    nn.Linear(64, num_class))


  def _make_dense(self, block, transition):

    blocks = []
    for index, num_layers in enumerate(self.layers):
        blocks.append(
            block(in_chan = self.hidden_chans,
                        num_layers = num_layers,
                        bottleneck_size = self.bottleneck_size,
                        growth_rate = self.growth_rate)
        )
        self.hidden_chans = self.hidden_chans + num_layers * self.growth_rate
        if index < (len(self.layers) - 1):
            blocks.append(
                transition(in_chan = self.hidden_chans,
                                out_chan = self.hidden_chans // 2))
            self.hidden_chans = self.hidden_chans // 2

    return nn.Sequential(*blocks)


  def forward(self, x):

    x = self.inconv(x)
    x = self.blocks(x)
    x = self.last_layer(x)
    x = self.dropout(x)
    out = self.classifier(x)

    return out

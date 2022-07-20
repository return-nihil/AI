import torch
import torch.nn as nn


class SimpleSVM(nn.Module):
    def __init__(self, in_chans, num_class, dropout): # in_chans & dropout not used!! Just for immediate integration.
        super(SimpleSVM).__init__()

        self.flat = nn.Flatten()

        self.linear1 = nn.Sequential(nn.BatchNorm1d(196608), # Fix 3x256x256 
                                     nn.Linear(196608, 64))

        self.classifier = nn.Sequential(nn.BatchNorm1d(64),
                                     nn.Linear(64, num_class))     

        self.relu = nn.ReLU()


    def forward(self, input):

        x = self.flat(input)
        x = self.linear1(x)
        x = self.relu(x)

        out = self.classifier(x)

        return out




class SVM(nn.Module):
    def __init__(self, in_chans, num_class, dropout): # in_chans not used!! Just for immediate integration.
        super(SVM, self).__init__()

        self.flat = nn.Flatten()

        self.linear1 = nn.Sequential(nn.BatchNorm1d(196608), # Fix 3x128x128
                                     nn.Linear(196608, 1024))

        self.linear2 = nn.Sequential(nn.BatchNorm1d(1024),
                                     nn.Linear(1024, 64))

        self.classifier = nn.Sequential(nn.BatchNorm1d(64),
                                     nn.Linear(64, num_class))      

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)


    def forward(self, input):

        x = self.flat(input)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        x = self.dropout(x)
        out = self.classifier(x)

        return out
import torch 
import torch.nn as nn
import torch.nn.functional as f
import numpy 
import pytorch_lightning as pl
from .mlpbase import MlpModel
from .mpnn import MessagePassGates

class MpGruUnit(nn.Module):
    def __init__(self,
                d_in,
                num_units,
                support_len,
                order,
                activation=None):
        super(MpGruUnit, self).__init__()
        self.reset_gate=MessagePassGates(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.update_gate=MessagePassGates(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.cvalue_gate=MessagePassGates(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
    
    def forward(self,X,H,W):
        embedding1=torch.cat((X,H),dim=1)
        R=torch.sigmoid(self.reset_gate(embedding1,W))
        U=torch.sigmoid(self.update_gate(embedding1,W))
        embedding2=torch.cat((X,R*H),dim=1)
        c=torch.tanh(self.cvalue_gate(embedding2,W))
        
        new_h=U*H+(torch.ones(*U.shape).cuda()-U)*c

        return new_h


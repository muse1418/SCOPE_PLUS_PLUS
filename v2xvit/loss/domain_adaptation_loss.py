import torch
import torch.nn as nn
import numpy as np
from icecream import ic 
import torch.nn.functional as F
from icecream import ic

class CMD_Agnostic_loss(nn.Module):
    def __init__(self):
        super(CMD_Agnostic_loss, self).__init__()
    def forward(self, x1, x2, n_moments):
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        batch_size = x1.shape[0]
        mx1 = torch.mean(x1, dim=1, keepdim=True)
        mx2 = torch.mean(x2, dim=1, keepdim=True)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms/batch_size

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = abs(torch.sum(power)+1e-6)
        sqrt = summed**(0.5)
        return sqrt
   

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), dim=1, keepdim=True)
        ss2 = torch.mean(torch.pow(sx2, k), dim=1, keepdim=True)
        return self.matchnorm(ss1, ss2)
    
class Adaptaion_loss(nn.Module):
    def __init__(self):
        super(Adaptaion_loss, self).__init__()
        self.cal_loss = CMD_Agnostic_loss()
        self.n_moments = 5
        
    def forward(self, value, agent_num):
        pixel_num = value.shape[1] // agent_num
        ego_value = value[:, :pixel_num, :]
        
        loss_list = []
        for n in range(1, agent_num):
            colla_value = value[:, n*pixel_num:(n+1)*pixel_num, :]
            loss_list.append(self.cal_loss(ego_value, colla_value, self.n_moments))
            
        mean_loss = sum(loss_list)/len(loss_list)
        
        return mean_loss
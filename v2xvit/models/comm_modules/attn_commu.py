import torch
import torch.nn as nn
import numpy as np
from icecream import ic 
import torch.nn.functional as F
import random

class HSIC_Exclusive_loss(nn.Module):
    def __init__(self):
        super(HSIC_Exclusive_loss, self).__init__()
    
    def forward(self, emb1, emb2): 
        dim = emb1.size(1)
        emb1 = torch.mean(emb1.squeeze(0), dim = 2)
        emb2 = torch.mean(emb2.squeeze(0), dim = 2)
        R = torch.eye(dim) - (1/dim) * torch.ones(dim, dim)
        R = R.to(emb1.device)
        K1 = torch.mm(emb1, emb1.t())
        K2 = torch.mm(emb2, emb2.t())
        RK1 = torch.mm(R, K1)
        RK2 = torch.mm(R, K2)
        
        HSIC_loss = torch.trace(torch.mm(RK1, RK2))
        
        return HSIC_loss
    

class Exclusive_loss(nn.Module):
    def __init__(self):
        super(Exclusive_loss, self).__init__()
        self.cal_loss = HSIC_Exclusive_loss()
        
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
        
    def forward(self, x, record_len): 
        _, C, H, W = x.shape
        feats = self.regroup(x, record_len)
        B = len(feats)
        
        loss_list = []
        for b in range(B):
            ego_feat = feats[b][:1].permute(0,2,3,1)
            if feats[b].shape[0] == 1:
                loss_list.append(0)
                continue
            for n in range(1, feats[b].shape[0]):
                colla_feat = feats[b][n:n+1].permute(0,2,3,1)
                loss_list.append(self.cal_loss(ego_feat, colla_feat))
                
        mean_loss = sum(loss_list)/len(loss_list)
        
        return mean_loss

class AttnCommu(nn.Module):
    def __init__(self, args):
        super(AttnCommu, self).__init__()
        self.ego_proj = nn.Linear(args['channel'], 1)
        self.colla_proj = nn.Linear(args['channel'], args['channel'])
        self.sqrt_dim = np.sqrt(args['channel'])
        self.adapt_matrix = nn.Parameter(torch.ones(1,1,args['channel']))
        
        self.smooth = False
        self.thre = args['thre']
        if 'gaussian_smooth' in args:
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
            
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()
        
        
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
        
    def forward(self, x, record_len, confidence_map_list):
        _, C, H, W = x.shape
        feats = self.regroup(x, record_len)
        
        communication_masks = []
        communication_rates = []
        
        B = len(feats)
        for b in range(B):
            if feats[b].shape[0] == 1:
                communication_rates.append(torch.tensor(0).to(x.device))
                communication_masks.append(torch.ones(1,1,H,W).to(x.device))
                continue
            ego_feat = feats[b][:1].permute(0,2,3,1)  
            request = self.ego_proj(ego_feat)  
            request = request.view(1, H*W, 1) 
            score_list = [] 
            for n in range(1, feats[b].shape[0]):
                colla_feat = feats[b][n:n+1].permute(0,2,3,1)  
                colla_key = self.colla_proj(colla_feat)  
                colla_key = colla_key.view(1, H*W, -1).permute(1, 0, 2) 
                
                score_temp = torch.bmm(request, self.adapt_matrix).permute(1,0,2) 
                score = torch.bmm(score_temp, colla_key.transpose(1, 2)) / self.sqrt_dim
                score = score.permute(1, 2, 0).view(1, 1, H, W).sigmoid()
                score_list.append(score)
            communication_map = torch.cat(score_list, dim=0)  
            
            if self.smooth:
                communication_map = self.gaussian_filter(communication_map)
                
            L = communication_map.shape[0] 
            
            pre_confidence = confidence_map_list[b][1:]
            communication_map = communication_map * pre_confidence
            ic(communication_map.shape, L)
            
            ones_mask = torch.ones_like(communication_map).to(communication_map.device)
            zeros_mask = torch.zeros_like(communication_map).to(communication_map.device)
            communication_mask = torch.where(communication_map>self.thre, ones_mask, zeros_mask)
            
            communication_rate = communication_mask.sum()/(H*W*L)
            communication_rates.append(communication_rate)
            
            ones_mask = torch.ones(1,1,H,W).to(communication_mask.device)
            ic(ones_mask.shape, communication_mask.shape)
            communication_mask_nodiag = torch.cat([ones_mask, communication_mask], dim=0)
            communication_masks.append(communication_mask_nodiag)
            
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.cat(communication_masks, dim=0)
        return communication_masks, communication_rates
    
    
    
                
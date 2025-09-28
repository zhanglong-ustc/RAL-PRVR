# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch BERT model."""

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import sys
import ipdb
from Models.RAL4gmmv2.prob_models.contrast_loss import Contrastive_loss, reparameterise
sys.path.append('..')

logger = logging.getLogger(__name__)

def gelu(x):

    """
    Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):

        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class MILNCELoss(nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()
        # self.batch_size = batch_size
        # self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix, batch_size, n_pair):
        mm_mask = np.eye(batch_size)
        mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))     # Crocker product
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(batch_size).to(sim_matrix.device) * n_pair + (n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()

        return sim_loss

class MILNCELoss_BoF(nn.Module):
    def __init__(self):
        super(MILNCELoss_BoF, self).__init__()

        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix, batch_size, n_video, n_text):
        if sim_matrix.size(0) // batch_size == n_video:     # from v
            la = np.ones((n_video, n_text))
        else:
            la = np.ones((n_text, n_video))

        mm_mask = np.eye(batch_size)
        mm_mask = np.kron(mm_mask, la)     #  Crocker product

        "Expand the mask such that all elements in the mask are 1."

        mm_mask = torch.tensor(mm_mask).float().bool()
        mm_mask = mm_mask.to(sim_matrix.device)
        
        sim_loss = - (F.log_softmax(sim_matrix, dim=1) * mm_mask).sum(1) / mm_mask.sum(1)
        sim_loss = sim_loss.mean()
        return sim_loss

class KLdivergence(nn.Module):
    def __init__(self):
        super(KLdivergence, self).__init__()
    
    def kl_divergence(self, mu, logsigma):
        # -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum()
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())
    
    def forward(self, sampled_video_features, video_logsigma, sampled_text_features, text_logsigma):

        sub_kl_loss_v = self.kl_divergence(sampled_video_features.mean(dim=1), video_logsigma) # torch.Size([640, 384])
        sub_kl_loss_v = sub_kl_loss_v.sum(dim=1).mean() # tensor(33.8940)
        
        sub_kl_loss_t = self.kl_divergence(sampled_text_features.mean(dim=1), text_logsigma)
        sub_kl_loss_t = sub_kl_loss_t.sum(dim=1).mean() # tensor (48.5409)
        
        #### Cross-modal KL-loss
        mu_v = sampled_video_features.mean(dim=1)
        mu_t = sampled_text_features.mean(dim=1)
        var_t = torch.exp(text_logsigma)
        var_v = torch.exp(video_logsigma)
        KL_loss_1 = video_logsigma - text_logsigma +((var_t.pow(2)+(mu_t-mu_v).pow(2))/(2*var_v.pow(2)))-0.5

        # KL_loss_1: torch.Size([640, 384])   

        KL_loss_1 = KL_loss_1.sum(dim=1).mean()  # tensor(115.7747)

        vib_loss = sub_kl_loss_v + sub_kl_loss_t + KL_loss_1

        return vib_loss


class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()

class AllGather(torch.autograd.Function):

    """ An autograd function that performs allgather on a tensor. """

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )




class dual_softmax_loss(nn.Module):
    def __init__(self,):
        super(dual_softmax_loss, self).__init__()
        
    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix) 
        #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss

def con_loss(txt_mu, txt_logvar, img_mu, img_logvar):

    Conloss=Contrastive_loss(0.5)

    while True:
        t_z1 = reparameterise(txt_mu, txt_logvar)
        t_z2 = reparameterise(txt_mu, txt_logvar)
        
        if not np.array_equal(t_z1, t_z2):
            break 


    while True:
        i_z1=reparameterise(img_mu,img_logvar)
        i_z2=reparameterise(img_mu,img_logvar)
    
        if not np.array_equal(i_z1, i_z2):
            break 


    loss_t=Conloss(t_z1,t_z2)
    loss_i=Conloss(i_z1,i_z2)
   
    return loss_t + loss_i 



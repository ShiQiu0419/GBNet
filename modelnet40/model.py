#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Shi Qiu (based on Yue Wang's codes)
@Contact: shi.qiu@anu.edu.au
@File: model.py
@Time: 2021/04/23
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def geometric_point_descriptor(x, k=3, idx=None):
    # x: B,3,N
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([1])) # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([2])) # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st-org_x
    edge2 = neighbor_2nd-org_x
    normals = torch.cross(edge1, edge2, dim=1) # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True) # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True) # B,1,N

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1) # B,14,N

    return new_pts


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""
    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(1024//8)
        self.bn2 = nn.BatchNorm1d(1024//8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024//8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024//8, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat) 
        proj_key = self.key_conv(x_hat).permute(0, 2, 1) 
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat)-similarity_mat
        affinity_mat = self.softmax(affinity_mat)
        
        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha*out + x 
        return out


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class GBNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GBNet, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(28, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.bn11 = nn.BatchNorm2d(14)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 14, kernel_size=[1,args.k], bias=False),
                                   self.bn11,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.bn12 = nn.BatchNorm2d(64)
        self.conv12 = nn.Sequential(nn.Conv2d(28, 64, kernel_size=1, bias=False),
                                   self.bn12,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa1 = CAA_Module(64)
        
        self.bn13 = nn.BatchNorm2d(64)
        self.conv13 = nn.Sequential(nn.Conv2d(28, 64, kernel_size=[1,args.k], bias=False),
                                   self.bn13,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa11 = CAA_Module(64)

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn21 = nn.BatchNorm2d(64)
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=[1,args.k], bias=False),
                                   self.bn21,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn22 = nn.BatchNorm2d(64)
        self.conv22 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn22,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa2 = CAA_Module(64)

        self.bn23 = nn.BatchNorm2d(64)
        self.conv23 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=[1,args.k], bias=False),
                                   self.bn23,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa21 = CAA_Module(64)

        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn31 = nn.BatchNorm2d(64)
        self.conv31 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=[1,args.k], bias=False),
                                   self.bn31,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn32 = nn.BatchNorm2d(128)
        self.conv32 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn32,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa3 = CAA_Module(128)

        self.bn33 = nn.BatchNorm2d(128)
        self.conv33 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=[1,args.k], bias=False),
                                   self.bn33,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa31 = CAA_Module(128)

        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn41 = nn.BatchNorm2d(128)
        self.conv41 = nn.Sequential(nn.Conv2d(128 * 2, 128, kernel_size=[1,args.k], bias=False),
                                   self.bn41,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn42 = nn.BatchNorm2d(256)
        self.conv42 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn42,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa4 = CAA_Module(256)

        self.bn43 = nn.BatchNorm2d(256)
        self.conv43 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=[1,args.k], bias=False),
                                   self.bn43,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa41 = CAA_Module(256)

        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv5 = nn.Sequential(nn.Conv1d(1024, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
     
        self.caa5 = CAA_Module(1024)

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn_linear1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn_linear2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # x: B,3,N
        batch_size = x.size(0)
        #######################################
        # Geometric Point Descriptor:
        #######################################
        x = geometric_point_descriptor(x) # B,14,N

        #######################################
        # 1st Attentional Back-projection Edge Features Module (ABEM):
        #######################################
        # Prominent Feature Encoding
        x1_1 = x # input
        input_edge = get_graph_feature(x, k=self.k)
        x = self.conv1(input_edge)
        x1_2 = x # EdgeConv for input

        x = self.conv11(x) # LFC
        x = torch.squeeze(x, -1)
        x1_3 = x # Back-projection signal

        delta_1 = x1_3 - x1_1 # Error signal

        x = get_graph_feature(delta_1, k=self.k)  # EdgeConv for Error signal
        x = self.conv12(x)
        x1_4 = x

        x = x1_2 + x1_4 # Attentional feedback
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = self.caa1(x1) # B,64,N

        # Fine-grained Feature Encoding
        x1_local = self.conv13(input_edge)
        x1_local = torch.squeeze(x1_local, -1) 
        x1_local = self.caa11(x1_local) # B,64,N

        #######################################
        # 2nd Attentional Back-projection Edge Features Module (ABEM):
        #######################################
        # Prominent Feature Encoding
        x2_1 = x1
        input_edge = get_graph_feature(x1, k=self.k)
        x = self.conv2(input_edge)
        x2_2 = x

        x = self.conv21(x) 
        x = torch.squeeze(x, -1) 
        x2_3 = x

        delta_2 = x2_3 - x2_1 

        x = get_graph_feature(delta_2, k=self.k) 
        x = self.conv22(x) 
        x2_4 = x

        x = x2_2 + x2_4 
        x2 = x.max(dim=-1, keepdim=False)[0] 
        x2 = self.caa2(x2) # B,64,N

        # Fine-grained Feature Encoding
        x2_local = self.conv23(input_edge) 
        x2_local = torch.squeeze(x2_local, -1) 
        x2_local = self.caa21(x2_local) # B,64,N

        #######################################
        # 3rd Attentional Back-projection Edge Features Module (ABEM):
        #######################################
        # Prominent Feature Encoding
        x3_1 = x2
        input_edge = get_graph_feature(x2, k=self.k)
        x = self.conv3(input_edge)
        x3_2 = x

        x = self.conv31(x) 
        x = torch.squeeze(x, -1) 
        x3_3 = x

        delta_3 = x3_3 - x3_1 

        x = get_graph_feature(delta_3, k=self.k) 
        x = self.conv32(x)
        x3_4 = x

        x = x3_2 + x3_4
        x3 = x.max(dim=-1, keepdim=False)[0]  
        x3 = self.caa3(x3) # B,128,N

        # Fine-grained Feature Encoding
        x3_local = self.conv33(input_edge)
        x3_local = torch.squeeze(x3_local, -1)
        x3_local = self.caa31(x3_local) # B,128,N

        #######################################
        # 4th Attentional Back-projection Edge Features Module (ABEM):
        #######################################
        # Prominent Feature Encoding
        x4_1 = x3 
        input_edge = get_graph_feature(x3, k=self.k) 
        x = self.conv4(input_edge) 
        x4_2 = x

        x = self.conv41(x) 
        x = torch.squeeze(x, -1) 
        x4_3 = x

        delta_4 = x4_3 - x4_1 

        x = get_graph_feature(delta_4, k=self.k) 
        x = self.conv42(x)
        x4_4 = x

        x = x4_2 + x4_4 
        x4 = x.max(dim=-1, keepdim=False)[0]  
        x4 = self.caa4(x4) # B,256,N

        # Fine-grained Feature Encoding
        x4_local = self.conv43(input_edge) 
        x4_local = torch.squeeze(x4_local, -1) 
        x4_local = self.caa41(x4_local) # B,256,N

        #######################################
        # End of all 4 ABEMs 
        #######################################

        # Concatenate both prominent and fine-grained outputs of 4 ABEMs:
        x = torch.cat((x1, x1_local, x2, x2_local, x3, x3_local, x4, x4_local), dim=1)  # B,(64+64+128+256)x2,N
        # MLP
        x = self.conv5(x) # B,1024,N
        # CAA 
        x = self.caa5(x)

        # global embedding
        global_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        global_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((global_max, global_avg), 1)  # B,2048

        # FC layers with dropout
        x = F.leaky_relu(self.bn_linear1(self.linear1(x)), negative_slope=0.2)  # B,512
        x = self.dp1(x)
        x = F.leaky_relu(self.bn_linear2(self.linear2(x)), negative_slope=0.2)  # B,256
        x = self.dp2(x)
        x = self.linear3(x)  # B,C

        return x

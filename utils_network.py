import os
import h5py
from random import uniform
import torch
import numpy as np
from torch import Tensor, cudnn_affine_grid_generator
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Module
import torch.nn as nn
from torch.optim import optimizer
import torch.nn.functional as F
from utils import *


#主要包含pilot_network, VQVAE以及普通量化网络


class PhaseShifter_Pilot(Module):

    def __init__(self, BS_antenna: int, UE_antenna: int , BS_RF_chain: int,UE_RF_chain: int,Subcarrier: int, Subcarrier_group: int, Pilot_num: int) -> None:

        super(PhaseShifter_Pilot, self).__init__()
        self.BS_antenna = BS_antenna #基站天线数量
        self.UE_antenna = UE_antenna #用户端天线数量
        self.BS_RF_chain = BS_RF_chain # BS RF chain数量
        self.UE_RF_chain = UE_RF_chain # BS RF chain数量

        # self.DATA_stream = DATA_stream #数据流数量
        self.Subcarrier = Subcarrier #子载波总数
        self.Subcarrier_group = Subcarrier_group #每个subcarrier group中subcarrier的数量
        self.Subcarrier_gap = self.Subcarrier_group
        self.Pilot_subcarrier_index = range(0,self.Subcarrier,self.Subcarrier_gap) #发射pilot的子载波序号
        self.Subcarrier_group_num = int(self.Subcarrier/self.Subcarrier_group) #subcarrier group 的数量
        self.Pilot_num = Pilot_num #每次发射的pilot的数量



        self.BS_scale = np.sqrt(BS_antenna) #基站端归一化系数
        self.UE_scale = np.sqrt(UE_antenna) #用户端归一化系数

        
        #第三版~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.WRF_theta = Parameter(torch.Tensor(self.Pilot_subcarrier,self.Pilot_num,self.RF_chain, self.UE_antenna)) #UE端的analog precoding，Pilot_subcarrier×pilot数量×RF数量×UE天线
        # self.FRF_theta = Parameter(torch.Tensor(self.Pilot_subcarrier,self.Pilot_num,self.BS_antenna, self.RF_chain)) #BS端的analog precoding

        # self.bias_theta = Parameter(torch.Tensor(1, self.Subcarrier,self.Pilot_subcarrier*self.Pilot_num*self.RF_chain)) #[subcarrier, pilot_subcarrier*pilot_num*RF_chain]

        #第二版~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.WRF_theta = Parameter(torch.Tensor(1,self.Pilot_num,self.RF_chain, self.UE_antenna)) #UE端的analog precoding，1×pilot数量×RF数量×UE天线
        # self.FRF_theta = Parameter(torch.Tensor(1,self.Pilot_num,self.BS_antenna, self.RF_chain)) #BS端的analog precoding


        #初版~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.WBB_real_norm = Parameter(torch.Tensor(self.Pilot_num,self.Pilot_subcarrier,self.DATA_stream,self.RF_chain)) #UE端的digital precoding，pilot数量×发送Pilot的载波数量×数据流数量×RF数量
        # self.WBB_imag_norm = Parameter(torch.Tensor(self.Pilot_num,self.Pilot_subcarrier,self.DATA_stream,self.RF_chain)) #UE端的digital precoding

        # self.FBB_real_norm = Parameter(torch.Tensor(self.Pilot_num,self.Pilot_subcarrier,self.RF_chain,self.DATA_stream)) #BS端的digital precoding 维度[Pilot_num,Pilot_subcarrier,RF_chain,DATA_stream]
        # self.FBB_imag_norm = Parameter(torch.Tensor(self.Pilot_num,self.Pilot_subcarrier,self.RF_chain,self.DATA_stream)) #BS端的digital precoding

        #和孙老师以及师兄讨论后的版本~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.WRF_theta = Parameter(torch.Tensor(1,self.Pilot_num,self.UE_RF_chain, self.UE_antenna)) #[1, Pilot_num, UE_RF_chain, UE_antenna]
        self.FRF_theta = Parameter(torch.Tensor(1,self.Pilot_num,self.BS_antenna, self.BS_RF_chain)) #[1, Pilot_num, BS_antenna, BS_RF_chain]

        self.reset_parameters() #初始化参数

    def reset_parameters(self) -> None:  #初始化角度值和digital值

            init.uniform_(self.WRF_theta, a=0, b=2*np.pi)
            init.uniform_(self.FRF_theta, a=0, b=2*np.pi)

            # init.normal_(self.WBB_real_norm, mean=0, std=1)
            # init.normal_(self.WBB_imag_norm, mean=0, std=1)

            # init.normal_(self.FBB_real_norm, mean=0, std=1)
            # init.normal_(self.FBB_imag_norm, mean=0, std=1)

    def forward(self, H_real: Tensor,H_imag: Tensor) -> Tensor:  #输入是H矩阵的实部和虚部，维度都是[batch_size,Subcarrier,UE_antenna,BS_antenna]
        H_real = H_real[:,self.Pilot_subcarrier_index,:,:] #输入是H矩阵的实部和虚部，维度都是[batch_size,Subcarrier_group_num,UE_antenna,BS_antenna]
        H_imag = H_imag[:,self.Pilot_subcarrier_index,:,:] #输入是H矩阵的实部和虚部，维度都是[batch_size,Subcarrier_group_num,UE_antenna,BS_antenna]


        #首先是BS端的analog matrix和digital matrix
        FRF_real = (1 / self.BS_scale) * torch.cos(self.FRF_theta)  #维度[1,Pilot_num,BS_antenna,BS_RF_chain]
        FRF_imag = (1 / self.BS_scale) * torch.sin(self.FRF_theta)  
        F_complex = FRF_real + 1j*FRF_imag
        power_F = torch.pow(F_norm_complex(F_complex),2) #[1,Pilot_num,]

        WRF_real = (1 / self.UE_scale) * torch.cos(self.WRF_theta)  #维度[1,Pilot_num,UE_RF_chain,UE_antenna]
        WRF_imag = (1 / self.UE_scale) * torch.sin(self.WRF_theta)  
        W_complex = WRF_real + 1j*WRF_imag
        power_W = torch.pow(F_norm_complex(W_complex),2) #[1,Pilot_num,]


        #最后得到结果output

        # H_real = torch.reshape(H_real,(H_real.shape[0], self.Subcarrier_group_num, 1,self.UE_antenna, self.BS_antenna)) #于是维度变成[batch_size,Subcarrier_group_num,1,UE_antenna,BS_antenna]
        # H_imag = torch.reshape(H_imag,(H_real.shape[0], self.Subcarrier_group_num, 1,self.UE_antenna, self.BS_antenna)) #于是维度变成[batch_size,Subcarrier_group_num,1,UE_antenna,BS_antenna]
        H_real = torch.unsqueeze(H_real,dim = -3) #于是维度变成[batch_size,Subcarrier_group_num,1,UE_antenna,BS_antenna]
        H_imag = torch.unsqueeze(H_imag,dim = -3) #于是维度变成[batch_size,Subcarrier_group_num,1,UE_antenna,BS_antenna]

        temp_real,temp_imag = matrix_product(H_real,H_imag,FRF_real,FRF_imag) #乘完之后维度变成[batch_size,Subcarrier_group_num, Pilot_num,UE_antenna,BS_RF_chain]
       
        output_real,output_imag = matrix_product(WRF_real,WRF_imag,temp_real,temp_imag) #乘完之后维度变成[batch_size,Subcarrier_group_num, Pilot_num,UE_RF_chain,BS_RF_chain]

        output_real = torch.sum(output_real,dim=-1,keepdim=False)   #假设s是个1向量，求和之后维度变成[batch_size,Subcarrier_group_num, Pilot_num,UE_RF_chain]
        output_imag = torch.sum(output_imag,dim=-1,keepdim=False)   

        output_real = torch.reshape(output_real,(-1, self.Subcarrier_group_num, self.Pilot_num*self.UE_RF_chain)) #[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain]
        output_imag = torch.reshape(output_imag,(-1, self.Subcarrier_group_num, self.Pilot_num*self.UE_RF_chain)) #[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain]

        return output_real,output_imag


    




class FloatBiter(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b
        dot_base = 2 ** torch.arange(0, b)
        self.register_buffer("dot_base", dot_base.unsqueeze(0).float())

    @torch.no_grad()
    def to_bit(self, x):
        x = x + 1  #先增加1
        x = x.clamp(1., 16) #固定到1~4^2
        x = x.unsqueeze(-1)
        x = torch.log(x) / np.log(4) #固定到0~2
        bits = (x * self.dot_base).long() % 2 #分别是1,0.5,0.25,0.125...
        return bits

    @torch.no_grad()
    def to_float(self, bits):
        x = (bits.float() / self.dot_base).sum(-1)
        x = torch.pow(4,x)
        x = x - 1 #先增加1最后减去1
        return x

    def forward(self, x):
        bits = self.to_bit(x)
        x1 = self.to_float(bits)
        return (x1 - x).detach() + x



class VQVAE(nn.Module):
    def __init__(self, vq_b=8, vq_dim=2, s_bit=8,feadback_feature = 64): #s_bit表示scale所使用的bit, vq_b表示一个向量的index需要多少bit表示，字典里总共会有2*vq_b个向量, vq_dim表示每个向量的长度
        #总共需要的bit数为Feedback_feature/vq_dim*vq_b + s_bit
        super().__init__()
        self.vq_b = vq_b
        self.vq_dim = vq_dim
        self.s_bit = s_bit #用于scaling的比特
        self.Feedback_feature = feadback_feature #用于初始化字典

        self.scale_Q = FloatBiter(s_bit) # 用于scaling的量化
        k = 2**vq_b
        embed = torch.randn(k,vq_dim)
        self.embed = nn.Parameter(embed)    #字典集合
        self.reset_embed()

    def reset_embed(self) -> None:
        self.embed = nn.Parameter(self.embed/np.sqrt(self.Feedback_feature)*4)

    def preprocess(self, feedback):  #这里的feedback维度是[batch_size, Pilot_subcarrier,Pilot_group*Pilot_num × RF_chain × 2]
        b = feedback.size(0)   #batchsize的个数
        p = feedback.size(1)   #subcarrier的个数
        # feedback = feedback.view( b, -1 )
        scale = torch.norm(feedback, dim=-1)
        return scale
    def forward(self, feedback):
        b = feedback.size(0)   #batchsize的个数
        p = feedback.size(1)   #subcarrier的个数
        scale = self.preprocess(feedback) #feedback的模 维度是[batch_size, Pilot_subcarrier]
        scale_Q = self.scale_Q(scale)   #量化后的scale[batch_size, Pilot_subcarrier]

        scale = scale.unsqueeze(-1) #[batch_size, Pilot_subcarrier,1]
        scale_Q = scale_Q.unsqueeze(-1)


        feedback_norm = feedback / scale     #归一化后的feedback,维度是[batchsize,Pilot_subcarrier,Pilot_num × RF_chain × 2]
        
        feedback_norm = feedback_norm.view(b,p,-1,self.vq_dim) #将维度变为[batchsize,subcarrier, Pilot_subcarrier, Pilot_num × RF_chain × 2/vq_dim , vq_dim]
        dist = self.dist(feedback_norm.unsqueeze(-2), self.embed.unsqueeze(0).unsqueeze(1).unsqueeze(2)) #[batchsize,subcarrier, Pilot_num × RF_chain × 2/vq_dim ,1, vq_dim] 和 [1,1,1,k,vq_dim]
        _, ind = dist.min(-1)  # (b,p, Pilot_num × RF_chain × 2/vq_dim)

        feedback_Q_norm = F.embedding(ind, self.embed)  # [batchsize,p, Pilot_num × RF_chain × 2/vq_dim , vq_dim]
        loss1 = self.dist(feedback_Q_norm, feedback_norm.detach()).mean().mean()
        loss2 = self.dist(feedback_Q_norm.detach(), feedback_norm).mean().mean()
        
        feedback_Q = feedback_Q_norm.view(b,p,-1) * scale_Q# 将维度变为[batchsize,Pilot_subcarrier,Pilot_num × RF_chain × 2]

        return (feedback_Q-feedback).detach() + feedback, loss1,loss2


    @staticmethod
    def dist(x, y):
        return (x - y).pow(2).mean(-1)
    



class FloatBiterAE(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b
        dot_base = 2 ** torch.arange(0, b)
        self.register_buffer("dot_base", dot_base.unsqueeze(0).float())

    @torch.no_grad()
    def to_bit(self, x):
        # x = x + 1  #先增加1
        # x = x.clamp(1., 9) #固定到1~3^2
        x = x.unsqueeze(-1)
        # x = torch.log(x) / np.log(3) #固定到0~2
        bits = (x * self.dot_base).long() % 2 #分别是1,0.5,0.25,0.125...
        return bits

    @torch.no_grad()
    def to_float(self, bits):
        x = (bits.float() / self.dot_base).sum(-1)
        # x = torch.pow(3,x)
        # x = x - 1 #先增加1最后减去1
        return x

    def forward(self, x):
        bits = self.to_bit(x)
        x1 = self.to_float(bits)
        return (x1 - x).detach() + x

class NAE(nn.Module):
    def __init__(self, B=4): #s_bit表示scale所使用的bit, B表示多少个bit表示一个数
        
        super().__init__()
        self.B = B

        self.AE_B = FloatBiterAE(self.B) # 用于scaling的量化


    def forward(self, feedback): #这里的feedback维度是[batch_size, Subcarrier_group_num, Subcarrier_group*Pilot_num*RF_chain*2]
    
        feedback_Q = feedback.clamp(-4,4)
        feedback_Q = (feedback_Q + 4) / 4 #固定到0~2

        feedback_Q = self.AE_B(feedback_Q)   #量化后的scale[batch_size, Pilot_subcarrier]

        feedback_Q = feedback_Q * 4 - 4

        return (feedback_Q-feedback).detach() + feedback


    @staticmethod
    def dist(x, y):
        return (x - y).pow(2).mean(-1)
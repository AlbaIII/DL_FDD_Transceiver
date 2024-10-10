##最最正统的GNN网络
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



# def matrix_product(A_real,A_imag,B_real,B_imag): #A*B复数矩阵乘法， A维度 N1×N， B维度 N×N2， 最终结果 [R;I] 2N1×N2

#     cat_kernels_A_real = torch.cat((A_real,-A_imag),dim=-1)
#     cat_kernels_A_imag = torch.cat((A_imag,A_real),dim=-1)
#     # cat_kernels_A_complex = torch.cat((cat_kernels_A_real, cat_kernels_A_imag),dim=-2)

#     cat_kernels_B_complex = torch.cat((B_real,B_imag),dim=-2)

#     output_real = torch.matmul(cat_kernels_A_real,cat_kernels_B_complex)
#     output_imag = torch.matmul(cat_kernels_A_imag,cat_kernels_B_complex)

#     return output_real,output_imag

# def F_norm(real_part,imag_part): #计算矩阵的F范数，最后的结果是开平方的

#     sq_real = torch.pow(real_part,2)
#     sq_imag = torch.pow(imag_part,2)
#     abs_values = sq_real + sq_imag

#     return torch.pow( torch.sum( torch.sum(abs_values,dim=-1,keepdim=False) , dim=-1 , keepdim=False) , 0.5 )


class Hybrid(Module):

    def __init__(self, BS_antenna: int, UE_antenna: int , RF_chain: int,DATA_stream: int,Subcarrier: int, Subcarrier_gap: int, Pilot_num: int) -> None:

        super(Hybrid, self).__init__()
        self.BS_antenna = BS_antenna #基站天线数量
        self.UE_antenna = UE_antenna #用户端天线数量
        self.RF_chain = RF_chain #RF chain数量
        self.DATA_stream = DATA_stream #数据流数量
        self.Subcarrier = Subcarrier #子载波总数
        self.Subcarrier_gap = Subcarrier_gap #发射pilot的子载波之间的间隔数

        # self.Pilot_subcarrier_index = range(0,self.Subcarrier-1,self.Subcarrier_gap) #发射pilot的子载波序号
        self.Pilot_subcarrier = int(self.Subcarrier/self.Subcarrier_gap) #发射pilot的子载波数
        self.Pilot_num = Pilot_num #每次发射的pilot的数量



        self.BS_scale = np.sqrt(BS_antenna) #基站端归一化系数
        self.UE_scale = np.sqrt(UE_antenna) #用户端归一化系数

        self.WRF_theta = Parameter(torch.Tensor(self.Pilot_num,1,self.RF_chain, self.UE_antenna)) #UE端的analog precoding，pilot数量×1×RF数量×UE天线
        self.FRF_theta = Parameter(torch.Tensor(self.Pilot_num,1,self.BS_antenna, self.RF_chain)) #BS端的analog precoding
 
        self.WBB_real_norm = Parameter(torch.Tensor(self.Pilot_num,self.Pilot_subcarrier,self.DATA_stream,self.RF_chain)) #UE端的digital precoding，pilot数量×发送Pilot的载波数量×数据流数量×RF数量
        self.WBB_imag_norm = Parameter(torch.Tensor(self.Pilot_num,self.Pilot_subcarrier,self.DATA_stream,self.RF_chain)) #UE端的digital precoding

        self.FBB_real_norm = Parameter(torch.Tensor(self.Pilot_num,self.Pilot_subcarrier,self.RF_chain,self.DATA_stream)) #BS端的digital precoding 维度[Pilot_num,Pilot_subcarrier,RF_chain,DATA_stream]
        self.FBB_imag_norm = Parameter(torch.Tensor(self.Pilot_num,self.Pilot_subcarrier,self.RF_chain,self.DATA_stream)) #BS端的digital precoding

        self.reset_parameters() #初始化参数

    def reset_parameters(self) -> None:  #初始化角度值和digital值

            init.uniform_(self.WRF_theta, a=0, b=2*np.pi)
            init.uniform_(self.FRF_theta, a=0, b=2*np.pi)

            init.normal_(self.WBB_real_norm, mean=0, std=1)
            init.normal_(self.WBB_imag_norm, mean=0, std=1)

            init.normal_(self.FBB_real_norm, mean=0, std=1)
            init.normal_(self.FBB_imag_norm, mean=0, std=1)

    def forward(self, H_real: Tensor,H_imag: Tensor) -> Tensor:  #输入是H矩阵的实部和虚部，维度都是[batch_size,Pilot_subcarrier,UE_antenna,BS_antenna]

        #首先是BS端的analog matrix和digital matrix
        self.FRF_real = (1 / self.BS_scale) * torch.cos(self.FRF_theta)  #维度[Pilot_num,1,BS_antenna,RF_chain]
        self.FRF_imag = (1 / self.BS_scale) * torch.sin(self.FRF_theta)  

        BS_real,BS_imag = matrix_product(self.FRF_real,self.FRF_imag,self.FBB_real_norm,self.FBB_imag_norm) #没有施行归一化，[Pilot_num,Pilot_subcarrier,BS_antenna,DATA_stream]
        F_complex = BS_real + 1j*BS_imag
        
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是[Pilot_num,Pilot_subcarrier->1,1,1]
        BS_real,BS_imag = BS_real/torch.sqrt(power_F)*np.sqrt(H_real.shape[1]),BS_imag/torch.sqrt(power_F)*np.sqrt(H_imag.shape[1])

        F_complex = BS_real + 1j*BS_imag
        power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是[Pilot_num,Pilot_subcarrier->1,1,1]
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




        #然后是UE端的analog matrix和digital matrix
        self.WRF_real = (1 / self.UE_scale) * torch.cos(self.WRF_theta)  
        self.WRF_imag = (1 / self.UE_scale) * torch.sin(self.WRF_theta)  

        UE_real,UE_imag = matrix_product(self.WBB_real_norm,self.WBB_imag_norm,self.WRF_real,self.WRF_imag) #没有施行归一化，[Pilot_num,Pilot_subcarrier,DATA_stream,UE_antenna]
        W_complex = UE_real + 1j*UE_imag

        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是[Pilot_num,Pilot_subcarrier->1,1,1]
        UE_real,UE_imag = UE_real/torch.sqrt(power_W)*np.sqrt(H_real.shape[1])*np.sqrt(self.DATA_stream),UE_imag/torch.sqrt(power_W)*np.sqrt(H_imag.shape[1])*np.sqrt(self.DATA_stream)

        W_complex = UE_real + 1j*UE_imag
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是[Pilot_num,Pilot_subcarrier->1,1,1]
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        #最后得到结果output
        H_real = torch.unsqueeze(H_real,dim = -4) #于是维度变成[batch_size,1,Pilot_subcarrier,UE_antenna,BS_antenna]
        H_imag = torch.unsqueeze(H_imag,dim = -4)

        temp_real,temp_imag = matrix_product(H_real,H_imag,BS_real,BS_imag) #乘完之后维度变成[batch_size,Pilot_num,Pilot_subcarrier,UE_antenna,DATA_stream]
       
        output_real,output_imag = matrix_product(UE_real,UE_imag,temp_real,temp_imag) #乘完之后维度变成[batch_size,Pilot_num,Pilot_subcarrier,DATA_stream,DATA_stream]

        output_real = torch.sum(output_real,dim=-1,keepdim=False)   #假设s是个1向量，求和之后维度变成[batch_size,Pilot_num,Pilot_subcarrier,DATA_stream]
        output_imag = torch.sum(output_imag,dim=-1,keepdim=False)


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
        self.embed = nn.Parameter(self.embed/np.sqrt(self.Feedback_feature))

    def preprocess(self, feedback):  #这里feedback的维度是[batchsize,feedback_feature]
        b = feedback.size(0)   #batchsize的个数
        feedback = feedback.view( b, -1 )
        scale = torch.norm(feedback, dim=-1)
        return scale
    def forward(self, feedback):
        b = feedback.size(0)   #batchsize的个数
        scale = self.preprocess(feedback) #feedback的模 维度是[batchsize]
        scale_Q = self.scale_Q(scale)   #量化后的scale

        scale = scale.unsqueeze(-1) #[batchsize,1]
        scale_Q = scale_Q.unsqueeze(-1)


        feedback_norm = feedback / scale     #归一化后的feedback,维度是[batchsize,feedback_feature]
        
        feedback_norm = feedback_norm.view(b,-1,self.vq_dim) #将维度变为[batchsize, feedbackfeture/vq_dim , vq_dim]
        dist = self.dist(feedback_norm.unsqueeze(-2), self.embed.unsqueeze(0).unsqueeze(1)) #拟合到最接近的一个向量上，dist的维度是[batchsize,feedback_feture/2, k ]
        _, ind = dist.min(-1)  # (b, n)

        feedback_Q_norm = F.embedding(ind, self.embed)  # [batchsize, feedbackfeture/vq_dim , vq_dim]
        loss1 = self.dist(feedback_Q_norm, feedback_norm.detach()).mean()
        loss2 = self.dist(feedback_Q_norm.detach(), feedback_norm).mean()
        
        feedback_Q = feedback_Q_norm.view(b,-1) * scale_Q# 将维度变为[batchsize, feedbackfeture]

        return (feedback_Q-feedback).detach() + feedback, loss1,loss2


    @staticmethod
    def dist(x, y):
        return (x - y).pow(2).mean(-1)


class GNN_inital_layer(nn.Module): #GNN网络的初始层
    def __init__(self, antenna: int,  RF_chain: int,DATA_stream: int,Subcarrier: int,Feedback_feature: int): 
        #每个BB的维度是[DATA_stream,RF_chain],每个RF的维度是[antenna,RF_chain]
        super(GNN_inital_layer, self).__init__()
        self.subcarrier = Subcarrier
        self.feedback_feature = Feedback_feature
        self.BB_feature = 2*DATA_stream*RF_chain #有实部和虚部
        self.RF_feature = antenna*RF_chain #不用实部和虚部

        self.BB_net = nn.ModuleList()   
        for _ in range(self.subcarrier): #输入所有feedback，获得每个BB节点的特征
            self.BB_net.append(nn.Sequential(

                nn.Linear(in_features = Feedback_feature,
                            out_features=2 * self.BB_feature),
                # nn.Mish(),
                # nn.Linear(in_features=2 * self.BB_feature,
                #             out_features=4 * self.BB_feature),
                nn.Mish(),
                nn.Linear(in_features=2 * self.BB_feature,
                            out_features = 2 * self.BB_feature),
                nn.Mish(),
                ))

        self.RF_net = nn.Sequential(    #输入所有BB节点特征的均值，得到RF节点的特征
                nn.Linear(in_features = Feedback_feature,
                            out_features=2 * self.RF_feature),
                # nn.Mish(),
                # nn.Linear(in_features = 2 * self.RF_feature,
                #             out_features=4 * self.RF_feature),
                nn.Mish(),
                nn.Linear(in_features=2 * self.RF_feature,
                            out_features= 2 * self.RF_feature),
                nn.Mish(),
                )
        

    def forward(self, feedback): #feedback的维度是[batch_size,Pilot_num*Pilot_subcarrier*DATA_stream × 2]，需要变成subcarrier个向量
        batch_size = feedback.shape[0]
        BB_feature = (torch.zeros(size=(batch_size, self.subcarrier,2*self.BB_feature))).cuda()

        for i in range(self.subcarrier):
            BB_feature[:,i,:] = self.BB_net[i](feedback)
        
        # RF_feature = torch.mean(BB_feature,dim=1,keepdim=False)

        RF_feature = self.RF_net(feedback)

        return BB_feature,RF_feature

class GNN_layer(nn.Module): #GNN网络的迭代层
    def __init__(self, antenna: int,  RF_chain: int,DATA_stream: int,Subcarrier: int,Feedback_feature: int): 
        #每个BB的维度是[DATA_stream,RF_chain],每个RF的维度是[antenna,RF_chain]
        super(GNN_layer, self).__init__()
        self.subcarrier = Subcarrier
        self.BB_feature = 2*DATA_stream*RF_chain #有实部和虚部
        self.RF_feature = antenna*RF_chain #不用实部和虚部
        self.feature = self.BB_feature + self.RF_feature #RF和BB的feature数量的总和
        self.feedback_feature = Feedback_feature

        self.RF_BB = nn.Sequential(
                nn.Linear(in_features = 2*self.RF_feature,
                            out_features=2 * self.BB_feature),
                nn.Mish(),
                )
        self.RF_RF = nn.Sequential(
                nn.Linear(in_features = 2*self.RF_feature,
                            out_features=2 * self.RF_feature),
                nn.Mish(),
                )
        self.BB_RF = nn.Sequential(
                nn.Linear(in_features = 2*self.BB_feature,
                            out_features=2 * self.RF_feature),
                nn.Mish(),
                )
        self.BB_BB = nn.Sequential(
                nn.Linear(in_features = 2*self.BB_feature,
                            out_features=2 * self.BB_feature),
                nn.Mish(),
                )

    def forward(self, BB_feature,RF_feature): #BB_feature的维度是[batch_size,,subcarrier,BB_feature], RF_feature的维度是[batch_size,RF_feature]
        batch_size = BB_feature.shape[0]
        # new_BB_feature = (torch.zeros(size=(batch_size, self.subcarrier,2*self.BB_feature))).cuda()
        BB_mean = torch.mean(BB_feature,dim= 1 ,keepdim=False)

        RF_to_RF = self.RF_RF(RF_feature) #[batch_size,RF_feature]
        RF_to_BB = self.RF_BB(RF_feature) #[batch_size,BB_feature]
        BB_to_RF = self.BB_RF(BB_mean)  #[batch_size,RF_feature]
        BB_to_BB = self.BB_BB(BB_feature) #[batch_size,subcarrier,RF_feature]

        new_RF_feature = RF_to_RF + BB_to_RF
        new_BB_feature = BB_to_BB + torch.unsqueeze(RF_to_BB,dim=-2)

        return new_BB_feature,new_RF_feature


class GNN_Network(nn.Module): #GNN网络的迭代层
    def __init__(self, antenna: int,  RF_chain: int,DATA_stream: int,Subcarrier: int,Feedback_feature: int):
        super(GNN_Network, self).__init__()
        self.feedback_feature = Feedback_feature
        self.subcarrier = Subcarrier
        self.DATA_stream = DATA_stream
        self.RF_chain = RF_chain
        self.BB_feature = 2*DATA_stream*RF_chain #有实部和虚部
        self.RF_feature = antenna*RF_chain #不用实部和虚部

        self.inital_layer = GNN_inital_layer(antenna=antenna, RF_chain=RF_chain, DATA_stream=DATA_stream, Subcarrier=Subcarrier, Feedback_feature=Feedback_feature)
        self.layer1 = GNN_layer(antenna=antenna, RF_chain=RF_chain, DATA_stream=DATA_stream, Subcarrier=Subcarrier,Feedback_feature=Feedback_feature)
        self.layer2 = GNN_layer(antenna=antenna, RF_chain=RF_chain, DATA_stream=DATA_stream, Subcarrier=Subcarrier,Feedback_feature=Feedback_feature)
        self.layer3 = GNN_layer(antenna=antenna, RF_chain=RF_chain, DATA_stream=DATA_stream, Subcarrier=Subcarrier,Feedback_feature=Feedback_feature)


        self.analog = nn.Sequential(
            nn.Linear(in_features= 2*self.RF_feature,
                        out_features= 4*self.RF_feature),
            nn.Mish(),
            nn.Linear(in_features= 4*self.RF_feature,
                        out_features= antenna*RF_chain),
        )


        self.digital_real = nn.Sequential(
                nn.Linear(in_features= 2*self.BB_feature,
                            out_features=4 * self.BB_feature),
                nn.Mish(),
                nn.Linear(in_features=4 * self.BB_feature,
                            out_features=DATA_stream*RF_chain),
                )

        self.digital_imag = nn.Sequential(
                nn.Linear(in_features= 2*self.BB_feature,
                            out_features=4 * self.BB_feature),
                nn.Mish(),
                nn.Linear(in_features=4 * self.BB_feature,
                            out_features=DATA_stream*RF_chain),
                )




    def forward(self, feedback): #feedback的维度是[batch_size,Pilot_num*Pilot_subcarrier*DATA_stream × 2]
        batch_size = feedback.shape[0]
        BB_feature,RF_feature = self.inital_layer(feedback)
        BB_feature,RF_feature = self.layer1(BB_feature,RF_feature)
        BB_feature,RF_feature = self.layer2(BB_feature,RF_feature)
        BB_feature,RF_feature = self.layer3(BB_feature,RF_feature)

        RF_theta = self.analog(RF_feature) #维度是[batch_size,antenna*RF_chain]
        BB_real = self.digital_real(BB_feature) #维度是[batch_size,subcarrier,antenna*RF_chain]
        BB_imag = self.digital_imag(BB_feature) #维度是[batch_size,subcarrier,antenna*RF_chain]


        return RF_theta,BB_real,BB_imag


class Beamformer_Network(nn.Module): #总体网络
    B = 16
    
    def __init__(self, BS_antenna: int, UE_antenna: int , RF_chain: int,DATA_stream: int,Subcarrier: int, Subcarrier_gap: int, Pilot_num: int,Feedback_feature: int,Feedback_bit: int ,noise_power = 0.0,estimate_noise_power=0.0, norm_factor = 1.0) -> None:
        super(Beamformer_Network, self).__init__()
        
        self.BS_antenna = BS_antenna #基站天线数量
        self.UE_antenna = UE_antenna #用户端天线数量
        self.RF_chain = RF_chain #RF chain数量
        self.DATA_stream = DATA_stream #数据流数量
        self.Subcarrier = Subcarrier #子载波总数
        self.Subcarrier_gap = Subcarrier_gap #发射pilot的子载波之间的间隔数

        self.Pilot_subcarrier_index = range(0,self.Subcarrier,self.Subcarrier_gap) #发射pilot的子载波序号
        self.Pilot_subcarrier = int(self.Subcarrier/self.Subcarrier_gap) #发射pilot的子载波数
        self.Pilot_num = Pilot_num #每次发射的pilot的数量
        
        self.estimate_noise_power = estimate_noise_power #估计信道功率
        self.noise_power = noise_power #噪声功率
        self.norm_factor = norm_factor #信道归一化系数

        self.BS_scale = np.sqrt(BS_antenna) #基站端归一化系数
        self.UE_scale = np.sqrt(UE_antenna) #用户端归一化系数


        self.Pilot = Hybrid(BS_antenna=BS_antenna, UE_antenna=UE_antenna, RF_chain=RF_chain, DATA_stream=DATA_stream, Subcarrier=Subcarrier, Subcarrier_gap=Subcarrier_gap, Pilot_num=Pilot_num) #用于训练pilot的结构

        self.Feedback_feature = Feedback_feature #反馈时量化前的feature数量


        self.compress = nn.Linear(in_features=Feedback_feature,out_features=int(Feedback_bit/4))
        self.decompress = nn.Linear(in_features=int(Feedback_bit/4),out_features=Feedback_feature)


        #    
        #这里还需要加入一个用于量化的网络
        #
        self.quantizer = QuantizationLayer(4)
        self.dequantizer = DequantizationLayer(4)

        #接下来是图网络！！！！！！！
        self.BS_GNN = GNN_Network(antenna=BS_antenna,RF_chain=RF_chain,DATA_stream=DATA_stream,Subcarrier=Subcarrier,Feedback_feature=Feedback_feature)
        self.UE_GNN = GNN_Network(antenna=UE_antenna,RF_chain=RF_chain,DATA_stream=DATA_stream,Subcarrier=Subcarrier,Feedback_feature=Feedback_feature)


    def forward(self, H_real,H_imag): #H的维度是[batchsize,subcarrier,UE_antenna,BS_antenna], s的维度是[batchsize,subcarrier,DATA_stream]

        signal_real,signal_imag = self.Pilot(H_real[:,self.Pilot_subcarrier_index,:,:], H_imag[:,self.Pilot_subcarrier_index,:,:] )

        noise_real = (torch.normal(0,1, size=signal_real.size()) * torch.sqrt(torch.tensor([self.estimate_noise_power/2])) ).cuda() 
        noise_imag = (torch.normal(0,1, size=signal_imag.size()) * torch.sqrt(torch.tensor([self.estimate_noise_power/2])) ).cuda() 

        signal_real = signal_real + noise_real  #[batch_size,Pilot_num,Pilot_subcarrier,DATA_stream]
        signal_imag = signal_imag + noise_imag

        signal_real = torch.reshape(signal_real, (-1, self.Pilot_subcarrier * self.Pilot_num * self.DATA_stream)) #合并两个维度，维度变成[batch_size,Pilot_num × Pilot_subcarrier × DATA_stream]
        signal_imag = torch.reshape(signal_imag, (-1, self.Pilot_subcarrier * self.Pilot_num * self.DATA_stream))

        Feedback = torch.cat((signal_real,signal_imag),dim=-1) #这里的feedback维度是[batch_size,Pilot_subcarrier × Pilot_num × DATA_stream × 2]

        # Feedback = torch.cat((signal_real,signal_imag),dim=-1)

        #接下来输送到BS的信号理论上需要先经过量化和逆量化
        # temp = torch.max(torch.norm(Feedback, dim=-1))
        # if temp > 8:
        #     a = 0
        Feedback_Q = self.compress(Feedback)
        Feedback_Q = self.quantizer(Feedback_Q)
        Feedback_Q = self.dequantizer(Feedback_Q)
        Feedback_Q = self.decompress(Feedback_Q)
        

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #接下来要过图网络！！！
        FRF_theta, FBB_real, FBB_imag = self.BS_GNN(Feedback_Q) #FRF_theta[batch_size,UE_anteena*RF_chain],FBB[batch_size,subcarrier,RF_chain*DATA_stream]
        WRF_theta, WBB_real, WBB_imag = self.UE_GNN(Feedback) #WRF_theta[batch_size,UE_anteena*RF_chain],WBB[batch_size,subcarrier,RF_chain*DATA_stream]


        FBB_real = torch.reshape(FBB_real,(-1,self.Subcarrier,self.RF_chain,self.DATA_stream))
        FBB_imag = torch.reshape(FBB_imag,(-1,self.Subcarrier,self.RF_chain,self.DATA_stream)) #[batchsize,subcarrier,RF_chain,DATA_stream]

        FRF_theta = torch.reshape(FRF_theta,(-1, 1, self.BS_antenna, self.RF_chain))
        FRF_real = (1 / self.BS_scale) * torch.cos(FRF_theta)  #BS端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]
        FRF_imag = (1 / self.BS_scale) * torch.sin(FRF_theta)  #BS端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]
        
        BS_real,BS_imag = matrix_product(FRF_real,FRF_imag,FBB_real,FBB_imag) #没有施行归一化，[batchsize,subcarrier,BS_antenna,DATA_stream]

        F_complex = BS_real + 1j*BS_imag
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是[Pilot_num,Pilot_subcarrier->1,1,1]
        BS_real,BS_imag = BS_real/torch.sqrt(power_F)*np.sqrt(H_real.shape[1]),BS_imag/torch.sqrt(power_F)*np.sqrt(H_imag.shape[1])

        F_complex = BS_real + 1j*BS_imag
        power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是[Pilot_num,Pilot_subcarrier->1,1,1]
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        WBB_real = torch.reshape(WBB_real,(-1,self.Subcarrier,self.DATA_stream,self.RF_chain))
        WBB_imag = torch.reshape(WBB_imag,(-1,self.Subcarrier,self.DATA_stream,self.RF_chain))

        WRF_theta = torch.reshape(WRF_theta,(-1,1,self.RF_chain,self.UE_antenna))
        WRF_real = (1 / self.UE_scale) * torch.cos(WRF_theta)  #BS端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]
        WRF_imag = (1 / self.UE_scale) * torch.sin(WRF_theta)  #BS端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]

        UE_real,UE_imag = matrix_product(WBB_real,WBB_imag,WRF_real,WRF_imag) #没有施行归一化，[batchsize,subcarrier,BS_antenna,DATA_stream]

        W_complex = UE_real + 1j*UE_imag
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是[Pilot_num,Pilot_subcarrier->1,1,1]
        UE_real,UE_imag = UE_real/torch.sqrt(power_W)*np.sqrt(H_real.shape[1])*np.sqrt(self.DATA_stream),UE_imag/torch.sqrt(power_W)*np.sqrt(H_imag.shape[1])*np.sqrt(self.DATA_stream)

        W_complex = UE_real + 1j*UE_imag
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是[Pilot_num,Pilot_subcarrier->1,1,1]
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        #接下来计算symbol rate~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        H_complex = H_real + 1j*H_imag

        WHF = torch.matmul(W_complex,torch.matmul(H_complex,F_complex))

        A = self.noise_power  #噪声功率
        

        signal_SNR =  torch.matmul( WHF , conj_T(WHF) ) / A #维度[batchsize,subcarrier,DATA_stream,DATA_stream]

        W_WH_inverse = torch.linalg.pinv( torch.matmul(W_complex,conj_T(W_complex)) )

        signal_SNR = torch.matmul(W_WH_inverse,signal_SNR)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # U,S,V_H = torch.linalg.svd(W_WH_inverse) #维度 U[batchsize,Subcarrier,UE_antenna,UE_antenna], V_H[batchsize,Subcarrier,BS_antenna,BS_antenna], S[batchsize,Subcarrier,min(UE_antenna,BS_antenna)]

        # U1,S1,V_H1 = torch.linalg.svd(signal_SNR)

        # U2,S2,V_H2 = torch.linalg.svd(torch.eye(self.DATA_stream,self.DATA_stream).cuda() + signal_SNR)

        # signal_SNR_real = torch.real(torch.eye(self.DATA_stream,self.DATA_stream).cuda() + signal_SNR)

        # signal_SNR_imag = torch.imag(torch.eye(self.DATA_stream,self.DATA_stream).cuda() + signal_SNR) 

        # tap = torch.real( torch.det( torch.eye(self.DATA_stream,self.DATA_stream).cuda() + signal_SNR))

        # temp =  torch.log2( torch.real( torch.det( torch.eye(self.DATA_stream,self.DATA_stream).cuda() + signal_SNR))) 

        # if (torch.isnan(temp).sum()>0):
        #     print("here!")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        U,S,V_H = torch.linalg.svd(torch.eye(self.DATA_stream,self.DATA_stream).cuda() + signal_SNR)
        temp = torch.prod(S,dim=-1,keepdim=False)
        RATE =  torch.mean( torch.log2( torch.real( torch.prod(S,dim=-1,keepdim=False) )      ) )  #维度[batchsize,subcarrier]

        # RATE =  torch.mean( torch.log2( torch.real( torch.det( torch.eye(self.DATA_stream,self.DATA_stream).cuda() + signal_SNR))) )  #维度[batchsize,subcarrier]

        # if RATE == 'nan':
        #     a = 0



        return RATE

    
def Modulation(bits): #bits的维度是[batchsize,subcarrier,DATA_stream,2],这里的2是4QAM
    
    return 0.7071 * (2 * bits[:,:,:, 0] - 1) , 0.7071 * (2 * bits[:,:,:, 1] - 1)  # This is just for 4QAM modulation

def deModulation(symbol_real,symbol_imag): #symbol的维度是[batchsize,subcarrier,DATA_stream],这里是4QAM,采用硬判决，直接查看正负
    bit_real = torch.unsqueeze(symbol_real>0,dim=-1)
    bit_imag = torch.unsqueeze(symbol_imag>0,dim=-1)

    return torch.cat((bit_real,bit_imag),dim=-1)# This is just for 4QAM modulation


def Model_fit(model, h_train, h_val,  opt, scheduler,  EPOCHS,dataset_name , Pilot_num,bit,noise_power_dBm):
    optimizer = opt 
    train_loss_hist = []
    val_loss_hist = []
    max_rate = -np.inf

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_rate = 0
        for batch_idx,(h_real,h_imag) in enumerate(h_train):
            h_real_batch = h_real.float()
            h_imag_batch = h_imag.float()
            optimizer.zero_grad()
            
            RATE = model(h_real_batch,h_imag_batch)

            

            loss = -RATE
            loss.backward()
            optimizer.step()

        

            train_loss += loss.detach().item()
            
            train_rate += RATE
        train_loss /= batch_idx + 1
        train_rate /= batch_idx + 1
        
    
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            val_rate = 0
            for batch_idx,(h_real,h_imag) in enumerate(h_val):
                h_real_batch = h_real.float()
                h_imag_batch = h_imag.float()

                RATE = model(h_real_batch,h_imag_batch)

                loss = -RATE 

                val_loss += loss.detach().item()
                
                val_rate += RATE
            val_loss /= batch_idx + 1
            val_rate /= batch_idx + 1


            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)

            print('Epoch : {}, Training loss = {:.4f}, Training Rate = {:.8f}, , Validation loss = {:.4f}, Validation Rate = {:.8f}.'
            .format(epoch,train_loss,train_rate,val_loss,val_rate))
        

            if RATE>max_rate:
                max_rate = RATE
                print('Save Model')
                learnable_model_savefname = '/home/yangjunyi/NAIC/Saved Models/GNNAE_{}Dataset_{}Pilot_{}bit_{}dBm_noise.pt'.format(dataset_name, Pilot_num,bit,noise_power_dBm)
                torch.save(model.state_dict(),learnable_model_savefname)


        scheduler.step()

    return train_loss_hist, val_loss_hist

def Model_eval(model, h_val):

    
    val_rate = 0
    model.eval()
    for batch_idx,(h_real,h_imag) in enumerate(h_val):
        h_real_batch = h_real.float()
        h_imag_batch = h_imag.float()

        RATE = model(h_real_batch,h_imag_batch)
                
        val_rate += RATE
    val_rate /= batch_idx + 1

    print('Rate = {:.8f}'.format(val_rate))
    return val_rate
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
from utils_network import *




class GNN_inital_layer(nn.Module): 
    def __init__(self, antenna: int,  RF_chain: int, DATA_stream: int, Subcarrier: int,Subcarrier_group: int,Feedback_feature: int):  #[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain*2]
        #每个BB的维度是[DATA_stream,RF_chain],每个RF的维度是[antenna,RF_chain]
        super(GNN_inital_layer, self).__init__()
        self.subcarrier = Subcarrier
        self.Feedback_feature = Feedback_feature
        self.DATA_stream = DATA_stream 
        self.BB_feature = 2*RF_chain*DATA_stream #有实部和虚部
        self.RF_feature = 2*antenna*RF_chain #有实部和虚部


        self.BB_net = nn.Sequential(
                nn.Linear(in_features =  self.Feedback_feature,
                            out_features= Subcarrier_group*self.BB_feature),
                nn.Mish(),
                )
        
        self.RF_net = nn.Sequential(
                nn.Linear(in_features =  self.Feedback_feature,
                            out_features = self.RF_feature),
                nn.Mish(),
                )
        

    def forward(self, feedback): #这里的feedback维度是[batch_size, Pilot_subcarrier,Pilot_num × RF_chain × 2]，需要变成subcarrier个向量

        BB_feature = self.BB_net(feedback) #[batch_size, Subcarrier_group_num, self.BB_feature]
        BB_feature = torch.reshape(BB_feature,(-1, self.subcarrier, self.BB_feature)) #[batch_size, Subcarrier, self.BB_feature]
        RF_feature = self.RF_net(torch.mean(feedback,dim=-2)) #[batch_size,  self.RF_feature]

        # BB_feature = feedback #[batch_size, Pilot_subcarrier,Pilot_num × RF_chain × 2]
        # RF_feature = torch.mean(BB_feature,dim= -2 ,keepdim=False) #维度是[batch_size,Pilot_num × RF_chain × 2]

        return BB_feature,RF_feature


class GNN_layer(nn.Module): #GNN网络的迭代层#[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain*2]
    def __init__(self, antenna: int,  RF_chain: int, DATA_stream: int,Subcarrier: int,Subcarrier_group: int,Feedback_feature: int): 
        #每个BB的维度是[DATA_stream,RF_chain],每个RF的维度是[antenna,RF_chain]
        super(GNN_layer, self).__init__()
        self.subcarrier = Subcarrier
        self.input_dim = Feedback_feature
        self.DATA_stream = DATA_stream 
        self.BB_feature = 2*RF_chain*DATA_stream #有实部和虚部
        self.RF_feature = 2*antenna*RF_chain #有实部和虚部
        self.feedback_feature = Feedback_feature

        self.RF_BB = nn.Sequential(
                nn.Linear(in_features = self.RF_feature,
                            out_features= self.BB_feature),
                nn.Mish(),
                )
        self.RF_RF = nn.Sequential(
                nn.Linear(in_features = self.RF_feature,
                            out_features= self.RF_feature),
                nn.Mish(),
                )
        self.BB_RF = nn.Sequential(
                nn.Linear(in_features = self.BB_feature,
                            out_features= self.RF_feature),
                nn.Mish(),
                )
        self.BB_BB = nn.Sequential(
                nn.Linear(in_features = self.BB_feature,
                            out_features= self.BB_feature),
                nn.Mish(),
                )

    def forward(self, BB_feature,RF_feature): #BB_feature的维度是[batch_size, Pilot_subcarrier,2 * self.BB_feature], RF_feature的维度是[batch_size, 2 * self.RF_feature]
        batch_size = BB_feature.shape[0]
        # new_BB_feature = (torch.zeros(size=(batch_size, self.subcarrier,2*self.BB_feature))).cuda()
        BB_mean = torch.mean(BB_feature,dim= -2 ,keepdim=False) #维度是[batch_size,self.BB_feature]

        RF_to_RF = self.RF_RF(RF_feature) #[batch_size,RF_feature]
        RF_to_BB = self.RF_BB(RF_feature) #[batch_size,BB_feature]
        BB_to_RF = self.BB_RF(BB_mean)  #[batch_size,RF_feature]
        BB_to_BB = self.BB_BB(BB_feature) #[batch_size,subcarrier,BB_feature]

        new_RF_feature = RF_to_RF + BB_to_RF
        new_BB_feature = BB_to_BB + torch.unsqueeze(RF_to_BB,dim=-2)

        return new_BB_feature,new_RF_feature
    
class GNN_Network(nn.Module): #GNN网络的迭代层
    def __init__(self, antenna: int,  RF_chain: int,DATA_stream: int,Subcarrier: int,Subcarrier_group: int,Feedback_feature: int):
        super(GNN_Network, self).__init__()
        self.antenna = antenna
        self.feedback_feature = Feedback_feature
        self.subcarrier = Subcarrier
        self.input_dim = Feedback_feature
        self.RF_chain = RF_chain
        self.DATA_stream = DATA_stream 
        self.BB_feature = 2*RF_chain*DATA_stream #有实部和虚部
        self.RF_feature = 2*antenna*RF_chain #有实部和虚部

        self.inital_layer = GNN_inital_layer(antenna=antenna, RF_chain=RF_chain,DATA_stream=DATA_stream, Subcarrier=Subcarrier,Subcarrier_group=Subcarrier_group, Feedback_feature=Feedback_feature)
        self.layer1 = GNN_layer(antenna=antenna, RF_chain=RF_chain,DATA_stream=DATA_stream, Subcarrier=Subcarrier,Subcarrier_group=Subcarrier_group,Feedback_feature=Feedback_feature)
        self.layer2 = GNN_layer(antenna=antenna, RF_chain=RF_chain,DATA_stream=DATA_stream, Subcarrier=Subcarrier,Subcarrier_group=Subcarrier_group,Feedback_feature=Feedback_feature)
        self.layer3 = GNN_layer(antenna=antenna, RF_chain=RF_chain,DATA_stream=DATA_stream, Subcarrier=Subcarrier,Subcarrier_group=Subcarrier_group,Feedback_feature=Feedback_feature)
        # self.layer4 = GNN_layer(antenna=antenna, RF_chain=RF_chain,DATA_stream=DATA_stream, Subcarrier=Subcarrier,Subcarrier_group=Subcarrier_group,Feedback_feature=Feedback_feature)
        # self.layer5 = GNN_layer(antenna=antenna, RF_chain=RF_chain,DATA_stream=DATA_stream, Subcarrier=Subcarrier,Subcarrier_group=Subcarrier_group,Feedback_feature=Feedback_feature)
        


        self.analog_real = nn.Sequential(
            nn.Linear(in_features= self.RF_feature,
                        out_features= antenna*RF_chain),
        )

        self.analog_imag = nn.Sequential(
            nn.Linear(in_features= self.RF_feature,
                        out_features= antenna*RF_chain),
        )


        self.digital_real = nn.Sequential(
                nn.Linear(in_features= self.BB_feature,
                            out_features=RF_chain*DATA_stream),
                )

        self.digital_imag = nn.Sequential(
                nn.Linear(in_features= self.BB_feature,
                            out_features=RF_chain*DATA_stream),
                )




    def forward(self, feedback): #[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain*2]
        batch_size = feedback.shape[0]
        BB_feature,RF_feature = self.inital_layer(feedback)
        BB_feature,RF_feature = self.layer1(BB_feature,RF_feature)
        BB_feature,RF_feature = self.layer2(BB_feature,RF_feature)
        BB_feature,RF_feature = self.layer3(BB_feature,RF_feature)
        BB_feature,RF_feature = self.layer4(BB_feature,RF_feature)
        # BB_feature,RF_feature = self.layer5(BB_feature,RF_feature)

        RF_real = RF_feature[:,:self.antenna*self.RF_chain]
        RF_imag = RF_feature[:,self.antenna*self.RF_chain:]

        BB_real = BB_feature[:,:,:self.RF_chain*self.DATA_stream] 
        BB_imag = BB_feature[:,:,self.RF_chain*self.DATA_stream:] 


        # RF_real = self.analog_real(RF_feature) #维度是[batch_size,antenna*RF_chain]
        # RF_imag = self.analog_imag(RF_feature) #维度是[batch_size,antenna*RF_chain]
        # BB_real = self.digital_real(BB_feature) #维度是[batch_size,subcarrier,antenna*RF_chain]
        # BB_imag = self.digital_imag(BB_feature) #维度是[batch_size,subcarrier,antenna*RF_chain]


        return RF_real, RF_imag,BB_real,BB_imag


class Beamformer_Network(nn.Module): #总体网络

    
    def __init__(self, BS_antenna: int, UE_antenna: int , BS_RF_chain: int,UE_RF_chain: int,DATA_stream: int,Subcarrier: int, Subcarrier_group: int, Pilot_num: int,Feedback_feature: int,vq_dim: int,vq_b:int,s_bit:int ,noise_power = 0.0,estimate_noise_power=0.0, norm_factor = 1.0) -> None:
        super(Beamformer_Network, self).__init__()
        self.B = 2 #量化bit
        self.BS_antenna = BS_antenna #基站天线数量
        self.UE_antenna = UE_antenna #用户端天线数量
        self.BS_RF_chain = BS_RF_chain #BS RF chain数量
        self.UE_RF_chain = UE_RF_chain #BS RF chain数量
        self.DATA_stream = DATA_stream
        
        self.Subcarrier = Subcarrier #子载波总数
        self.Subcarrier_group = Subcarrier_group #每个subcarrier group中subcarrier的数量
        self.Subcarrier_group_num = int(self.Subcarrier/self.Subcarrier_group) #subcarrier group 的数量

        # self.Pilot_subcarrier_index = range(0,self.Subcarrier,self.Subcarrier_group) #发射pilot的子载波序号
        # self.Pilot_subcarrier = int(self.Subcarrier/self.Subcarrier_group) #发射pilot的子载波数
        self.Pilot_num = Pilot_num #每次发射的pilot的数量
        
        self.estimate_noise_power = estimate_noise_power #估计信道功率
        self.noise_power = noise_power #噪声功率
        self.norm_factor = norm_factor #信道归一化系数

        self.BS_scale = np.sqrt(BS_antenna) #基站端归一化系数
        self.UE_scale = np.sqrt(UE_antenna) #用户端归一化系数

        
        
        self.Pilot = PhaseShifter_Pilot(BS_antenna=BS_antenna, UE_antenna=UE_antenna, BS_RF_chain=BS_RF_chain,UE_RF_chain=UE_RF_chain,  Subcarrier=Subcarrier, Subcarrier_group=Subcarrier_group, Pilot_num=Pilot_num) #用于训练pilot的结构

        self.Feedback_feature = Feedback_feature #反馈时量化前的feature数量

        self.Feedback_bit = Feedback_feature/vq_dim*vq_b #每个subcarrier的feedback_bit
        self.compress = nn.Linear(in_features=Feedback_feature,out_features=int(self.Feedback_bit/self.B))
        self.decompress = nn.Linear(in_features=int(self.Feedback_bit/self.B),out_features=Feedback_feature)

        #    
        #这里还需要加入一个用于量化的网络
        #
        self.AE = NAE(self.B)

        #    
        #这里还需要加入一个用于量化的网络
        #
        self.quantizer = VQVAE(vq_b=vq_b,vq_dim=vq_dim,s_bit=s_bit,feadback_feature=Feedback_feature)

        #总共需要的bit数为Feedback_feature/vq_dim*vq_b + s_bit
        #其中feedback_feature为(Pilot_num * Subcarrier * DATA_stream / Subcarrier_gap * 2) #为量化的反馈的向量长度
        #

        #接下来是图网络！！！！！！！
        self.BS_GNN = GNN_Network(antenna=BS_antenna,RF_chain=BS_RF_chain, DATA_stream=DATA_stream,Subcarrier=Subcarrier, Subcarrier_group=self.Subcarrier_group,Feedback_feature=Feedback_feature)
        self.UE_GNN = GNN_Network(antenna=UE_antenna,RF_chain=UE_RF_chain, DATA_stream=DATA_stream,Subcarrier=Subcarrier, Subcarrier_group=self.Subcarrier_group,Feedback_feature=Feedback_feature)


    def forward(self, H_real,H_imag): #H的维度是[batchsize,subcarrier,UE_antenna,BS_antenna], s的维度是[batchsize,subcarrier,DATA_stream]

        signal_real,signal_imag = self.Pilot(H_real, H_imag ) #[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain]

        # signal_real = torch.reshape(signal_real, (-1, self.Pilot_subcarrier , self.Pilot_num * self.RF_chain)) #合并两个维度，维度变成[batch_size, Pilot_subcarrier,Pilot_num × RF_chain]
        # signal_imag = torch.reshape(signal_imag, (-1, self.Pilot_subcarrier , self.Pilot_num * self.RF_chain))

        signal_cat = torch.cat((signal_real,signal_imag),dim=-1) #[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain*2]
        noise_vec = (torch.normal(0, 1, size=signal_cat.size()) * torch.sqrt(torch.tensor([self.estimate_noise_power / 2])) / torch.tensor([self.norm_factor])).cuda()

        Feedback = signal_cat + noise_vec #[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain*2]


        Feedback_Q, loss1 ,loss2 = self.quantizer(Feedback) #[batch_size, Subcarrier_group_num, Pilot_num*UE_RF_chain*2]
        
        # Feedback_Q = self.compress(Feedback)
        # Feedback_Q = self.AE(Feedback_Q)
        # Feedback_Q = self.decompress(Feedback_Q)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #接下来要过图网络！！！
        FRF_real, FRF_imag, FBB_real, FBB_imag = self.BS_GNN(Feedback_Q) #FRF_theta[batch_size,BS_anteena*BS_RF_chain],FBB[batch_size,Subcarrier,RF_chain*DATA_stream]
        WRF_real, WRF_imag, WBB_real, WBB_imag = self.UE_GNN(Feedback) #WRF_theta[batch_size,UE_anteena*UE_RF_chain],WBB[batch_size,Subcarrier,RF_chain*DATA_stream]

        # FBB_real = torch.unsqueeze(FBB_real,dim=2)*torch.ones(1,1,self.Subcarrier_group,1).cuda()
        # FBB_imag = torch.unsqueeze(FBB_imag,dim=2)*torch.ones(1,1,self.Subcarrier_group,1).cuda()

        FBB_real = torch.reshape(FBB_real,(-1,self.Subcarrier,self.BS_RF_chain,self.DATA_stream))
        FBB_imag = torch.reshape(FBB_imag,(-1,self.Subcarrier,self.BS_RF_chain,self.DATA_stream)) #[batchsize,subcarrier,RF_chain,DATA_stream]

        FRF_pow = torch.pow(   torch.pow(FRF_real,2)+torch.pow(FRF_imag,2),0.5 ) #[batch_size,BS_anteena*BS_RF_chain]
        FRF_real = FRF_real/FRF_pow
        FRF_imag = FRF_imag/FRF_pow

        FRF_real = torch.reshape(FRF_real,(-1, 1, self.BS_antenna, self.BS_RF_chain)) #BS端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]
        FRF_imag = torch.reshape(FRF_imag,(-1, 1, self.BS_antenna, self.BS_RF_chain)) #BS端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]


        # FRF_theta = torch.reshape(FRF_theta,(-1, 1, self.BS_antenna, self.RF_chain))
        # FRF_real = (1 / self.BS_scale) * torch.cos(FRF_theta)  #BS端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]
        # FRF_imag = (1 / self.BS_scale) * torch.sin(FRF_theta)  #BS端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]
        
        BS_real,BS_imag = matrix_product(FRF_real,FRF_imag,FBB_real,FBB_imag) #没有施行归一化，[batchsize,subcarrier,BS_antenna,DATA_stream]

        F_complex = BS_real + 1j*BS_imag
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]
        BS_real,BS_imag = BS_real/torch.sqrt(power_F)*np.sqrt(H_real.shape[1]),BS_imag/torch.sqrt(power_F)*np.sqrt(H_imag.shape[1])

        F_complex = BS_real + 1j*BS_imag
        power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # WBB_real = torch.unsqueeze(WBB_real,dim=2)*torch.ones(1,1,self.Subcarrier_group,1).cuda()
        # WBB_imag = torch.unsqueeze(WBB_imag,dim=2)*torch.ones(1,1,self.Subcarrier_group,1).cuda()

        WBB_real = torch.reshape(WBB_real,(-1,self.Subcarrier,self.DATA_stream,self.UE_RF_chain))
        WBB_imag = torch.reshape(WBB_imag,(-1,self.Subcarrier,self.DATA_stream,self.UE_RF_chain))

        WRF_pow = torch.pow(   torch.pow(WRF_real,2)+torch.pow(WRF_imag,2),0.5 ) #[batch_size,UE_anteena*RF_chain]
        WRF_real = WRF_real/WRF_pow
        WRF_imag = WRF_imag/WRF_pow

        WRF_real = torch.reshape(WRF_real,(-1, 1, self.UE_RF_chain,self.UE_antenna)) #UE端的analog matrix，维度是[batchsize,1,UE_RF_chain, UE_antenna]
        WRF_imag = torch.reshape(WRF_imag,(-1, 1, self.UE_RF_chain,self.UE_antenna)) ##UE端的analog matrix，维度是[batchsize,1,UE_RF_chain, UE_antenna]

        # WRF_theta = torch.reshape(WRF_theta,(-1,1,self.RF_chain,self.UE_antenna))
        # WRF_real = (1 / self.UE_scale) * torch.cos(WRF_theta)  #UE端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]
        # WRF_imag = (1 / self.UE_scale) * torch.sin(WRF_theta)  #UE端的analog matrix，维度是[batchsize,1,BS_antenna,RF_chain]

        UE_real,UE_imag = matrix_product(WBB_real,WBB_imag,WRF_real,WRF_imag) #没有施行归一化，[batchsize,subcarrier,BS_antenna,DATA_stream]

        W_complex = UE_real + 1j*UE_imag
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]
        UE_real,UE_imag = UE_real/torch.sqrt(power_W)*np.sqrt(H_real.shape[1])*np.sqrt(self.DATA_stream),UE_imag/torch.sqrt(power_W)*np.sqrt(H_imag.shape[1])*np.sqrt(self.DATA_stream)
        # UE_real,UE_imag = UE_real/torch.sqrt(power_W)*np.sqrt(H_real.shape[1]),UE_imag/torch.sqrt(power_W)*np.sqrt(H_imag.shape[1])

        W_complex = UE_real + 1j*UE_imag
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        #接下来计算symbol rate~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        H_complex = H_real + 1j*H_imag

        WHF = torch.matmul(W_complex,torch.matmul(H_complex,F_complex))

        A = self.noise_power/self.norm_factor/self.norm_factor  #噪声功率
        

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
        temp1 = torch.log2( torch.real( temp )      )


        
        RATE =  torch.mean( torch.log2( torch.real( torch.prod(S,dim=-1,keepdim=False) )      ) )  #维度[batchsize,subcarrier]





        return RATE,loss1,loss2

    
def Modulation(bits): #bits的维度是[batchsize,subcarrier,DATA_stream,2],这里的2是4QAM
    
    return 0.7071 * (2 * bits[:,:,:, 0] - 1) , 0.7071 * (2 * bits[:,:,:, 1] - 1)  # This is just for 4QAM modulation

def deModulation(symbol_real,symbol_imag): #symbol的维度是[batchsize,subcarrier,DATA_stream],这里是4QAM,采用硬判决，直接查看正负
    bit_real = torch.unsqueeze(symbol_real>0,dim=-1)
    bit_imag = torch.unsqueeze(symbol_imag>0,dim=-1)

    return torch.cat((bit_real,bit_imag),dim=-1)# This is just for 4QAM modulation


def Model_fit(model, h_train, h_val,  opt,scheduler,   EPOCHS,dataset_name , Pilot_num,bit,noise_power_dBm):
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
            
            RATE,loss1,loss2 = model(h_real_batch,h_imag_batch)

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            loss = -RATE + 10* loss1 + 10*loss2
            loss.backward()
            optimizer.step()

        
            train_loss += loss.detach().item()
            
            train_rate += RATE.detach().item()
        train_loss /= batch_idx + 1
        train_rate /= batch_idx + 1
        
    
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            val_rate = 0
            for batch_idx,(h_real,h_imag) in enumerate(h_val):
                h_real_batch = h_real.float()
                h_imag_batch = h_imag.float()

                RATE,loss1,loss2 = model(h_real_batch,h_imag_batch)

                loss = -RATE + 10*loss1 + 10*loss2

                val_loss += loss.detach().item()
                
                val_rate += RATE.detach().item()
            val_loss /= batch_idx + 1
            val_rate /= batch_idx + 1


            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)



            print('Epoch : {}, Training loss = {:.4f}, Training Rate = {:.8f}, , Validation loss = {:.4f}, Validation Rate = {:.8f}.'
            .format(epoch,train_loss,train_rate,val_loss,val_rate))

            if val_rate>max_rate:
                max_rate = val_rate
                print('Save Model ')
                learnable_model_savefname = '/home/yangjunyi/BeamformingRevise_V3/Saved Models/GNN_{}Dataset_{}Pilot_{}bit_{}dBm_noise.pt'.format(dataset_name, Pilot_num,bit,noise_power_dBm)
                torch.save(model.state_dict(),learnable_model_savefname)

        scheduler.step()

    return train_loss_hist, val_loss_hist

def Model_eval(model, h_val):

    
    val_rate = 0
    model.eval()
    for batch_idx,(h_real,h_imag) in enumerate(h_val):
        h_real_batch = h_real.float()
        h_imag_batch = h_imag.float()

        RATE,loss1,loss2 = model(h_real_batch,h_imag_batch)
                
        val_rate += RATE.detach().item()
    val_rate /= batch_idx + 1

    print('Rate = {:.8f}'.format(val_rate))
    return val_rate
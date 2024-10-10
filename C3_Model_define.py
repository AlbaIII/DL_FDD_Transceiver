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





class Encoder(nn.Module): #precoder的encoder层 #[batch_size, subcarrier, 2 ,UE_antenna, BS_antenna]
    def __init__(self,  Subcarrier: int, Subcarrier_group: int, feedback_bit:int ): 
        #每个BB的维度是[DATA_stream,RF_chain],每个RF的维度是[antenna,RF_chain]
        super(Encoder, self).__init__()

        self.Subcarrier = Subcarrier #子载波总数
        self.Subcarrier_group = Subcarrier_group #每个subcarrier group中subcarrier的数量
        self.Subcarrier_gap = self.Subcarrier_group
        self.Pilot_subcarrier_index = range(0,self.Subcarrier,self.Subcarrier_gap) #发射pilot的子载波序号
        self.Subcarrier_group_num = int(self.Subcarrier/self.Subcarrier_group) #subcarrier group 的数量

        self.cov = nn.Sequential(
                nn.Conv2d(in_channels = 2,
                            out_channels= 32,
                            kernel_size=[7,7]),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels = 32,
                            out_channels= 64,
                            kernel_size=[5,5]),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels = 64,
                            out_channels= 16,
                            kernel_size=[3,3]),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels = 16,
                            out_channels= 1,
                            kernel_size=[1,1]),
                nn.LeakyReLU(),
                )
        
        self.linear = nn.Sequential(
                nn.Linear(in_features = 16,
                            out_features= 16),
                )

        

    def forward(self, H): #H的维度是[batch_size, subcarrier, 2 ,UE_antenna, BS_antenna]
        x = H[:, self.Pilot_subcarrier_index ,:,:,:]
        x = torch.reshape(x, (x.shape[0], x.shape[1], 2,16,16) ) #维度[batch_size, subcarrier_group,2, 16, 16]

        x = torch.reshape(x, (x.shape[0]* x.shape[1], 2,16,16)) #维度[batch_size * subcarrier_group , 2, 16, 16]
        x = self.cov(x) #维度[batch_size*subcarrier_group,1, 4,4]
        x = torch.reshape(x, (H.shape[0], -1 , 1,4,4) ) #维度[batch_size , subcarrier_group , 1, 4, 4]

        x = torch.reshape(x , (x.shape[0], x.shape[1], -1) ) #维度[batch_size, subcarrier_group, 16]
        x = self.linear(x) #维度[batch_size, subcarrier_group, 1]
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1)) #维度[batch_size, subcarrier_group, 1]
        
        return x
    
class Decoder_precoder(nn.Module): #GNN网络的迭代层
    def __init__(self, antenna: int,  RF_chain: int, feedback_bit:int):
        super(Decoder_precoder, self).__init__()
        
        self.inital = nn.Linear(in_features=16,out_features=8)
        self.cov = nn.Sequential(
                nn.Conv2d(in_channels = 2,
                            out_channels= 32,
                            kernel_size=[5,5],
                            padding=2),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels = 32,
                            out_channels= 8,
                            kernel_size=[3,3],
                            padding=1),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels = 8,
                            out_channels= 2,
                            kernel_size=[1,1]),
                nn.LeakyReLU(),
                )
        
        self.analog_real = nn.Sequential(
            nn.Linear(in_features= 128,
                        out_features= antenna*RF_chain),
        )

        self.analog_imag = nn.Sequential(
            nn.Linear(in_features= 128,
                        out_features= antenna*RF_chain),
        )

    def forward(self, feedback): #[batch_size, Subcarrier_group, 32]
    
        feedback = self.inital(feedback)
        x = torch.reshape(feedback, (feedback.shape[0], 2,8,8)) #[batch_size, 2,8,8]

        x = torch.reshape(x, (x.shape[0], 2,8,8)) #[batch_size,2 ,8,8]
        x = self.cov(x) #[batch_size, 2,8,8]
        x = torch.reshape(x, (feedback.shape[0], 2,8,8)) #[batch_size,2 ,8,8]

        x = torch.reshape(x, (x.shape[0], -1)) #[batch_size, 128]

        analog_real = self.analog_real(x)
        analog_imag = self.analog_imag(x)

        return analog_real, analog_imag

class Decoder_combiner(nn.Module): #GNN网络的迭代层
    def __init__(self, antenna: int,  RF_chain: int, feedback_bit:int):
        super(Decoder_combiner, self).__init__()

        self.inital = nn.Linear(in_features=16,out_features=8)
        self.cov = nn.Sequential(
                nn.Conv2d(in_channels = 2,
                            out_channels= 32,
                            kernel_size=[5,5],
                            padding=2),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels = 32,
                            out_channels= 8,
                            kernel_size=[3,3],
                            padding=1),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels = 8,
                            out_channels= 2,
                            kernel_size=[1,1]),
                nn.LeakyReLU(),
                )
        
        self.analog_real = nn.Sequential(
            nn.Linear(in_features= 128,
                        out_features= antenna*RF_chain),
        )

        self.analog_imag = nn.Sequential(
            nn.Linear(in_features= 128,
                        out_features= antenna*RF_chain),
        )

    def forward(self, feature): #[batch_size, Subcarrier_group, 32]

        feature = self.inital(feature)
        x = torch.reshape(feature, (feature.shape[0], 2,8,8)) #[batch_size, 2,8,8]

        x = torch.reshape(x, (x.shape[0], 2,8,8)) #[batch_size, 2,8,8]
        x = self.cov(x) #[batch_size, 2,8,8]
        x = torch.reshape(x, (feature.shape[0], 2,8,8)) #[batch_size, 2,8,8]

        x = torch.reshape(x, (x.shape[0], -1)) #[batch_size, 128]

        analog_real = self.analog_real(x)
        analog_imag = self.analog_imag(x)

        return analog_real, analog_imag


class Beamformer_Network(nn.Module): #总体网络

    
    def __init__(self, BS_antenna: int, UE_antenna: int , BS_RF_chain: int,UE_RF_chain: int, DATA_stream:int ,Subcarrier: int, Subcarrier_group: int,feedback_bit:int,  noise_power = 0.0,norm_factor = 1.0) -> None:
        super(Beamformer_Network, self).__init__()
        self.BS_antenna = BS_antenna
        self.BS_RF_chain = BS_RF_chain

        self.UE_antenna = UE_antenna
        self.UE_RF_chain = UE_RF_chain

        self.DATA_stream = DATA_stream

        self.noise_power = noise_power
        self.norm_factor = norm_factor

        self.encoder = Encoder(Subcarrier = Subcarrier, Subcarrier_group = Subcarrier_group,feedback_bit=feedback_bit)
        self.quantizer = NAE(4)

        self.decoder_precoder = Decoder_precoder(antenna = BS_antenna,  RF_chain = BS_RF_chain,feedback_bit=feedback_bit)
        self.decoder_combiner = Decoder_combiner(antenna = UE_antenna,  RF_chain = UE_RF_chain,feedback_bit=feedback_bit)

    def forward(self, H_cat): #维度[batch_size,subcarrier, 2 ,UE_antenna, BS_antenna]

        feature = self.encoder(H_cat) #维度[batch_size, subcarrier_group, 8]
        feature_bit = self.quantizer(feature) #维度[batch_size, subcarrier_group, 32]

        FRF_real, FRF_imag = self.decoder_precoder(feature_bit)
        WRF_real, WRF_imag = self.decoder_combiner(feature)

        FRF_pow = torch.pow(   torch.pow(FRF_real,2)+torch.pow(FRF_imag,2),0.5 ) #[batch_size,BS_anteena*BS_RF_chain]
        FRF_real = FRF_real/FRF_pow
        FRF_imag = FRF_imag/FRF_pow
        FRF_real = torch.reshape(FRF_real,(-1, 1, self.BS_antenna, self.BS_RF_chain)) #BS端的analog matrix，维度是[batchsize,1,BS_antenna,BS_RF_chain]
        FRF_imag = torch.reshape(FRF_imag,(-1, 1, self.BS_antenna, self.BS_RF_chain)) #BS端的analog matrix，维度是[batchsize,1,BS_antenna,BS_RF_chain]

        WRF_pow = torch.pow(   torch.pow(WRF_real,2)+torch.pow(WRF_imag,2),0.5 ) #[batch_size,UE_anteena*RF_chain]
        WRF_real = WRF_real/WRF_pow
        WRF_imag = WRF_imag/WRF_pow
        WRF_real = torch.reshape(WRF_real,(-1, 1, self.UE_RF_chain,self.UE_antenna)) #UE端的analog matrix，维度是[batchsize,1,UE_RF_chain, UE_antenna]
        WRF_imag = torch.reshape(WRF_imag,(-1, 1, self.UE_RF_chain,self.UE_antenna)) ##UE端的analog matrix，维度是[batchsize,1,UE_RF_chain, UE_antenna]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        H_real = torch.squeeze(H_cat[:,:,0,:,:])
        H_imag = torch.squeeze(H_cat[:,:,1,:,:])
        H_complex = H_real + 1j*H_imag  #维度[batch_size,subcarrier,UE_antenna, BS_antenna]

        FRF = (FRF_real + 1j*FRF_imag) #维度是[batchsize,1,BS_antenna,BS_RF_chain]
        WRF = WRF_real + 1j*WRF_imag #维度是[batchsize,1,UE_RF_chain, UE_antenna]

        #接下来计算symbol rate~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        H_eq = torch.matmul(WRF,torch.matmul(H_complex,FRF)) #维度是[batchsize,subcarrier,UE_RF_chain, BS_RF_chain]
        U,S,V_H = torch.linalg.svd(H_eq) #维度 U[batchsize,Subcarrier,UE_RF_chain,UE_RF_chain], V_H[batchsize,Subcarrier,BS_RF_chain,BS_RF_chain], S[batchsize,Subcarrier,min(UE_RF_chain, BS_RF_chain)]
        A = self.noise_power/self.norm_factor/self.norm_factor  #噪声功率

        FBB = V_H.conj().permute(0,1,-1,-2)[:,:,:,0:self.DATA_stream] #维度[batchsize,Subcarrier,BS_RF_chain,DATA_stream]
        F_complex = torch.matmul(FRF, FBB)
        power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]
        F_complex = F_complex/torch.sqrt(power_F)*np.sqrt(F_complex.shape[1])
        power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]


        WBB = U.conj().permute(0,1,-1,-2)[:,:,0:self.DATA_stream,:] #维度[batchsize,Subcarrier,DATA_stream,UE_RF_chain] #用SVD去做！检验代码写对没有！
        W_complex = torch.matmul(WBB, WRF)
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]
        W_complex = W_complex/torch.sqrt(power_W)*np.sqrt(W_complex.shape[1])*np.sqrt(self.DATA_stream)
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]

        #注水算法~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # gamma_matrix = water_filling(H_eq,self.DATA_stream,A) #注水求得注水算法矩阵，维度为[batch_size,subcarrier_gap,1,DATA_stream]
        # F_complex = F_complex * torch.sqrt(gamma_matrix)
        # power_F =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]




        WHF = torch.matmul(W_complex,torch.matmul(H_complex,F_complex)) #维度是[batchsize,subcarrier,UE_RF_chain, BS_RF_chain]

        signal_SNR =  torch.matmul( WHF , conj_T(WHF) ) / A #维度[batchsize,subcarrier,UE_RF_chain,UE_RF_chain]

        W_WH_inverse = torch.linalg.pinv( torch.matmul(W_complex,conj_T(W_complex)) ) #维度[batchsize,1,UE_RF_chain,UE_RF_chain]

        signal_SNR = torch.matmul(W_WH_inverse,signal_SNR) #维度[batchsize,subcarrier,UE_RF_chain,UE_RF_chain]

        U,S,V_H = torch.linalg.svd(torch.eye(self.UE_RF_chain,self.UE_RF_chain).cuda() + signal_SNR)

        
        RATE =  torch.mean( torch.log2( torch.real( torch.prod(S,dim=-1,keepdim=False) )      ) )  #维度[batchsize,subcarrier]


        return RATE

    
def Modulation(bits): #bits的维度是[batchsize,subcarrier,DATA_stream,2],这里的2是4QAM
    
    return 0.7071 * (2 * bits[:,:,:, 0] - 1) , 0.7071 * (2 * bits[:,:,:, 1] - 1)  # This is just for 4QAM modulation

def deModulation(symbol_real,symbol_imag): #symbol的维度是[batchsize,subcarrier,DATA_stream],这里是4QAM,采用硬判决，直接查看正负
    bit_real = torch.unsqueeze(symbol_real>0,dim=-1)
    bit_imag = torch.unsqueeze(symbol_imag>0,dim=-1)

    return torch.cat((bit_real,bit_imag),dim=-1)# This is just for 4QAM modulation


def Model_fit(model, h_train, h_val, opt,scheduler, EPOCHS,dataset_name,bit ,noise_power_dBm):
    optimizer = opt 
    train_loss_hist = []
    val_loss_hist = []
    max_rate = -np.inf

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_rate = 0
        for batch_idx,(H_cat) in enumerate(h_train):
            H_cat = H_cat[0].float()
            optimizer.zero_grad()
            
            RATE = model(H_cat)

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            loss = -RATE
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
            for batch_idx,(H_cat) in enumerate(h_val):
                H_cat = H_cat[0].float()

                RATE = model(H_cat)

                loss = -RATE

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
                learnable_model_savefname = '/home/yangjunyi/BeamformingRevise_V3/Saved Models/C3_{}Dataset_{}bit_{}dBm_noise.pt'.format(dataset_name,bit,noise_power_dBm)
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
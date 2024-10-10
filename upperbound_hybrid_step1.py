from logging import BASIC_FORMAT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import scipy.io
from sklearn.model_selection import train_test_split
from utils import *

#纯数字没有数据流限制
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BS_antenna = 64
UE_antenna = 4
BS_RF_chain = 4 #RF_chain数量，仅当采用Hybird系统时有用
UE_RF_chain = 2
DATA_stream = UE_RF_chain #数据流数量
Subcarrier = 128  #载波数量 

Subcarrier_gap = 1  #发送pilot的载波间隔，得到的数据量为Subcarrier/Subcarrier_gap*Pilot_num*DATA_stream
Pilot_num = 16 #载波数量
Feedback_feature = int(Pilot_num  * DATA_stream * 2) #为量化的反馈的向量长度

estimate_noise_power_dbm = -94
all_noise_power_dBm = [-79,-84,-89,-94,-99,-104,-109,-114]
noise_factor = -13 #dB

noiseless = False
nepoch = 101
batch_size = 128



# 选择数据集————————————————————————————————————————————————————————————————————————————————————————————————
dataset_name = 'I3_60_ULA' #O1_28_ULA或I3_60_ULA
# fname_h_real = '/home/yangjunyi/BeamformingRevise_V2/Dataset/' + dataset_name + '/h_real.mat'
# fname_h_imag = '/home/yangjunyi/BeamformingRevise_V2/Dataset/' + dataset_name + '/h_imag.mat'
# UE_location = '/home/yangjunyi/BeamformingRevise_V2/Dataset/' + dataset_name + '/ue_loc.mat'
fname_h_real = '/home/yangjunyi/BeamformingRevise_V2/Dataset/h_real.mat'
fname_h_imag = '/home/yangjunyi/BeamformingRevise_V2/Dataset/h_imag.mat'
UE_location = '/home/yangjunyi/BeamformingRevise_V2/Dataset/ue_loc.mat'

h_real = np.float32(np.transpose(h5py.File(fname_h_real)['h_real']))
h_imag = np.float32(np.transpose(h5py.File(fname_h_imag)['h_imag']))
UE_location = np.float32(np.transpose(h5py.File(UE_location)['ue_loc']))
tx_power_dBm = 10

# #天线数量均为1时的预处理
# h_real = h_real.reshape(h_real.shape[0],h_real.shape[1],1,1)
# h_imag = h_imag.reshape(h_imag.shape[0],h_imag.shape[1],1,1)




# 数据预处理（实部虚部结合、归一化）————————————————————————————————————————————————————————————————————————————————————————————————
h = h_real + 1j * h_imag  # h是复数，维度[N,subcarrier,UE_antenna,BS_antenna]
norm_factor = np.max(abs(h))  # 模的最大值，也是之后的归一化系数


# 数据集分割（训练集、验证集、测试集）————————————————————————————————————————————————————————————————————————————————————————————————
train_idc, test_idc = train_test_split(np.arange(h.shape[0]), test_size=0.4, random_state=11037)
val_idc, test_idc = train_test_split(test_idc, test_size=0.5, random_state=11037)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
h_val = torch.from_numpy(h[val_idc,:,:,:] / norm_factor).cuda()

h_hat = h_val   #精准信道估计
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
U,S,V_H = torch.linalg.svd(h_hat)


#用SVD分解做beamforming~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WBB_inital = U.conj().permute(0,1,-1,-2)[:,:,0:DATA_stream,:] #维度[batchsize,Subcarrier,DATA_stream,UE_antenna] #用SVD去做！检验代码写对没有！
power_W_inital = torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(WBB_inital),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]

FBB_inital = V_H.conj().permute(0,1,-1,-2)[:,:,:,0:DATA_stream] #维度[batchsize,Subcarrier,BS_antenna,DATA_stream]
power_F_inital = torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(FBB_inital),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #保存成.mat形式
# scipy.io.savemat('/home/yangjunyi/NAIC/SNR_training/upperbound_result/h_val.mat', mdict={'h_val': h_val.cpu().numpy(),})
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for tap,noise_power_dBm in enumerate(all_noise_power_dBm):
    #计算noise
    if noiseless:
        noise_power_dBm = -np.inf    
    noise_power = (10**((noise_power_dBm-tx_power_dBm-noise_factor)/10))

    RATE = 0
    
    #注水赋形
    A = noise_power/norm_factor/norm_factor #噪声功率
    gamma_matrix = water_filling(h_hat,DATA_stream,A) #注水求得注水算法矩阵，维度为[N,subcarrier,1,DATA_stream]
    FBB = FBB_inital * torch.sqrt(gamma_matrix)
    power_F = torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(FBB),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]

    WBB = WBB_inital

    # 开始校验
    WHF = torch.matmul(WBB,torch.matmul(h_val,FBB))

    signal_SNR =  torch.matmul( WHF , conj_T(WHF) ) / A #维度[batchsize,subcarrier,DATA_stream,DATA_stream]

    RATE =  torch.mean( torch.log2( torch.real( torch.det( torch.eye(DATA_stream,DATA_stream).cuda() + signal_SNR))) ) #维度[batchsize,subcarrier]



    
    print('upper bound rate = {:.8f}'.format(RATE))

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #保存成.mat形式
    WBB = torch.reshape(WBB, (-1,Subcarrier*DATA_stream,UE_antenna)) #维度从[N,Subcarrier，DATA_stream,UE_antenna]变为[N,Subcarrier*DATA_stream,UE_antenna]
    WBB = conj_T(WBB) #维度[N, UE_antenna, Subcarrier*DATA_stream]
    

    FBB = conj_T(FBB) #维度从[N,Subcarrier，BS_antenna, DATA_stream]变为[N, Subcarrier, DATA_stream, BS_antenna]
    FBB = torch.reshape(FBB, (-1,Subcarrier*DATA_stream,BS_antenna)) #维度[N,Subcarrier*DATA_stream,BS_anteena]
    FBB = conj_T(FBB) #维度[N,BS_antenna , Subcarrier*DATA_stream]


    scipy.io.savemat('/home/yangjunyi/BeamformingRevise_V2/upperbound_result/beamforming_{}Noise.mat'.format(noise_power_dBm), mdict={'Wopt': WBB.cpu().numpy(),'Fopt': FBB.cpu().numpy(),})

    #

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    




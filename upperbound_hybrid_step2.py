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
# fname_h_real = '/home/yangjunyi/BeamformingRevise_V3/Dataset/' + dataset_name + '/h_real.mat'
# fname_h_imag = '/home/yangjunyi/BeamformingRevise_V3/Dataset/' + dataset_name + '/h_imag.mat'
# UE_location = '/home/yangjunyi/BeamformingRevise_V3/Dataset/' + dataset_name + '/ue_loc.mat'
fname_h_real = '/home/yangjunyi/BeamformingRevise_V3/Dataset/h_real.mat'
fname_h_imag = '/home/yangjunyi/BeamformingRevise_V3/Dataset/h_imag.mat'
UE_location = '/home/yangjunyi/BeamformingRevise_V3/Dataset/ue_loc.mat'



h_real = np.float32(np.transpose(h5py.File(fname_h_real)['h_real']))
h_imag = np.float32(np.transpose(h5py.File(fname_h_imag)['h_imag']))
UE_location = np.float32(np.transpose(h5py.File(UE_location)['ue_loc']))
tx_power_dBm = 10


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

for tap,noise_power_dBm in enumerate(all_noise_power_dBm):
    #计算noise
    if noiseless:
        noise_power_dBm = -np.inf    
    noise_power = (10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)/torch.tensor([norm_factor])/torch.tensor([norm_factor])).cuda()

    #读取数据
    beamforming_file = '/home/yangjunyi/BeamformingRevise_V3/upperbound_result/hybrid_beamforming_'+str(noise_power_dBm)+'Noise.mat'
    FRF_real = np.float32(np.transpose(h5py.File(beamforming_file)['FRF_real']))
    FRF_imag = np.float32(np.transpose(h5py.File(beamforming_file)['FRF_imag']))
    FBB_real = np.float32(np.transpose(h5py.File(beamforming_file)['FBB_real']))
    FBB_imag = np.float32(np.transpose(h5py.File(beamforming_file)['FBB_imag']))
    WRF_real = np.float32(np.transpose(h5py.File(beamforming_file)['WRF_real']))
    WRF_imag = np.float32(np.transpose(h5py.File(beamforming_file)['WRF_imag']))
    WBB_real = np.float32(np.transpose(h5py.File(beamforming_file)['WBB_real']))
    WBB_imag = np.float32(np.transpose(h5py.File(beamforming_file)['WBB_imag']))

    FRF = torch.from_numpy(FRF_real + 1j * FRF_imag).cuda()
    FBB = torch.from_numpy(FBB_real + 1j * FBB_imag).cuda()
    WRF = torch.from_numpy(WRF_real + 1j * WRF_imag).cuda()
    WBB = torch.from_numpy(WBB_real + 1j * WBB_imag).cuda()

    F_complex = torch.matmul(FRF,FBB) #维度[N, BS_antenna , Subcarrier*DATA_stream]
    F_complex = conj_T(F_complex) #维度[N , Subcarrier*DATA_stream , BS_antenna ]
    F_complex = torch.reshape(F_complex, (-1,Subcarrier,DATA_stream,BS_antenna)) #维度[N , Subcarrier,DATA_stream , BS_antenna ]
    F_complex = conj_T(F_complex) #维度[N , Subcarrier, BS_antenna , DATA_stream ]
    power_F = torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(F_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]

    W_complex = torch.matmul(WRF,WBB) #维度[N, UE_antenna, Subcarrier*DATA_stream]
    W_complex = conj_T(W_complex) #维度[N, Subcarrier*DATA_stream , UE_antenna]
    W_complex = torch.reshape(W_complex, (-1,Subcarrier,DATA_stream,UE_antenna)) #[N,Subcarrier,DATA_stream,UE_antenna]
    power_W = torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(W_complex),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]

    
    # 开始校验
    WHF = torch.matmul(W_complex,torch.matmul(h_val,F_complex))
    A = noise_power #噪声功率

    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # temp = WHF[7,0,:,:]
    # temp1 = W_complex[7,0,:,:]
    # temp2 = F_complex[7,0,:,:]
    # temp3 = FRF[7,:,:]
    # temp4 = FBB[7,:,:]
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    signal_SNR =  torch.matmul( WHF , conj_T(WHF) ) / A #维度[batchsize,subcarrier,DATA_stream,DATA_stream]

    
    W_WH_inverse = torch.linalg.pinv( torch.matmul(W_complex,conj_T(W_complex)) )

    signal_SNR = torch.matmul(W_WH_inverse,signal_SNR)
    
    U,S,V_H = torch.linalg.svd(torch.eye(DATA_stream,DATA_stream).cuda() + signal_SNR)

    prod_S = torch.log2(torch.real( torch.prod(S,dim=-1,keepdim=False) ) ) 

    prod_S = torch.nan_to_num(prod_S, nan=0.0)  #替换掉nan值

    RATE =  torch.mean( prod_S )  #维度[batchsize,subcarrier]
    
    print('upper bound rate = {:.8f}'.format(RATE))

    

    




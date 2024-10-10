#用于验证和孙老师的讨论的方法，即把一段相邻的subcarrier的信道看作是相同的

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

# Subcarrier_gap = 4  #发送pilot的载波间隔，得到的数据量为Subcarrier/Subcarrier_gap*Pilot_num*DATA_stream
subcarrier_group = 16  #每相邻的subcarrier_group个subcarrier看作是相同的
Pilot_subcarrier_index = range(0,Subcarrier,subcarrier_group) #发射pilot的子载波序号

Pilot_num = 16 #PILOT数量
Feedback_feature = int(Pilot_num  * DATA_stream * 2) #为量化的反馈的向量长度


estimate_noise_power_dbm = -94
all_noise_power_dBm = [-94]
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
h_scaled = h / norm_factor  # 归一化，复数
h_concat_scaled = np.concatenate((h_real / norm_factor, h_imag / norm_factor), axis=-1)  # 将实部和虚部拼接起来，维度[N,subcarrier,UE_antenna, 2 * BS_antenna]


# 数据集分割（训练集、验证集、测试集）————————————————————————————————————————————————————————————————————————————————————————————————
train_idc, test_idc = train_test_split(np.arange(h.shape[0]), test_size=0.3, random_state=11037)
val_idc, test_idc = train_test_split(test_idc, test_size=0.01, random_state=11037)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #保存成.mat形式
# h_val = h[val_idc,:,:,:]
# scipy.io.savemat('h_val_32_8.mat', mdict={'h_val': h_val, 'norm_factor': norm_factor,})
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 制作数据集的tensor————————————————————————————————————————————————————————————————————————————————————————————————
h_train = h_concat_scaled[train_idc, :]
h_val = h_concat_scaled[val_idc, :]
h_test = h_concat_scaled[test_idc, :]


h_train = torch.from_numpy(h_train)
h_val = torch.from_numpy(h_val)
h_test = torch.from_numpy(h_test)


# Pytorch train and test sets
h_train = torch.utils.data.TensorDataset(h_train[:,:,:,:BS_antenna], h_train[:,:,:,BS_antenna:]) #实部与虚部分开存放
h_val = torch.utils.data.TensorDataset(h_val[:,:,:,:BS_antenna], h_val[:,:,:,BS_antenna:])
h_test = torch.utils.data.TensorDataset(h_test[:,:,:,:BS_antenna], h_test[:,:,:,BS_antenna:])

# data loader
h_train = torch.utils.data.DataLoader(h_train, batch_size=batch_size, shuffle=False)
h_val = torch.utils.data.DataLoader(h_val, batch_size=batch_size, shuffle=False)
h_test = torch.utils.data.DataLoader(h_test, batch_size=batch_size, shuffle=False)


#计算noise
# if noiseless:
#     noise_power_dBm = -np.inf    
# noise_power = (10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)/torch.tensor([norm_factor])/torch.tensor([norm_factor])).cuda()



#LS过程
for tap,noise_power_dBm in enumerate(all_noise_power_dBm):
    #计算noise
    if noiseless:
        noise_power_dBm = -np.inf    
    noise_power = (10**((noise_power_dBm-tx_power_dBm-noise_factor)/10))

    RATE = 0
    for batch_idx,(h_real,h_imag) in enumerate(h_val):

        h_real_muti = h_real[:,:,:,:].cuda()
        h_imag_muti = h_imag[:,:,:,:].cuda()
        h_complex_muti = h_real_muti + 1j*h_imag_muti
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        h_hat = h_complex_muti[:,Pilot_subcarrier_index,:,:] #完美估计在index上的信道
        h_hat = torch.unsqueeze(h_hat,dim = 2)*torch.ones(1,1,subcarrier_group,1,1).cuda()
        h_hat = torch.reshape(h_hat,(-1,Subcarrier,UE_antenna,BS_antenna))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        U,S,V_H = torch.linalg.svd(h_hat) #维度 U[batchsize,Subcarrier,UE_antenna,UE_antenna], V_H[batchsize,Subcarrier,BS_antenna,BS_antenna], S[batchsize,Subcarrier,min(UE_antenna,BS_antenna)]

        A = noise_power/norm_factor/norm_factor #噪声功率

        #用SVD分解做beamforming~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        WBB = U.conj().permute(0,1,-1,-2)[:,:,0:DATA_stream,:] #维度[batchsize,Subcarrier,DATA_stream,UE_antenna] #用SVD去做！检验代码写对没有！
        power_W =  torch.unsqueeze(torch.unsqueeze(torch.sum(torch.pow(F_norm_complex(WBB),2),dim=-1,keepdim=True),dim=-1),dim=-1) #功率维度是batchsize,1,1,1]
        # temp = torch.matmul(WBB,conj_T(WBB))
        # WBB = WBB  / power_W 

        FBB = V_H.conj().permute(0,1,-1,-2)[:,:,:,0:DATA_stream] #维度[batchsize,Subcarrier,BS_antenna,DATA_stream]
        gamma_matrix = water_filling(h_hat,DATA_stream,A) #注水求得注水算法矩阵，维度为[batch_size,subcarrier_gap,1,DATA_stream]

        # temp = torch.sum(torch.sum(torch.sum(gamma_matrix,dim=-1),dim=-1),dim=-1)

        power_F =  torch.unsqueeze(torch.unsqueeze(  torch.sum(F_norm_complex(FBB),dim=-1,keepdim=True)  ,dim=-1),dim=-1)
        FBB = FBB * torch.sqrt(gamma_matrix)
        # power_F = torch.unsqueeze(torch.unsqueeze(F_norm_complex(FBB),dim=-1),dim=-1)
        power_F =  torch.unsqueeze(torch.unsqueeze(  torch.sum(torch.pow(F_norm_complex(FBB),2),dim=-1,keepdim=True)  ,dim=-1),dim=-1)

        # torch.pow(torch.abs(complex_part),2)
        # temp = torch.sum(torch.pow(torch.abs(FBB),2),dim=-2,keepdim=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 开始校验
        WHF = torch.matmul(WBB,torch.matmul(h_complex_muti,FBB))

        signal_SNR =  torch.matmul( WHF , conj_T(WHF) ) / A #维度[batchsize,subcarrier,DATA_stream,DATA_stream]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # W_WH_inverse = torch.linalg.pinv( torch.matmul(WBB,conj_T(WBB)) )
        # signal_SNR = torch.matmul(W_WH_inverse,signal_SNR)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        RATE +=  torch.mean( torch.log2( torch.real( torch.det( torch.eye(DATA_stream,DATA_stream).cuda() + signal_SNR))) ) #维度[batchsize,subcarrier]
            

    RATE /= batch_idx + 1

    print('upper bound rate = {:.8f}'.format(RATE))
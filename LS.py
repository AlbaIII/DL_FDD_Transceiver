from logging import BASIC_FORMAT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from utils import *


np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BS_antenna = 32
UE_antenna = 8
RF_chain = 2 #RF_chain数量，仅当采用Hybird系统时有用
DATA_stream = RF_chain #数据流数量
Subcarrier = 16  #载波数量
Subcarrier_gap = 4  #发送pilot的载波间隔，得到的数据量为Subcarrier/Subcarrier_gap*Pilot_num*DATA_stream
Pilot_num = int(32 *  Subcarrier / Subcarrier_gap)#载波数量
Feedback_feature = int(Pilot_num * Subcarrier * DATA_stream / Subcarrier_gap * 2) #为量化的反馈的向量长度

all_noise_power_dBm = [-79,-84,-89,-94,-99,-104,-109,-114]
noise_factor = -13 #dB

noiseless = False
nepoch = 101
batch_size = 128



# 选择数据集————————————————————————————————————————————————————————————————————————————————————————————————
dataset_name = 'I3_60_ULA' #O1_28_ULA或I3_60_ULA
fname_h_real = '/home/yangjunyi/NAIC/Dataset/' + dataset_name + '/h_real'+str(BS_antenna)+str(UE_antenna)+'.mat'
fname_h_imag = '/home/yangjunyi/NAIC/Dataset/' + dataset_name + '/h_imag'+str(BS_antenna)+str(UE_antenna)+'.mat'
UE_location = '/home/yangjunyi/NAIC/Dataset/' + dataset_name + '/ue_loc'+str(BS_antenna)+str(UE_antenna)+'.mat'


# fname_h_real = '/home/yangjunyi/NAIC/Dataset/h_real.mat'
# fname_h_imag = '/home/yangjunyi/NAIC/Dataset/h_imag.mat'
# UE_location = '/home/yangjunyi/NAIC/Dataset/ue_loc.mat'

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
train_idc, test_idc = train_test_split(np.arange(h.shape[0]), test_size=0.4, random_state=11037)
val_idc, test_idc = train_test_split(test_idc, test_size=0.5, random_state=11037)


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


#构建UE和BS侧的调制矩阵,所有的子载波都是用相同的导频,其中，A是UE端，B是BS端,由于不需要是用UE端，因此A应该是[UE_antenna.UE_antenna]的对角矩阵
W_real = torch.from_numpy(np.random.randn(DATA_stream,UE_antenna)).float().cuda()   #维度[DATA_stream,BS_antenna]
W_imag = torch.from_numpy(np.random.randn(DATA_stream,UE_antenna)).float().cuda()   #维度[DATA_stream,BS_antenna]
power_W = torch.unsqueeze(torch.unsqueeze(F_norm(W_real,W_imag),dim=-1),dim=-1)
W_real = W_real / power_W * np.sqrt(DATA_stream)
W_imag = W_imag / power_W * np.sqrt(DATA_stream)#维度[DATA_stream,UE_antenna]
W = W_real + 1j*W_imag
power_W = F_norm_complex(W)


F_real = torch.from_numpy(np.random.randn(Pilot_num,BS_antenna,DATA_stream)).float().cuda()
F_imag = torch.from_numpy(np.random.randn(Pilot_num,BS_antenna,DATA_stream)).float().cuda()
power_F = torch.unsqueeze(torch.unsqueeze(F_norm(F_real,F_imag),dim=-1),dim=-1)
F_real = F_real / power_F
F_imag = F_imag / power_F
F_real = torch.sum(F_real,dim=-1,keepdim=False)/ np.sqrt(DATA_stream)   #假设s是个0.7071向量，求和之后维度变成[Pilot_num,BS_antenna]
F_imag = torch.sum(F_imag,dim=-1,keepdim=False)/ np.sqrt(DATA_stream)
F_real = F_real.permute(1,0) #交换维度后变成[BS_antenna,Pliot_num]
F_imag = F_imag.permute(1,0) #交换维度后变成[BS_antenna,Pliot_num]
F = F_real + 1j*F_imag
power_F = F_norm_complex(F)

#导频矩阵
print(F.is_contiguous())
print(F.T.is_contiguous())
print(W.is_contiguous())
X = torch.kron(F.T,W) #维度[Pilot_num*DATA_stream , BS_anteena*UE_antenna]
XH_X = torch.matmul( X.conj().T,X) #维度[BS_anteena*UE_antenna , BS_anteena*UE_antenna]
XH_X_inv = torch.linalg.pinv(  XH_X   )
temp = torch.matmul(  XH_X_inv , XH_X   )

# 噪声设定————————————————————————————————————————————————————————————————————————————————————————————————



for tap,noise_power_dBm in enumerate(all_noise_power_dBm):
    # 噪声设定————————————————————————————————————————————————————————————————————————————————————————————————

    if noiseless:
        noise_power_dBm = -np.inf    
    noise_power = (10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)/torch.tensor([norm_factor])/torch.tensor([norm_factor])).cuda()

    RATE = 0
    for batch_idx,(h_real,h_imag) in enumerate(h_val):
        for i in range(0,int(Subcarrier/Subcarrier_gap)): #第i个发送pilot的子载波
            h_real_muti = h_real[:,i*Subcarrier_gap:(i+1)*Subcarrier_gap,:,:].cuda()
            h_imag_muti = h_imag[:,i*Subcarrier_gap:(i+1)*Subcarrier_gap,:,:].cuda()
            h_complex_muti = h_real_muti + 1j*h_imag_muti

            h_real_single = torch.unsqueeze(h_real[:,i*Subcarrier_gap,:,:].cuda(),dim=1) #信道维度[batchsize, 1 ,BS_antenna,UE_antenna]
            h_imag_single = torch.unsqueeze(h_imag[:,i*Subcarrier_gap,:,:].cuda(),dim=1)
            h_complex_single = h_real_single + 1j*h_imag_single
            
            
            
            #信道和Y的向量化
            h_vector = vector(h_complex_single)  #维度[batchsize,1 , BS_antenna×UE_antenna , 1]
            Y_vector = torch.matmul(X,h_vector) #维度[[batchsize,1 , Pilot_num*UE_antenna , 1]]

            #添加噪声
            noise_real = (torch.normal(0,1, size=Y_vector.size()) * torch.sqrt(torch.tensor([noise_power/2]))).cuda()
            noise_imag = (torch.normal(0,1, size=Y_vector.size()) * torch.sqrt(torch.tensor([noise_power/2]))).cuda()
            Y_vector = Y_vector + noise_real +1j*noise_imag

            #估计信道
            h_vector_hat = torch.matmul(  torch.matmul(XH_X_inv ,X.conj().T),Y_vector  ) #维度[batchsize,1 , BS_antenna×UE_antenna , 1]
            h_hat = anti_vector(h_vector_hat,UE_antenna,BS_antenna) * torch.ones(1,Subcarrier_gap,1,1).cuda() #维度[batchsize, Subcarrier_gap  , UE_antenna,BS_antenna]

            U,S,V_H = torch.linalg.svd(h_hat) #维度 U[batchsize,Subcarrier,UE_antenna,UE_antenna], V_H[batchsize,Subcarrier,BS_antenna,BS_antenna], S[batchsize,Subcarrier,min(UE_antenna,BS_antenna)]

            A = noise_power #噪声功率

            #用SVD分解做beamforming~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            WBB = U.conj().permute(0,1,-1,-2)[:,:,0:DATA_stream,:] #维度[batchsize,Subcarrier,DATA_stream,UE_antenna] #用SVD去做！检验代码写对没有！
            power_W = torch.unsqueeze(torch.unsqueeze(F_norm_complex(WBB),dim=-1),dim=-1)
            temp = torch.matmul(WBB,conj_T(WBB))
            # WBB = WBB  / power_W 

            FBB = V_H.conj().permute(0,1,-1,-2)[:,:,:,0:DATA_stream] #维度[batchsize,Subcarrier_gap,BS_antenna,DATA_stream]
            gamma_matrix = water_filling(h_hat,DATA_stream,A) #注水求得注水算法矩阵，维度为[batch_size,subcarrier_gap,1,DATA_stream]
            FBB = FBB * torch.sqrt(gamma_matrix)
            power_F = torch.unsqueeze(torch.unsqueeze(F_norm_complex(FBB),dim=-1),dim=-1)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # 开始校验
            WHF = torch.matmul(WBB,torch.matmul(h_complex_muti,FBB))

            signal_SNR =  torch.matmul( WHF , conj_T(WHF) ) / A #维度[batchsize,subcarrier,DATA_stream,DATA_stream]

            W_WH_inverse = torch.linalg.pinv( torch.matmul(WBB,conj_T(WBB)) )

            signal_SNR = torch.matmul(W_WH_inverse,signal_SNR)
            RATE +=  torch.mean( torch.log2( torch.real( torch.det( torch.eye(DATA_stream,DATA_stream).cuda() + signal_SNR))) ) / ( Subcarrier/ Subcarrier_gap) #维度[batchsize,subcarrier]



    RATE /= batch_idx + 1

    print('LS rate = {:.8f}'.format(RATE))

    




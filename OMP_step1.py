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


np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BS_antenna = 64
UE_antenna = 4
BS_RF_chain = 4 #RF_chain数量，仅当采用Hybird系统时有用
UE_RF_chain = 2
DATA_stream = UE_RF_chain #数据流数量
Subcarrier = 128  #载波数量 


subcarrier_group = 16  #每相邻的subcarrier_group个subcarrier看作是相同的
Subcarrier_gap = subcarrier_group  #发送pilot的载波间隔，得到的数据量为Subcarrier/Subcarrier_gap*Pilot_num*DATA_stream
Pilot_subcarrier_index = range(0,Subcarrier,subcarrier_group) #发射pilot的子载波序号

Pilot_num = 16 #PILOT数量
Feedback_feature = int(Pilot_num  * DATA_stream * 2) #为量化的反馈的向量长度

estimate_noise_power_dbm = -94
noise_factor = -13 #dB
all_noise_power_dBm = [-79,-84,-89,-94,-99,-104,-109,-114]


noiseless = False
nepoch = 501
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

# 数据预处理（实部虚部结合、归一化）————————————————————————————————————————————————————————————————————————————————————————————————
h = h_real + 1j * h_imag  # h是复数，维度[N,subcarrier,UE_antenna,BS_antenna]
norm_factor = np.max(abs(h))  # 模的最大值，也是之后的归一化系数


# 数据集分割（训练集、验证集、测试集）————————————————————————————————————————————————————————————————————————————————————————————————
train_idc, test_idc = train_test_split(np.arange(h.shape[0]), test_size=0.4, random_state=11037)
val_idc, test_idc = train_test_split(test_idc, test_size=0.5, random_state=11037)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
h_val = torch.from_numpy(h[val_idc,:,:,:] / norm_factor).cuda()


#构建UE和BS侧的调制矩阵,所有的子载波都是用相同的导频,其中，A是UE端，B是BS端,由于不需要是用UE端，因此A应该是[UE_antenna.UE_antenna]的对角矩阵
W_real = torch.from_numpy(np.random.randn(Pilot_num,DATA_stream,UE_antenna)).float().cuda()   #维度[Pilot_num,DATA_stream,UE_antenna]
W_imag = torch.from_numpy(np.random.randn(Pilot_num,DATA_stream,UE_antenna)).float().cuda()   #维度[Pilot_num,DATA_stream,UE_antenna]
power_W = torch.unsqueeze(torch.unsqueeze(F_norm(W_real,W_imag),dim=-1),dim=-1)     #维度[Pilot_num,1,1]
W_real = W_real / power_W * np.sqrt(DATA_stream)
W_imag = W_imag / power_W * np.sqrt(DATA_stream)#维度[Pilot_num,DATA_stream,UE_antenna]
W = W_real + 1j*W_imag
power_W = torch.pow(F_norm_complex(W),2)


F_real = torch.from_numpy(np.random.randn(Pilot_num,BS_antenna,DATA_stream)).float().cuda() #维度[Pilot_num,BS_antenna,DATA_stream]
F_imag = torch.from_numpy(np.random.randn(Pilot_num,BS_antenna,DATA_stream)).float().cuda() #维度[Pilot_num,BS_antenna,DATA_stream]
power_F = torch.unsqueeze(torch.unsqueeze(F_norm(F_real,F_imag),dim=-1),dim=-1)
F_real = F_real / power_F * np.sqrt(DATA_stream)
F_imag = F_imag / power_F * np.sqrt(DATA_stream)

F = F_real+1j*F_imag
power_F = torch.pow(F_norm_complex(F),2) #维度[Pilot_num,1,1]


F_real = torch.sum(F_real,dim=-1,keepdim=True)   #假设s是个 1 向量，求和之后维度变成[Pilot_num,BS_antenna,1]
F_imag = torch.sum(F_imag,dim=-1,keepdim=True)
F = F_real + 1j*F_imag #维度[Pilot_num,BS_antenna,1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


print(W.is_contiguous())
print(F.transpose(-1,-2).is_contiguous())
X = Kron_Twodim(F.transpose(-1,-2),W) #维度[Pilot_num*DATA_stream , BS_antenna*UE_antenna],观测矩阵
temp = torch.linalg.matrix_rank(X)

UE_narrow = 4*UE_antenna
BS_narrow = 4*BS_antenna
AR = torch.from_numpy( ULA_DFT_codebook(nseg=UE_narrow,n_antenna=UE_antenna) ).T.cuda() #维度[UE_antenna,narrow_UE]
AT = torch.from_numpy( ULA_DFT_codebook(nseg=BS_narrow,n_antenna=BS_antenna) ).T.cuda() #维度[BS_antenna,narrow_BS]

WA = torch.matmul(W,AR) #维度[Pilot_num,DATA_stream,narrow_UE]
AF = torch.matmul(conj_T(AT),F) #维度[Pilot_num,narrow_BS,1]

#导频矩阵
print(WA.is_contiguous())
print(AF.transpose(-1,-2).contiguous().is_contiguous())
A = Kron_Twodim(AF.transpose(-1,-2).contiguous(),WA) #维度[Pilot_num*DATA_stream , narrow_BS*narrow_UE],观测矩阵

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# h_complex_single = torch.unsqueeze(torch.unsqueeze(h_val[0,1,:,:],dim=0),dim=0) #信道维度[1, 1 ,UE_antenna,BS_antenna]

# h_vector = vector(h_complex_single)  #维度[1,1 , BS_antenna×UE_antenna , 1]
# Y_vector = torch.matmul(X,h_vector) #维度[[1,1 , Pilot_num*DATA_stream , 1]]

# Lambda_vector_hat = OMP(A,Y_vector,1000) #维度[1,1 , UE_narrow*BS_narrow , 1]
# Lambda_hat = anti_vector(Lambda_vector_hat,UE_narrow,BS_narrow) #维度[batchsize, Subcarrier_gap  , UE_narrow,BS_narrow]

# h_hat = torch.matmul(torch.matmul(AR,Lambda_hat),conj_T(AT))
# Y_hat = torch.matmul(X,vector(h_hat))
# Y_real = torch.matmul(X,h_vector)
# Y_vector_hat = vector(Y_hat)
# U,S,V_H = torch.linalg.svd(h_hat)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for tap,noise_power_dBm in enumerate(all_noise_power_dBm):
    # 噪声设定————————————————————————————————————————————————————————————————————————————————————————————————

    #计算noise
    if noiseless:
        noise_power_dBm = -np.inf
        estimate_noise_power_dbm = -np.inf
    noise_power = (10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)/torch.tensor([norm_factor])/torch.tensor([norm_factor])).cuda()
    estimate_noise_power = (10**((estimate_noise_power_dbm-tx_power_dBm-noise_factor)/10)/torch.tensor([norm_factor])/torch.tensor([norm_factor])).cuda()

    RATE = 0
    for i in range(0,int(Subcarrier/Subcarrier_gap)): #第i个发送pilot的子载波

        h_complex_muti = h_val[:,i*Subcarrier_gap:(i+1)*Subcarrier_gap,:,:]

        h_complex_single = torch.unsqueeze(h_val[:,i*Subcarrier_gap,:,:],dim=1) #信道维度[batchsize, 1 ,BS_antenna,UE_antenna]

        #OMP过程
        
        #信道和Y的向量化
        h_vector = vector(h_complex_single)  #维度[batchsize,1 , BS_antenna×UE_antenna , 1]
        Y_vector = torch.matmul(X,h_vector) #维度[[batchsize,1 , Pilot_num*DATA_stream , 1]]

        #添加噪声
        noise_real = (torch.normal(0,1, size=Y_vector.size()) * torch.sqrt(torch.tensor([estimate_noise_power/2]))).cuda()
        noise_imag = (torch.normal(0,1, size=Y_vector.size()) * torch.sqrt(torch.tensor([estimate_noise_power/2]))).cuda()
        Y_vector = Y_vector + noise_real +1j*noise_imag

        #估计信道
        Lambda_vector_hat = OMP(A,Y_vector,1000) #维度[batchsize,1 , UE_narrow*BS_narrow , 1]
        Lambda_hat = anti_vector(Lambda_vector_hat,UE_narrow,BS_narrow) * torch.ones(1,Subcarrier_gap,1,1).cuda() #维度[batchsize, Subcarrier_gap  , UE_narrow,BS_narrow]
        h_hat = torch.matmul(torch.matmul(AR,Lambda_hat),conj_T(AT)) #维度[batchsize, Subcarrier_gap  , UE_antenna,BS_antenna]

        U,S,V_H = torch.linalg.svd(h_hat) #维度 U[batchsize,Subcarrier,UE_antenna,UE_antenna], V_H[batchsize,Subcarrier,BS_antenna,BS_antenna], S[batchsize,Subcarrier,min(UE_antenna,BS_antenna)]

        #用SVD分解做beamforming~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        WBB = U.conj().permute(0,1,-1,-2)[:,:,0:DATA_stream,:] #维度[batchsize,Subcarrier,DATA_stream,UE_antenna] #用SVD去做！检验代码写对没有！
        power_W = torch.unsqueeze(torch.unsqueeze(F_norm_complex(WBB),dim=-1),dim=-1)
        # WBB = WBB  / power_W 

        FBB = V_H.conj().permute(0,1,-1,-2)[:,:,:,0:DATA_stream] #维度[batchsize,Subcarrier_gap,BS_antenna,DATA_stream]
        gamma_matrix = water_filling(h_hat,DATA_stream,noise_power) #注水求得注水算法矩阵，维度为[batch_size,subcarrier_gap,1,DATA_stream]
        FBB = FBB * torch.sqrt(gamma_matrix)
        power_F = torch.unsqueeze(torch.unsqueeze(F_norm_complex(FBB),dim=-1),dim=-1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # 开始校验
        WHF = torch.matmul(WBB,torch.matmul(h_complex_muti,FBB))

        signal_SNR =  torch.matmul( WHF , conj_T(WHF) ) / noise_power #维度[batchsize,subcarrier,DATA_stream,DATA_stream]

        # W_WH_inverse = torch.linalg.pinv( torch.matmul(WBB,conj_T(WBB)) )
        # signal_SNR = torch.matmul(W_WH_inverse,signal_SNR)

        # RATE_sum = torch.log2( torch.real( torch.det( torch.eye(DATA_stream,DATA_stream).cuda() + signal_SNR)))
        # RATE_sum = torch.nan_to_num(RATE_sum, nan=0.0)  #替换掉nan值
        RATE +=  torch.mean( torch.log2( torch.real( torch.det( torch.eye(DATA_stream,DATA_stream).cuda() + signal_SNR))) ) / ( Subcarrier/ Subcarrier_gap) #维度[batchsize,subcarrier]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #保存成.mat形式
        if i == 0:
            WBB_total = WBB
            FBB_total = FBB
        else:
            WBB_total = torch.cat((WBB_total,WBB),dim=1)
            FBB_total = torch.cat((FBB_total,FBB),dim=1)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print('omp rate = {:.8f}'.format(RATE))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #保存成.mat形式
    WBB_total = torch.reshape(WBB_total, (-1,Subcarrier*DATA_stream,UE_antenna)) #维度从[N,Subcarrier，DATA_stream,UE_antenna]变为[N,Subcarrier*DATA_stream,UE_antenna]
    WBB_total = conj_T(WBB_total) #维度[N, UE_antenna, Subcarrier*DATA_stream]
    

    FBB_total = conj_T(FBB_total) #维度从[N,Subcarrier，BS_antenna, DATA_stream]变为[N, Subcarrier, DATA_stream, BS_antenna]
    FBB_total = torch.reshape(FBB_total, (-1,Subcarrier*DATA_stream,BS_antenna)) #维度[N,Subcarrier*DATA_stream,BS_anteena]
    FBB_total = conj_T(FBB_total) #维度[N,BS_antenna , Subcarrier*DATA_stream]


    scipy.io.savemat('/home/yangjunyi/BeamformingRevise_V2/OMP_result/beamforming_{}Noise.mat'.format(noise_power_dBm), mdict={'Wopt': WBB_total.cpu().numpy(),'Fopt': FBB_total.cpu().numpy(),})



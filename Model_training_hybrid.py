import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Hybrid_Model_define import *  #在此处切换使用Hybird系统或Digital系统
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split


np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



BS_antenna = 32
UE_antenna = 8
RF_chain = 2 #RF_chain数量，仅当采用Hybird系统时有用
DATA_stream = RF_chain #数据流数量
Subcarrier = 16  #载波数量
Subcarrier_gap = 4  #发送pilot的载波间隔，得到的数据量为Subcarrier/Subcarrier_gap*Pilot_num*DATA_stream
Pilot_num = 16 #载波数量
Feedback_feature = int(Pilot_num * Subcarrier * DATA_stream / Subcarrier_gap * 2) #为量化的反馈的向量长度

estimate_noise_power_dbm = -94
noise_factor = -13 #dB

noiseless = False
nepoch = 101
batch_size = 128

#改变噪声强度
# all_noise_power_dBm = [-79,-84,-89,-94,-99,-104,-109,-114]
# all_vq_dim=[8,8,8,8,8,8,8,8]
# all_vq_b= [8,8,8,8,8,8,8,8]
# all_scale = [8,8,8,8,8,8,8,8]

#改变传输bit数量
all_noise_power_dBm = [-94,-94,-94,-94,-94,-94,-94,-94]
all_vq_dim=[256,128 ,  64, 32,16,16,8,8]
all_vq_b=  [ 2,  2  ,  2,  2, 2, 4, 4,8]
all_scale = [1,2,3,4,5,6,7,8]


# 选择数据集————————————————————————————————————————————————————————————————————————————————————————————————
dataset_name = 'I3_60_ULA' #O1_28_ULA或I3_60_ULA
fname_h_real = '/home/yangjunyi/NAIC/Dataset/' + dataset_name + '/h_real'+str(BS_antenna)+str(UE_antenna)+'.mat'
fname_h_imag = '/home/yangjunyi/NAIC/Dataset/' + dataset_name + '/h_imag'+str(BS_antenna)+str(UE_antenna)+'.mat'
UE_location = '/home/yangjunyi/NAIC/Dataset/' + dataset_name + '/ue_loc'+str(BS_antenna)+str(UE_antenna)+'.mat'

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


h_train = torch.from_numpy(h_train).cuda()
h_val = torch.from_numpy(h_val).cuda()
h_test = torch.from_numpy(h_test).cuda()


# Pytorch train and test sets
h_train = torch.utils.data.TensorDataset(h_train[:,:,:,:BS_antenna], h_train[:,:,:,BS_antenna:]) #实部与虚部分开存放
h_val = torch.utils.data.TensorDataset(h_val[:,:,:,:BS_antenna], h_val[:,:,:,BS_antenna:])
h_test = torch.utils.data.TensorDataset(h_test[:,:,:,:BS_antenna], h_test[:,:,:,BS_antenna:])

# data loader
h_train = torch.utils.data.DataLoader(h_train, batch_size=batch_size, shuffle=False)
h_val = torch.utils.data.DataLoader(h_val, batch_size=batch_size, shuffle=False)
h_test = torch.utils.data.DataLoader(h_test, batch_size=batch_size, shuffle=False)



# #计算noise

h_Fnorm_pow = torch.pow(  F_norm_complex(torch.tensor(h_scaled)),2 )
for tap,noise_power_dBm in enumerate(all_noise_power_dBm):
    vq_dim = all_vq_dim[tap]
    vq_b = all_vq_b[tap]
    s_bit = all_scale[tap]

    #计算noise
    if noiseless:
        noise_power_dBm = -np.inf
        estimate_noise_power_dbm = -np.inf
    noise_power = (10**((noise_power_dBm-tx_power_dBm-noise_factor)/10)/torch.tensor([norm_factor])/torch.tensor([norm_factor])).cuda()
    estimate_noise_power = (10**((estimate_noise_power_dbm-tx_power_dBm-noise_factor)/10)/torch.tensor([norm_factor])/torch.tensor([norm_factor])).cuda()

    #模型构建与训练

    #Hybrid系统模型
    model = Beamformer_Network( BS_antenna=BS_antenna, UE_antenna=UE_antenna, RF_chain=RF_chain, 
                                DATA_stream=DATA_stream, Subcarrier=Subcarrier, Subcarrier_gap=Subcarrier_gap, Pilot_num=Pilot_num,
                                Feedback_feature=Feedback_feature, vq_dim=vq_dim,vq_b=vq_b,s_bit = s_bit,
                                noise_power=noise_power,estimate_noise_power=estimate_noise_power, norm_factor=norm_factor ).cuda()


    # model.load_state_dict(torch.load("/home/yangjunyi/NAIC/Saved Models/hybrid_64bit.pt")) #复制一个已经训练好的模型！！

    learnable_codebook_opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),amsgrad=False)

    print('{}Dataset_{}Pilot_{}Subcarriergap_{}RF_{}Datastream.pt'.format(dataset_name, Pilot_num,Subcarrier_gap,RF_chain,DATA_stream))
    train_loss_hist, val_loss_hist = Model_fit(model=model, h_train=h_train, h_val=h_val, opt=learnable_codebook_opt, EPOCHS=nepoch)


    learnable_model_savefname = '/home/yangjunyi/NAIC/Saved Models/MLP_{}Dataset_{}Pilot_{}bit_{}dBm_noise.pt'.format(dataset_name, Pilot_num,int(Feedback_feature/vq_dim*vq_b),noise_power_dBm)
    torch.save(model.state_dict(),learnable_model_savefname)






# plt.figure()
# plt.plot(train_loss_hist,label='training loss')
# plt.plot(val_loss_hist,label='validation loss')
# plt.legend()
# plt.show()
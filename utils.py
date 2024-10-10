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

def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def matrix_product(A_real,A_imag,B_real,B_imag): #A*B复数矩阵乘法， A维度 N1×N， B维度 N×N2， 最终结果 R,I, 维度都是[N1,N2]

    cat_kernels_A_real = torch.cat((A_real,-A_imag),dim=-1)
    cat_kernels_A_imag = torch.cat((A_imag,A_real),dim=-1)
    # cat_kernels_A_complex = torch.cat((cat_kernels_A_real, cat_kernels_A_imag),dim=-2)

    cat_kernels_B_complex = torch.cat((B_real,B_imag),dim=-2)

    output_real = torch.matmul(cat_kernels_A_real,cat_kernels_B_complex)
    output_imag = torch.matmul(cat_kernels_A_imag,cat_kernels_B_complex)

    return output_real,output_imag

def F_norm(real_part,imag_part): #计算矩阵的F范数，最后的结果是开根号的

    sq_real = torch.pow(real_part,2)
    sq_imag = torch.pow(imag_part,2)
    abs_values = sq_real + sq_imag

    return torch.pow( torch.sum( torch.sum(abs_values,dim=-1,keepdim=False) , dim=-1 , keepdim=False) , 0.5 )

def F_norm_complex(complex_part): #计算矩阵的F范数，最后的结果是开根号的

    abs_square = torch.pow(torch.abs(complex_part),2)

    return torch.pow( torch.sum( torch.sum(abs_square,dim=-1,keepdim=False) , dim=-1 , keepdim=False) , 0.5 )


def DFT_angles(n_beam):
    delta_theta = 1/n_beam
    if n_beam % 2 == 1:
        thetas = np.arange(0,1/2,delta_theta)
        # thetas = np.linspace(0,1/2,n_beam//2+1,endpoint=False)
        thetas = np.concatenate((-np.flip(thetas[1:]),thetas))
    else:
        thetas = np.arange(delta_theta/2,1/2,delta_theta) 
        thetas = np.concatenate((-np.flip(thetas),thetas))
    return thetas

def ULA_DFT_codebook(nseg,n_antenna,spacing=0.5):
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex64)
    thetas = DFT_angles(nseg)
    # temp = 1/spacing*thetas
    azimuths = np.arcsin(1/spacing*thetas)
    temp = np.sin(azimuths)
    for i,theta in enumerate(azimuths):
        arr_response_vec = [-1j*2*np.pi*k*spacing*np.sin(theta) for k in range(n_antenna)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

def ComputePower(real_part,imag_part):
    sq_real = torch.pow(real_part,2)
    sq_imag = torch.pow(imag_part,2)
    abs_values = sq_real + sq_imag
    return abs_values

def Modulation(bits): #bits的维度是[batchsize,subcarrier,DATA_stream,2],这里的2是4QAM
    
    return 0.7071 * (2 * bits[:,:,:, 0] - 1) , 0.7071 * (2 * bits[:,:,:, 1] - 1)  # This is just for 4QAM modulation

def deModulation(symbol_real,symbol_imag): #symbol的维度是[batchsize,subcarrier,DATA_stream],这里是4QAM,采用硬判决，直接查看正负
    bit_real = torch.unsqueeze(symbol_real>0,dim=-1)
    bit_imag = torch.unsqueeze(symbol_imag>0,dim=-1)

    return torch.cat((bit_real,bit_imag),dim=-1)# This is just for 4QAM modulation

def vector(H):  #使信道矩阵向量化，这里矩阵的维度是[batchsize,subcarriers,UE_antenna,BS_antenna]
    temp = H.permute(0,1,-1,-2)
    
    output = torch.reshape(temp,(-1,temp.shape[1],temp.shape[-1]*temp.shape[-2],1))
    

    return output

def anti_vector(H_vector,UE_antenna,BS_antenna):
    temp = torch.reshape(H_vector,(H_vector.shape[0],H_vector.shape[1],BS_antenna,UE_antenna))
    output = temp.permute(0,1,-1,-2)
   
    return output

def conj_T(X): #返回最后两个维度的共轭转置
    return X.conj().transpose(-1,-2)

def Kron_Twodim(A,B): #这里A和B的的第一维必须相同，[N,pilot_num，a,b]，且N>1
    batch_size = A.shape[0]
    for i in range(batch_size):
        A_part = A[i,:,:] #维度[a,b]
        B_part = B[i,:,:] #维度[A,B]
        kron_part = torch.kron(A_part,B_part) #维度[Aa,Bb]
        if i == 0:
            output = kron_part
        else:
            output = torch.cat((output,kron_part),dim=-2)
    return output


def water_filling_single(H,DATA_stream,A):

    batch_size,carrier_size = H.shape[0],H.shape[1]
    gamma_matrix = torch.zeros(batch_size,carrier_size,1,DATA_stream).cuda()

    H_HH = torch.matmul(H,conj_T(H))
    U,S,V_H = torch.linalg.svd(H_HH)

    S_part = S[:,:,0:DATA_stream] #维度[total_num,Subcarrier,DATA_stream]，
    alpha = A / S_part

    for batch_index in range(batch_size):
        for carrier_index in range(carrier_size):
            flag = 1 #指示变量，表示是否找到合适的v，1表示还没找到，0表示已经找到了
            i = 1 #从第i个台阶开始注水
            while flag == 1:
                v = i / (  1 + torch.sum(alpha[batch_index,carrier_index, 0:i ],dim=-1)  )  #维度是1
                gamma = torch.max( 1/v - alpha[batch_index,carrier_index, : ], torch.zeros( DATA_stream  ).cuda()) #维度为[total_num,Subcarrier_gap,DATA_stream]

                if i < DATA_stream:
                    comp = alpha[batch_index,carrier_index, i]
                else:
                    flag = 0 
                if 1/v < comp:
                    flag = 0

                i = i + 1

            gamma_matrix[batch_index,carrier_index,0,:] = gamma
    return gamma_matrix



def water_filling(H,DATA_stream,A):

    batch_size,carrier_size = H.shape[0],H.shape[1]
    gamma_matrix = torch.zeros(batch_size,carrier_size*DATA_stream).cuda()
    sum_gamma = torch.zeros(batch_size).cuda()
    # gamma_matrix = torch.zeros(batch_size,carrier_size,1,DATA_stream).cuda()

    H_HH = torch.matmul(H,conj_T(H))
    U,S,V_H = torch.linalg.svd(H_HH)

    S_part = S[:,:,0:DATA_stream] #维度[total_num,Subcarrier,DATA_stream]
    S_part = torch.reshape(S_part,(batch_size,-1)) #维度[total_num,Subcarrier*DATA_stream]，大小大小大小

    S_sort,S_indices = torch.sort(S_part,dim=-1,descending=True) #维度[total_num,Subcarrier*DATA_stream]
    
    alpha = A / S_sort #维度[total_num,Subcarrier*DATA_stream]

    for batch_index in range(batch_size):
        flag = 1 #指示变量，表示是否找到合适的v，1表示还没找到，0表示已经找到了
        i = 1#从第i个台阶开始注水
        while flag == 1:
            v = i / (  carrier_size + torch.sum(alpha[batch_index, 0:i ],dim=-1)  )  #维度是1
            gamma = torch.max( 1/v - alpha[batch_index, : ], torch.zeros( DATA_stream*carrier_size  ).cuda()) #维度为[Subcarrier_gap*DATA_stream]

            if i < DATA_stream*carrier_size:
                comp = alpha[batch_index,i]
            else:
                flag = 0

            if 1/v < comp:
                flag = 0

            
            i = i+1
            
        sum_gamma[batch_index] = torch.sum(gamma)
        gamma_matrix[batch_index,S_indices[batch_index,:]] = gamma #维度为[tbatch_size,carrier_size*DATA_stream]

    temp = torch.sum(gamma_matrix,dim=-1)
    gamma_matrix = torch.reshape(gamma_matrix,(batch_size,carrier_size,1,DATA_stream))
    temp = torch.sum(torch.sum(torch.sum(gamma_matrix,dim=-1),dim=-1),dim=-1)

    return gamma_matrix

def OMP_single(A,b,sparsity):
    #A维度[Pilot_num*DATA_stream , narrow_BS*narrow_UE]
    #b维度[Pilot_num*DATA_stream , 1]
    Am = A.shape[-2]  #Pilot_num*DATA_stream
    An = A.shape[-1]  #BS_narrow*UE_narrow
    r = b #记录残差#维度[Pilot_num*DATA_stream , 1]
    cor = torch.matmul( conj_T(A) , r) #维度[BS_anteena*UE_antenna , 1]

    k = 0
    index = -torch.ones(sparsity,dtype = torch.int64).cuda() #维度[sparsity]
    while k < sparsity:
        
        ind = torch.argmax(abs(cor), dim=-2) #维度[1]
        if ind in index:
            break
        index[k] = ind
        k = k + 1
        
        temp = index[0:k]
        AS = A[:,index[0:k]] #维度[Pilot_num*DATA_stream,k]

        ASH_AS_inv = torch.linalg.pinv(torch.matmul(conj_T(AS) , AS)) #维度[k , k  ]
        P = torch.matmul( torch.matmul(AS,ASH_AS_inv) , conj_T(AS) ) #维度[Pilot_num*DATA_stream , Pilot_num*DATA_stream  ]
        r = torch.matmul( (torch.eye(Am,Am).cuda() - P) , b ) #记录残差#维度[Pilot_num*DATA_stream , 1] 

        cor = torch.matmul( conj_T(A) , r) #维度[BS_anteena*UE_antenna , 1] 
        
    
    xS = torch.matmul( torch.matmul( ASH_AS_inv,conj_T(AS) ) , b) #维度[sparsity , 1 ] 
    # temp1 = torch.matmul(AS,xS)
    x = torch.zeros(An,1,dtype = torch.complex64).cuda() #维度[BS_narrow*UE_narrow , 1]
    x[index[0:k],:] = xS
    # temp2 = torch.matmul(A,x)
    return x


def OMP(A,b,sparsity):
    Am = A.shape[-2]  #Pilot_num*DATA_stream
    An = A.shape[-1]  #narrow_BS*narrow_UE
    batch_size = b.shape[0]
    subcarrier_size = b.shape[1]
    

    x = torch.zeros(batch_size,subcarrier_size,An,1,dtype = torch.complex64).cuda() #维度[batchsize,1 , narrow_BS*narrow_UE , 1]
    for batch_index in range(batch_size):
        for subcarrier_index in range(subcarrier_size):
            x_single = OMP_single(A,b[batch_index,subcarrier_index,:,:],sparsity)
            x[batch_index,subcarrier_index,:,:] = x_single
    return x




# def OMP(A,b,sparsity):
#     batch_size = b.shape[0]
#     subcarrier_size = b.shape[1]
#     #记录A的长与宽,#维度[Pilot_num*DATA_stream , BS_anteena*UE_antenna]
#     Am = A.shape[-2]  #Pilot_num*DATA_stream
#     An = A.shape[-1]  #BS_anteena*UE_antenna
#     A = torch.unsqueeze(torch.unsqueeze(A,dim=0),dim=0).repeat(b.shape[0],1,1,1) #维度[batchsize , 1 ,  Pilot_num*DATA_stream , BS_anteena*UE_antenna]

#     r = b #记录残差#维度[batchsize,1 , Pilot_num*DATA_stream , 1]
#     cor = torch.matmul( conj_T(A) , r) #维度[batchsize,1 , BS_anteena*UE_antenna , 1]

#     k = 0
#     index = torch.zeros(batch_size,subcarrier_size,sparsity,dtype = torch.int64) #维度[batchsize,1,sparsity]
#     while k < sparsity:
#         ind = torch.argmax(abs(cor), dim=-2) #维度[batchsize,1 , 1]
#         ind = torch.squeeze(ind) #维度[batchsize]
#         index[:,:,k-1] = torch.unsqueeze(ind,dim=-1) #没有办法，subcarrier是1，这里只能用unsqueeze找补一下

#         AS = A[:,:,:,index[:,:,0:k]] #维度[batchsize ,1, Pilot_num*DATA_stream , batchsize, 1 , k]
#         AS = torch.diagonal(AS,dim1=0,dim2=3) #维度[1,Pilot_num*DATA_stream , 1 , k , batchsize]
#         AS = torch.diagonal(AS,dim1=0,dim2=2) #维度[Pilot_num*DATA_stream , k , batchsize , 1]
#         AS = AS.permute(2,3,0,1) #维度[batchsize, 1 , Pilot_num*DATA_stream , k  ]

#         ASH_AS_inv = torch.linalg.pinv(torch.matmul(conj_T(AS) , AS)) #维度[batchsize, 1 , k , k  ]
#         P = torch.matmul( torch.matmul(AS,ASH_AS_inv) , conj_T(AS) ) #维度[batchsize ,1 , Pilot_num*DATA_stream , Pilot_num*DATA_stream  ]
#         r = torch.matmul( (torch.eye(Am,Am).cuda() - P) , b ) #记录残差#维度[batchsize,1 , Pilot_num*DATA_stream , 1] 

#         cor = torch.matmul( conj_T(A) , r) #维度[batchsize,1 , BS_anteena*UE_antenna , 1] 

#         k = k + 1

#     xS = torch.matmul( torch.matmul( ASH_AS_inv,conj_T(AS) ) , b) #维度[batchsize ，1 , sparsity , 1 ] 

#     x = torch.zeros(batch_size,subcarrier_size,An,1,dtype = torch.complex64).cuda() #维度[batchsize,1 , BS_anteena*UE_antenna , 1]

#     for batch_index in range(batch_size):
#         for subcarrier_index in range(subcarrier_size):
#             temp1 = x[batch_index,subcarrier_index,index[batch_index,subcarrier_index,:],:]
#             temp2 = xS[batch_index,subcarrier_index,:,:]
#             x[batch_index,subcarrier_index,index[batch_index,subcarrier_index,:],:] = xS[batch_index,subcarrier_index,:,:]
    
#     return x

    

    

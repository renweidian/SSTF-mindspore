# -*- coding:utf-8 -*-
#from torch.autograd import Variable
from tqdm import tqdm
from mindspore import nn, Model, Tensor, context,save_checkpoint
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.ops import functional as F
from mindspore.common.initializer import Normal
import mindspore.ops as ops
import mindspore.nn as nn
import hdf5storage
import math
from scipy.io import loadmat
from scipy import signal
import h5py
from mindspore.common.initializer import Normal
import numpy as np
from mindspore.ops import operations as P
from mindspore import ops
from mindspore import Tensor
import mindspore
import math
# from model3 import *
import time
import os
from numpy import *

from SSTFmodel import *
import mindspore.dataset as ds


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def Gaussian_downsample(x,psf,s):
    y=np.zeros((x.shape[0],int(x.shape[1]/s),int(x.shape[2]/s)))
    if x.ndim==2:
        x=np.expand_dims(x,axis=0)
    for i in range(x.shape[0]):
        x1=x[i,:,:]
        x2=signal.convolve2d(x1,psf, boundary='symm',mode='same')
        y[i,:,:]=x2[0::s,0::s]
    return y


def sam(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
#    if not img1.shape == img2.shape:
#        raise ValueError('Input images must have the same dimensions.')
#    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
#    img1_ = img1.astype(np.float64)
#    img2_ = img2.astype(np.float64)
    img1_ = np.float64(np.uint8(np.round(img1*255)))
    img2_ = np.float64(np.uint8(np.round(img2*255)))
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))


def rmse1(Fuse1, HRHSI):
    ap = []
    ae = []
    for j in range(Fuse1.shape[0]):
        be = np.mean(np.square(
            np.float64(np.uint8(np.round(Fuse1[j, ...] * 255))) - np.float64(np.uint8(np.round(HRHSI[j, ...] * 255)))))
        bp = 10 * np.log10((255 ** 2) / be)
        ap.append(bp)
        ae.append(be)

    temp_rmse = np.sqrt(np.mean(np.array(ae)))
    temp_psnr = np.mean(np.array(ap))
    return temp_rmse, temp_psnr

def reconstruction(net2, R, R_inv, MSI, training_size, stride):
    index_matrix = P.Zeros()(R.shape[1], MSI.shape[2], MSI.shape[3], mstype.float32)
    abundance_t = P.Zeros()(R.shape[1], MSI.shape[2], MSI.shape[3], mstype.float32)
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j+training_size, k:k+training_size]
            HSI = net2(Tensor(R), Tensor(R_inv), Tensor(temp_hrms))
            HSI = HSI.squeeze()
            HSI = P.clip_by_value(HSI, 0, 1)
            abundance_t[:, j:j+training_size, k:k+training_size] = abundance_t[:, j:j+training_size, k:k+training_size] + HSI
            index_matrix[:, j:j+training_size, k:k+training_size] = 1 + index_matrix[:, j:j+training_size, k:k+training_size]

    HSI_recon = abundance_t / index_matrix
    return HSI_recon

class HSIDataset:
    """自定义数据集类"""
    def __init__(self, path, R, training_size, stride, num,downsample_factor):
            imglist = os.listdir(path)
            train_hrhs = []
            train_hrms = []
            train_lrhs = []
            for i in range(num):
                img = loadmat(path + imglist[i])
                img1 = img["b"]
                # img1 = img1/img1.max()
                HRHSI = np.transpose(img1, (2, 0, 1))
                HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
                MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
                for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                    for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                        temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                        temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                        temp_lrhs = HSI_LR[:,int(j/downsample_factor):int((j+training_size)/downsample_factor), int(k/downsample_factor):int((k+training_size)/downsample_factor)]
                        train_hrhs.append(temp_hrhs)
                        train_hrms.append(temp_hrms)
                        train_lrhs.append(temp_lrhs)
            self.train_hrhs_all = train_hrhs
            self.train_hrms_all = train_hrms
            self.train_lrhs_all = train_lrhs




    def __getitem__(self, index):
            train_hrhs = self.train_hrhs_all[index]
            train_hrms = self.train_hrms_all[index]
            train_lshs = self.train_lrhs_all[index]
            return train_hrms, train_hrhs,train_lshs

    def __len__(self):
            return len(self.train_hrms_all)

def create_F():
    F = np.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div;
    return F

def fspecial(func_name,kernel_size,sigma):
    if func_name=='gaussian':
        m=n=(kernel_size-1.)/2.
        y,x=ogrid[-m:m+1,-n:n+1]
        h=exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h

R = create_F().astype(np.float32)
stride = 32
downsample_factor=16
PSF = fspecial('gaussian', 16, 3)
training_size = 64
LR = 1e-4
batch_size = 1
EPOCH = 100
BATCH_SIZE = 32
batch=1
init_lr2 = 2e-4
init_lr1 = init_lr2 / 10
rmse_optimal = 4
psnr_optimal = 40.4
decay_power = 1.5
num = 20
maxiteration = math.ceil(((512 - training_size) // stride + 1) ** 2 * num / BATCH_SIZE) * EPOCH
#    maxiteration=math.ceil(((1040-training_size)//stride+1)*((1392-training_size)//stride+1)*num/BATCH_SIZE)*EPOCH
warm_iter = math.floor(maxiteration / 40)
print(maxiteration)

path1 = '/home/tanlishan/model_final/data/cave_train/'
path2 = '/home/tanlishan/model_final/data/caveall/'
imglist1 = os.listdir(path1)
imglist2 = os.listdir(path2)
train_data = HSIDataset(path1, R, training_size, stride,downsample_factor=16, num=20)
trainset = ds.GeneratorDataset(train_data, ["data", "label"])
trainset = trainset.batch(32, True)
R_inv = np.linalg.pinv(R)
R_inv = Tensor(R_inv)
R2 = Tensor(R)
cnn =  SSTF_Unet()
optimizer = mindspore.nn.Adam(cnn.trainable_params(), learning_rate=LR,  beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0)
scheduler =nn.cosine_decay_lr(min_lr=1e-6, max_lr=1e-4, total_step=maxiteration, step_per_epoch=2, decay_epoch=2)
# optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.L1Loss(reduction='mean')
step=0

def train_loop(cnn, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data1, label,data2):
        output = cnn(data1,data2)
        loss = loss_fn(output, label)
        return loss, output

    # Get gradient function
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data1, label,data2):
        (loss, _), grads = grad_fn(data1, label,data2)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    cnn.set_train()
    for batch, (data1, label,data2) in enumerate(dataset.create_tuple_iterator()):
        data1=mindspore.Tensor.float(data1)
        data2 = mindspore.Tensor.float(data2)
        label = mindspore.Tensor.float(label)
        loss = train_step(data1, label,data2)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(cnn, trainset, loss_func, optimizer)  
    if t%2==0:       
        save_checkpoint(cnn, 'check/' + 'cave' + str(t) + 'last.ckpt')



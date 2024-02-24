from swin import BasicLayer
from mst import MSAB
import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import nn
from mindspore import numpy
from mindspore import ops
import os
import mindspore.common.dtype as mstype
import collections.abc
from itertools import repeat
from mindspore.ops import operations as P

class MST(nn.Cell):
    def __init__(self,):
        super(MST, self).__init__()
        self.kj1=nn.SequentialCell([nn.Conv2d(62,64,kernel_size=3,stride=1,padding=1,pad_mode='pad'),
                                   BasicLayer(64, (32, 32), 2, 2, 8),
                                   nn.Conv2d(64, 62, kernel_size=3, stride=1, padding=1,pad_mode='pad')])
        self.kj2 = nn.SequentialCell([nn.Conv2d(124,128,kernel_size=3,stride=1,padding=1,pad_mode='pad'),
                                   BasicLayer(128, (16, 16), 2, 4, 8),
                                   nn.Conv2d(128, 124, kernel_size=3, stride=1, padding=1,pad_mode='pad')])
        self.kj3 =nn.SequentialCell([nn.Conv2d(248,256,kernel_size=3,stride=1,padding=1,pad_mode='pad'),
                                   BasicLayer(256, (8, 8), 2, 8, 8),
                                   nn.Conv2d(256, 248, kernel_size=3, stride=1, padding=1,pad_mode='pad')])
        self.kj5 = nn.SequentialCell([nn.Conv2d(62,64,kernel_size=3,stride=1,padding=1,pad_mode='pad'),
                                   BasicLayer(64, (32, 32), 2, 2, 8),
                                   nn.Conv2d(64, 62, kernel_size=3, stride=1, padding=1,pad_mode='pad')])
        self.kj4 = nn.SequentialCell([nn.Conv2d(124,128,kernel_size=3,stride=1,padding=1,pad_mode='pad'),
                                   BasicLayer(128, (16, 16), 2, 4, 8),
                                   nn.Conv2d(128, 124, kernel_size=3, stride=1, padding=1,pad_mode='pad')])
        self.down1=nn.Conv2d(31, 62, kernel_size=4, stride=2, padding=1,pad_mode='pad')
        self.down11 = nn.Conv2d(31, 62, kernel_size=4, stride=2, padding=1,pad_mode='pad')
        self.down2 = nn.Conv2d(62, 124, kernel_size=4, stride=2, padding=1,pad_mode='pad')
        self.down21 = nn.Conv2d(62, 124, kernel_size=4, stride=2, padding=1,pad_mode='pad')
        self.down3 = nn.Conv2d(124, 248, kernel_size=4, stride=2, padding=1,pad_mode='pad')
        self.down31 = nn.Conv2d(124, 248, kernel_size=4, stride=2, padding=1,pad_mode='pad')
        self.gp1=MSAB(62,62,1,1)
        self.gp2 = MSAB(124, 124, 1, 1)
        self.gp3 = MSAB(248, 248, 1, 1)
        self.gp5 = MSAB(62, 62, 1, 1)
        self.gp4 = MSAB(124, 124, 1, 1)
        self.up1 = nn.Conv2dTranspose(248, 124, stride=2, kernel_size=2, pad_mode='pad', padding=0)
        self.up2 = nn.Conv2dTranspose(124, 62, stride=2, kernel_size=2, pad_mode='pad', padding=0)
        self.up3 = nn.Conv2dTranspose(62, 31, stride=2, kernel_size=2, pad_mode='pad', padding=0)
        self.c1= nn.Conv2d(124, 62, 1, 1, pad_mode='pad', padding=0)
        self.c2 = nn.Conv2d(248, 124, 1, 1, pad_mode='pad', padding=0)
        self.c3 = nn.Conv2d(496, 248, 1, 1, pad_mode='pad', padding=0)


        self.c4 = nn.Conv2d(124, 62, 1, 1, pad_mode='pad', padding=0)
        self.c5 = nn.Conv2d(248, 124, 1, 1, pad_mode='pad', padding=0)
        self.c6 = nn.Conv2d(124, 62, 1, 1, pad_mode='pad', padding=0)
        self.c7 = nn.Conv2d(248, 124, 1, 1, pad_mode='pad', padding=0)






    def construct(self, x, y): # x:MSI y:HSI
        x1=self.down1(x)
        x1=self.kj1(x1)
        y1=self.down11(y)
        y1=self.gp1(y1)
        o1=ops.Concat(1)((y1, x1))
        y1=self.c1(o1)

        x2 = self.down2(x1)
        x2 = self.kj2(x2)
        y2 = self.down21(y1)
        y2 = self.gp2(y2)
        o2 = ops.Concat(1)((y2, x2))
        y2 = self.c2(o2)

        x3 = self.down3(x2)
        x3 = self.kj3(x3)
        y3 = self.down31(y2)
        y3 = self.gp3(y3)
        o3 = ops.Concat(1)((y3, x3))
        y3 = self.c3(o3)

        y3=self.up1(y3)
        y3 = ops.Concat(1)((y3, y2))
        y3 = self.c5(y3)
        y31=self.gp4(y3)
        x31=self.kj4(y3)
        o4 = ops.Concat(1)((y31, x31))
        y4 = self.c7(o4)

        y4=self.up2(y3)
        y4 = ops.Concat(1)((y4, y1))
        y4 = self.c4(y4)
        y41=self.gp5(y4)
        x41=self.kj5(y4)
        o5 = ops.Concat(1)((y41, x41))
        y4 = self.c6(o5)
        out=self.up3(y4)

        return out

class SSTF_Unet(nn.Cell):
    def __init__(self, in_channels_MSI=3, out_channels=31, n_feat=31):
        super(SSTF_Unet, self).__init__()
        self.up_sample = nn.ResizeBilinear()
        self.conv_in_MSI = nn.Conv2d(in_channels_MSI, n_feat, kernel_size=3, pad_mode='pad',padding=(3 - 1) // 2)

        self.body = MST()
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, pad_mode='pad',padding=(3 - 1) // 2)

    def construct(self, x, y): # x:MSI y:HSI
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        y = self.up_sample(y,(64,64))

        x = self.conv_in_MSI(x)   # 扩展维度

        h = self.body(x, y)

        h = self.conv_out(h)
        h = y + h
        return h


a = np.ones((2, 3,64,64))
a=a.astype(np.float32)
a = Tensor(a)
b = np.ones((2, 31,8,8))
b=b.astype(np.float32)
b = Tensor(b)
model = SSTF_Unet()
d=model(a,b)
print(d.shape)

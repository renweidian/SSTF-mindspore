import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import nn
from mindspore import numpy
from mindspore import ops
import os
from einops import rearrange
import mindspore
import mindspore.common.dtype as mstype
import collections.abc
from itertools import repeat
from mindspore.ops import operations as P

class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        #self.norm = nn.LayerNorm(dim)

    def construct(self, x):
        shape1 = x.shape[1:]
        m = nn.LayerNorm(shape1, begin_norm_axis=1, begin_params_axis=1)
        x = m(x)
        return self.fn(x)


class GELU(nn.Cell):
    def __init__(self, ):
        super().__init__()
        self.act=nn.GELU()
    def construct(self, x):
        x=self.act(x)
        return x

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

def shift_back(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MS_MSA(nn.Cell):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Dense(dim, dim_head * heads)
        self.to_k = nn.Dense(dim, dim_head * heads)
        self.to_v = nn.Dense(dim, dim_head * heads)
        ones = ops.Ones()
        self.rescale = mindspore.Parameter(ones((self.num_heads, 1, 1),mindspore.float32))
        self.proj = nn.Dense(dim_head * heads, dim)
        self.pos_emb = nn.SequentialCell(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1,pad_mode='pad', padding=1, group=dim),
            GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1,pad_mode='pad', padding=1, group=dim),
        )
        self.dim = dim

    def construct(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        b1,c1,h1=q_inp.shape

        q = q_inp.reshape(b1,int(h1/self.dim),c1,self.dim)
        k = k_inp.reshape(b1, int(h1 / self.dim), c1, self.dim)
        v = v_inp.reshape(b1, int(h1 / self.dim), c1, self.dim)
        v = v
        input_perm = (0,1,3 ,2)
        transpose = ops.Transpose()
        q = transpose(q, input_perm)
        k = transpose(k, input_perm)
        v = transpose(v, input_perm)
        l2_normalize = ops.L2Normalize(axis=-1)
        q = l2_normalize(q)
        k = l2_normalize(k)
        q=transpose(q, input_perm)
        attn = (k @ q)   # A = K^T*Q
        attn = attn * self.rescale
        softmax = ops.Softmax(axis=-1)
        attn=softmax(attn)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Cell):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Conv2d(dim, dim * mult, stride=1, kernel_size=1, has_bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False, group=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, kernel_size=1, stride=1, has_bias=False),
        )

    def construct(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Cell):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.CellList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.CellList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def construct(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

# a = np.ones((1, 62,32,32))
# a=a.astype(np.float32)
# a = Tensor(a)
# cnn=MSAB(62,31,2,1)
# d=cnn(a)



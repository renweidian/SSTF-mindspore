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


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

class Identity(nn.Cell):
    """Identity"""
    def construct(self, x):
        return x

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath1D(DropPath):
    def __init__(self, drop_prob):
        super(DropPath1D, self).__init__(drop_prob=drop_prob, ndim=1)

to_2tuple = _ntuple(2)

act_layers = {
    "GELU": nn.GELU,
    "gelu": nn.GELU,
}

class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = np.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows
class WindowPartitionConstruct(nn.Cell):
    """WindowPartitionConstruct Cell"""

    def __init__(self, window_size):
        super(WindowPartitionConstruct, self).__init__()

        self.window_size = window_size

    def construct(self, x):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = P.Reshape()(x, (B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C))
        x = P.Transpose()(x, (0, 1, 3, 2, 4, 5))
        x = P.Reshape()(x, (B * H * W // (self.window_size ** 2), self.window_size, self.window_size, C))

        return x
class WindowReverseConstruct(nn.Cell):
    """WindowReverseConstruct Cell"""

    def construct(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = windows.shape[0] // (H * W // window_size // window_size)
        x = ops.Reshape()(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
        x = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))
        x = ops.Reshape()(x, (B, H, W, -1))
        return x
class RelativeBias(nn.Cell):
    """RelativeBias Cell"""

    def __init__(self, window_size, num_heads):
        super(RelativeBias, self).__init__()
        self.window_size = window_size
        # define a parameter table of relative position bias
        coords_h = np.arange(self.window_size[0]).reshape(self.window_size[0], 1).repeat(self.window_size[0],
                                                                                         1).reshape(1, -1)
        coords_w = np.arange(self.window_size[1]).reshape(1, self.window_size[1]).repeat(self.window_size[1],
                                                                                         0).reshape(1, -1)
        coords_flatten = np.concatenate([coords_h, coords_w], axis=0)  # 2, Wh, Ww
        relative_coords = coords_flatten[:, :, np.newaxis] - coords_flatten[:, np.newaxis, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = Tensor(relative_coords.sum(-1).reshape(-1))  # Wh*Ww, Wh*Ww
        self.relative_position_bias_table = Parameter(
            Tensor(np.random.randn((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
                   dtype=mstype.float32))  # 2*Wh-1 * 2*Ww-1, nH
        self.one_hot = nn.OneHot(axis=-1, depth=(2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                                 dtype=mstype.float32)
        self.index = Parameter(self.one_hot(self.relative_position_index), requires_grad=False)

    def construct(self, axis=0):
        out = ops.MatMul()(self.index, self.relative_position_bias_table)
        out = P.Reshape()(out, (self.window_size[0] * self.window_size[1],
                                self.window_size[0] * self.window_size[1], -1))
        out = P.Transpose()(out, (2, 0, 1))
        out = ops.ExpandDims()(out, 0)
        return out

class Roll(nn.Cell):
    """Roll Cell"""

    def __init__(self, shift_size, shift_axis=(1, 2)):
        super(Roll, self).__init__()
        self.shift_size = to_2tuple(shift_size)
        self.shift_axis = shift_axis

    def construct(self, x):
        x = numpy.roll(x, self.shift_size, self.shift_axis)
        return x
class WindowAttention(nn.Cell):
    r""" Window based multi-head self attention (W-MSA) Cell with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qZk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = Tensor(qk_scale or head_dim ** -0.5, mstype.float32)
        self.relative_bias = RelativeBias(self.window_size, num_heads)

        # get pair-wise relative position index for each token inside the window
        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = ops.Reshape()(self.q(x), (B_, N, self.num_heads, C // self.num_heads)) * self.scale
        q = ops.Transpose()(q, (0, 2, 1, 3))
        k = ops.Reshape()(self.k(x), (B_, N, self.num_heads, C // self.num_heads))
        k = ops.Transpose()(k, (0, 2, 3, 1))
        v = ops.Reshape()(self.v(x), (B_, N, self.num_heads, C // self.num_heads))
        v = ops.Transpose()(v, (0, 2, 1, 3))

        attn = ops.BatchMatMul()(q, k)
        attn = attn + self.relative_bias()

        if mask is not None:
            nW = mask.shape[1]
            attn = P.Reshape()(attn, (B_ // nW, nW, self.num_heads, N, N,)) + mask
            attn = P.Reshape()(attn, (-1, self.num_heads, N, N,))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = ops.Reshape()(ops.Transpose()(ops.BatchMatMul()(attn, v), (0, 2, 1, 3)), (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class SwinTransformerBlock(nn.Cell):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Cell, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Cell, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        if isinstance(dim, int):
            dim = (dim,)

        self.norm1 = norm_layer(dim, epsilon=1e-5)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim, epsilon=1e-5)
        mlp_hidden_dim = int((dim[0] if isinstance(dim, tuple) else dim) * mlp_ratio)
        self.mlp = Mlp(in_features=dim[0] if isinstance(dim, tuple) else dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # img_mask: [1, 56, 56, 1] window_size: 7
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            # [64, 49, 49] ==> [1, 64, 1, 49, 49]
            attn_mask = np.expand_dims(attn_mask, axis=1)
            attn_mask = np.expand_dims(attn_mask, axis=0)
            attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False)
            self.roll_pos = Roll(self.shift_size)
            self.roll_neg = Roll(-self.shift_size)
        else:
            self.attn_mask = None

        self.window_partition = WindowPartitionConstruct(self.window_size)
        self.window_reverse = WindowReverseConstruct()

    def construct(self, x):
        """construct function"""
        H, W = self.input_resolution
        B, _, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = P.Reshape()(x, (B, H, W, C,))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
            # shifted_x = numpy.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = self.window_partition(shifted_x)  # nW*B, window_size, window_size, C
        x_windows = ops.Reshape()(x_windows,
                                  (-1, self.window_size * self.window_size, C,))  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = P.Reshape()(attn_windows, (-1, self.window_size, self.window_size, C,))
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
            # x = numpy.roll(shifted_x, (self.shift_size, self.shift_size), (1, 2))  # TODO:Don't stupid
        else:
            x = shifted_x

        x = P.Reshape()(x, (B, H * W, C,))

        # FFN
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

class BasicLayer(nn.Cell):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Cell, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Cell | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,  # TODO: 这里window_size//2的时候特别慢
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def construct(self, x):
        B,C,H,W=x.shape
        x=P.Reshape()(x, (B, H*W, C,))
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = P.Reshape()(x, (B, C, H, W))
        return x


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


a = np.ones((2, 64,32,32))
a=a.astype(np.float32)
a = Tensor(a)
cnn=BasicLayer(64,(32,32),2,2,8)
c=cnn(a)

import warnings
import torch
import torch.nn as nn
import sys
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
sys.path.append('~/SDPL/models/')
from CheXRelFormerBaseNetworks import *
from help_funcs import TwoLayerConv2d
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.layers import DropPath, to_2tuple
import types
import math
from abc import ABCMeta, abstractmethod
import pdb
from pixel_shuffel_up import PS_UP
from typing import Sequence
from collections import OrderedDict
from norm import trunc_normal_

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio # [8, 4, 2, 1]
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def conv_scale(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels),
        nn.Linear(out_channels, out_channels),
        nn.ReLU()
    )

#Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

# Transformer Decoder
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

#Transormer Ecoder with x2, x4, x8, x16 scales
# Current settings:
#    'patch_size' : 7
#    'num_classes': 3
#    'embed_dims': [64, 128, 320, 512]
#    'num_heads':  [1, 2, 4, 8]
#    'mlp_ratios': [4, 4, 4, 4]
#    'qkv_bias': True
#    'drop_rate': 0.1
#    'attn_drop_rate': 0.1
#    'drop_path_rate': 0.1
#    'depths': [3, 3, 4, 3]
#    'sr_ratios': [8, 4, 2, 1]
    
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


# SDPL_class_decoder:
class Class_Decoder(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                     attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_query = norm_layer(dim)
        self.attn = Query_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SDPL_MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(input_resolution[0]*input_resolution[1])
        self.mlp2 = SDPL_MLP(in_features=input_resolution[0]*input_resolution[1], hidden_features=input_resolution[0]*input_resolution[1], act_layer=act_layer, drop=drop)
    def forward(self, query, feat):
        query, attn = self.attn(self.norm1_query(query), self.norm1(feat))
        query = query + self.drop_path(query)
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        attn = attn + self.drop_path(attn)
        attn = attn + self.drop_path(self.mlp2(self.norm3(attn)))
        return query, attn

class Query_Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.fc_q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.fc_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x):
        B, N, C = x.shape
        num_classes = q.shape[1]
        q = self.fc_q(q).reshape(B, self.num_heads, num_classes, C // self.num_heads)
        kv = self.fc_kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, num_head, N, C/num_head]
        attn1 = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_head, class, N]
        attn2 = attn1.softmax(dim=-1)
        attn3 = self.attn_drop(attn2)  # [B, num_head, class, N]
        x = (attn3 @ v).reshape(B, num_classes, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # [B, class, 256]
        attn=attn1.permute(0, 2, 1, 3) # # [B, class, num_head, N]
        return x,attn

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)
    
class SDPL_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# SDPL:
class SDPL(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, input_transform='multiple_select', decoder_softmax=False):
        super(SDPL, self).__init__()
        #Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3] #[3, 3, 6, 18, 3]
        self.output_nc = output_nc
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 
        self.input_transform = input_transform
        self.symp_embedding_dim = 64
        self.Tenc_x2 = SDPL_EncoderTransformer_v3(img_size=256, patch_size = 7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads = [1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])

        #Transformer Decoder
        self.params = {'in_chns': [512],
                  'feature_chns': [64, 128, 256],
                  'input_resolution':(8,),
                  'num_heads':(8,),
                  'depths':(2, ),
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': 8,
                  'bilinear': False,
                  'acti_func': 'relu'}
        
        self.TDec_x2 = Symtom_disentangler(
            in_chans=self.params['in_chns'], # different in_chans for different scales of feature maps
            depths=self.params['depths'],
            input_resolution=self.params['input_resolution'],
            num_classes=8,
            num_heads=self.params['num_heads'],
            norm_layer=nn.LayerNorm)
        
        self.prog_Dec = Progression_learner(
            num_classes=8,
            symp_embedding_dim = self.symp_embedding_dim,
            output_nc = self.output_nc
        )
    
        self.adapt_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_3 = nn.Linear(self.symp_embedding_dim, 1)
        self.fc_4 = nn.Linear(self.symp_embedding_dim, 1)
        self.make_pred = make_prediction_symp(self.symp_embedding_dim, self.output_nc)


    def forward(self, x1, x2):
        fx1 = self.Tenc_x2(x1)
        fx2 = self.Tenc_x2(x2)
        BS = fx1[-1].shape[0]
        mul_scale_feats_A = self.TDec_x2(fx1)
        mul_scale_feats_B = self.TDec_x2(fx2)
        # A presents the prior image, B presents the post image
        symp_pred_A_s3 = self.fc_3(self.adapt_avg_pooling(mul_scale_feats_A[-1]).flatten(1)).reshape(BS, self.params['class_num'])
        symp_pred_B_s3 = self.fc_3(self.adapt_avg_pooling(mul_scale_feats_B[-1]).flatten(1)).reshape(BS, self.params['class_num'])
        
        # Return symptom-level progression prediction and progression feature 
        prog_pred = self.prog_Dec(mul_scale_feats_A[-1], mul_scale_feats_B[-1], BS)

        output = {'symp_pred_A': symp_pred_A_s3, 
                  'symp_pred_B': symp_pred_B_s3, 
                  'prog_pred': prog_pred}
        
        return output

# SDPL_encoder:
class SDPL_EncoderTransformer_v3(nn.Module):
    def __init__(self, img_size=256, patch_size=3, in_chans=3, num_classes=3, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1]):
        super(SDPL_EncoderTransformer_v3, self).__init__()
        self.num_classes    = num_classes
        self.depths         = depths
        self.embed_dims     = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=patch_size, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=patch_size, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=patch_size, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
       
       # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        
        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        # stage 1
        x1, H1, W1 = self.patch_embed1(x)  # x1: [B, 64*64, 64]
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        # outs.append(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x2, H1, W1 = self.patch_embed2(x1) # x2: [B, 32*32, 128]
        for i, blk in enumerate(self.block2):
            x2 = blk(x2, H1, W1)
        x2 = self.norm2(x2)
        # outs.append(x1)
        x2 = x2.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        
        # stage 3
        x3, H1, W1 = self.patch_embed3(x2) # x3: [B, 16*16, 320]
        for i, blk in enumerate(self.block3):
            x3 = blk(x3, H1, W1)
        x3 = self.norm3(x3)
        # outs.append(x3)
        x3 = x3.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        
        # stage 4
        x4, H1, W1 = self.patch_embed4(x3) # x4: [B, 8*8, 512]
        for i, blk in enumerate(self.block4):
            x4 = blk(x4, H1, W1)
        x4 = self.norm4(x4)
        outs.append(x4)
        x4 = x4.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class Symtom_disentangler(nn.Module):
    def __init__(
        self,
        in_chans: Sequence[int], # different in_chans for different scales of feature maps
        depths: Sequence[int],
        input_resolution: Sequence[int], 
        num_classes: int,
        num_heads: Sequence[int],
        norm_layer: nn.LayerNorm,
        patch_norm: bool = False,
        spatial_dims: int = 2,
        drop_path_rate: float = 0.1

    ):
        super(Symtom_disentangler, self).__init__()  
        self.in_chans = in_chans
        self.patch_norm = patch_norm
        self.depth = depths
        self.hid_dim = 64
        self.num_classes = num_classes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # Tokenized Projection --> Normalized Layers --> Class Decoders --> Conv to fuse multi-head features
        self.proj_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.class_decoders = nn.ModuleList()
        self.attn_convs0 = nn.ModuleList()
        self.attn_convs1 = nn.ModuleList()
        self.attn_convs2 = nn.ModuleList()
        
        for i_layer in range(len(depths)):
            self.norm_layers.append(norm_layer(in_chans[i_layer]))
            self.class_decoders.append(Class_Decoder(dim=in_chans[i_layer], input_resolution=(input_resolution[i_layer], input_resolution[i_layer]),
                                 num_heads=num_heads[i_layer],
                                 mlp_ratio=4.,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0., 
                                 attn_drop=0.,
                                 drop_path=dpr[1],
                                 norm_layer=norm_layer))
            self.attn_convs0.append(SeparableConv2d(num_heads[i_layer]+in_chans[i_layer], self.hid_dim, (3,3), norm_layer=nn.BatchNorm2d, relu_first=False))                                
            self.attn_convs1.append(SeparableConv2d(self.hid_dim, self.hid_dim, (3,3), norm_layer=nn.BatchNorm2d, relu_first=False))
            self.attn_convs2.append(SeparableConv2d(self.hid_dim, self.hid_dim, (3,3), norm_layer=nn.BatchNorm2d, relu_first=False))
            
        # Symptom-aware Query Initialized -->
        self.sym_Q = []
        self.sym_Q_s3 = nn.Parameter(torch.zeros(1, num_classes, in_chans[-1]))
        
        self.register_parameter('sym_Q_s3', self.sym_Q_s3)
        trunc_normal_(self.sym_Q_s3, std=.02)
        self.sym_Q.append(self.sym_Q_s3)
        
    def forward(self, feats):
        mul_scale_feats = []
        BS = feats[0].shape[0]
        # Tokenized --> Normalized --> Class Decoders --> Conv to fuse multi-head features
        for i_layer in range(len(self.depth)):
            next_Q = self.sym_Q[i_layer].expand(BS, -1, -1)
            _, attn_map = self.class_decoders[i_layer](next_Q, feats[i_layer])

            bs, num_classes, num_heads, N_patch = attn_map.size()
            bs, N_tokens, dim = feats[i_layer].size()
            h, w = int(np.sqrt(N_patch)), int(np.sqrt(N_patch))
            
            attn_map = attn_map.reshape(bs*num_classes, num_heads, h, w)   # [BS*num_classes, num_heads, h, w]
            feat_map = feats[i_layer].permute(0, 2, 1).contiguous().view(bs, dim, h, w)
            fused_feat = torch.cat([_expand(feat_map, self.num_classes), attn_map], 1) # [BS*num_classes, num_heads+feat_dim, h, w]
            
            fused_feat = self.attn_convs0[i_layer](fused_feat)       # [BS*num_classes, hid_dim, h, w]
            fused_feat = self.attn_convs1[i_layer](fused_feat)       # [BS*num_classes, hid_dim, h, w]

            mul_scale_feats.append(fused_feat)

        return mul_scale_feats
    
def _expand(x, nclass):
    return x.unsqueeze(1).repeat(1, nclass, 1, 1, 1).flatten(0, 1)

class Progression_learner(nn.Module):
    def __init__(
        self,
        num_classes: int,
        symp_embedding_dim: int,
        output_nc: int,

    ) -> None:
        super(Progression_learner, self).__init__()  
        self.num_classes = num_classes
        self.symp_embedding_dim =symp_embedding_dim
        self.output_nc = output_nc

        self.symp_prog_fc = nn.ModuleList()

        for cls_ in range(self.num_classes):
            self.symp_prog_fc.append(nn.Linear(self.symp_embedding_dim, self.output_nc))

        self.adapt_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_prog = conv_diff_symp(self.symp_embedding_dim*2, self.symp_embedding_dim)

    def forward(self, feat_A, feat_B, BS):
        prog_feat = torch.cat((feat_A, feat_B), dim=1)
        prog_feat = self.conv_prog(prog_feat) # [BS*num_classes, hid_dim, h, w]
        prog_feat = self.adapt_avg_pooling(prog_feat).flatten(1).reshape(BS, self.num_classes, -1)

        symp_prog_pred = []
        # extract progression features of different symptoms
        for cls_ in range(self.num_classes):
            symp_prog_pred.append(self.symp_prog_fc[cls_](prog_feat[:, cls_, :]))

        return symp_prog_pred


def conv_diff_symp(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

def make_prediction_symp(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels),
        nn.Linear(out_channels, out_channels)
    )
# -*- coding: utf-8 -*-
#
# This script provides a standalone implementation of the VideoGigaGAN generator and discriminator,
# inspired by the paper "VideoGigaGAN: Towards Detail-rich Video Super-Resolution".
# Core components like SPyNet and flow_warp are re-implemented in pure PyTorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.ops import ModulatedDeformConv2d  # 使用 ModulatedDeformConv2d（DCNv2）
from collections import OrderedDict


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interpolation (str): The interpolation mode. Defaults to 'bilinear'.
        padding_mode (str): The padding mode. Defaults to 'zeros'.
        align_corners (bool): Same as F.grid_sample. Defaults to True.
    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    
    vgrid = grid + flow
    # scale grid to [-1,1]
    output = F.grid_sample(
        x,
        vgrid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):           
            self.basic_module.load_state_dict(torch.load(pretrained),strict=False)            
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        res = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return res + out


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResidualBlocksWithInputConv, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(num_blocks)])
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)     
        self.attn = SelfAttention(out_channels, norm_groups=1)

    def forward(self, x):
        x = self.lrelu(self.conv_in(x))
        x = self.blocks(x)
        x = self.attn(x)
        return x

class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        return x

# ###########################################################################
# # VideoGigaGAN Specific Modules
# ###########################################################################

class LowPassFilter(nn.Module):
    def __init__(self, channels):
        super(LowPassFilter, self).__init__()
        kernel = torch.tensor([[1, 4, 6, 4, 1]], dtype=torch.float32)
        kernel = (kernel.T @ kernel) / 256.0
        kernel = kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=2, groups=self.groups)

class AntiAliasingDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.low_pass = LowPassFilter(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv(x))
        feat_lf = self.low_pass(feat)
        feat_hf = feat - feat_lf
        feat_lf_down = F.interpolate(feat_lf, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return feat_lf_down, feat_hf

# ======================
# Temporal Attention 
# ======================
class TemporalAttentionModule(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # Identity initialization
        nn.init.constant_(self.temporal_conv.weight, 0)
        nn.init.constant_(self.temporal_conv.bias, 0)
        self.temporal_conv.weight.data[:, :, 1] = 1.0

    def forward(self, x):
        n, c, h, w = x.shape
        # This module expects a sequence, so we assume the input is flattened in the batch dim
        # The calling function should handle reshaping. Let's assume input is (n*t, c, h, w)
        # It's better to operate on (n, c, t, h, w) for video.
        # This implementation requires modification in the main model's propagate function.
        # Let's keep it simple and assume it's applied after propagation on the whole sequence.
        # For now, this is a placeholder. A real implementation needs careful sequence handling.
        return x # Placeholder

class SecondOrderDeformableAlignment(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, deform_groups=16, max_residue_magnitude=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude

        # The offset conv takes warped features, current feature, and flows
        offset_conv_in_channels = 3 * out_channels + 4
        self.conv_offset = nn.Sequential(
            nn.Conv2d(offset_conv_in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),
        )
        self.dcn = ModulatedDeformConv2d(
            in_channels, out_channels, kernel_size, padding=padding, deform_groups=deform_groups)
        self.init_offset()

    def init_offset(self):
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        
        num_repeats = offset_1.size(1) // 2
        offset_1 = offset_1 + flow_1.repeat(1, num_repeats, 1, 1)
        offset_2 = offset_2 + flow_2.repeat(1, num_repeats, 1, 1)

        offset = torch.cat([offset_1, offset_2], dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn(x, offset, mask)


class VideoGigaGAN_Generator(nn.Module):
    def __init__(self, mid_channels=64, num_blocks_extract=5, num_blocks_prop=7, spynet_pretrained=None, scale=4):
        super().__init__()
        self.mid_channels = mid_channels
        self.scale = scale
        assert scale in [2, 4], "scale must be 2 or 4"

        # Flow-guided propagation
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        
        # Feature extraction
        self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, num_blocks_extract)
        
        # Propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                in_channels=2 * mid_channels, out_channels=mid_channels, max_residue_magnitude=10)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks_prop)
        
        # Downsample for HF shuttle (only needed for scale=4)
        if scale == 4:
            self.down1 = AntiAliasingDownsample(mid_channels, mid_channels)
            self.down2 = AntiAliasingDownsample(mid_channels, mid_channels)
        elif scale == 2:
            self.down1 = AntiAliasingDownsample(mid_channels, mid_channels)
            self.down2 = None  # Not used

        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.fusion1 = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.fusion2 = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        
        # Upsampling layers
        if scale == 4:
            self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
            self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        elif scale == 2:
            self.upsample1 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
            self.upsample2 = None  # Not used

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)
        
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        
        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        t = len(feats['spatial'])
        n, _, h, w = feats['spatial'][0].size()
        
        feat_prop = torch.zeros(n, self.mid_channels, h, w, device=feats['spatial'][0].device)
        propagated_features = []

        # Determine frame order
        if 'backward' in module_name:
            frame_idx = list(reversed(range(t)))
            flow_idx = list(reversed(range(t - 1)))  # flows[i] = frame i+1 -> i
        else:  # forward
            frame_idx = list(range(t))
            flow_idx = list(range(t - 1))  # flows[i] = frame i -> i+1

        # Store previous propagated features for 2nd-order alignment
        feat_props = []

        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][idx]

            if i > 0:
                # First-order flow
                flow_n1 = flows[:, flow_idx[i - 1]]  # shape (n, 2, h, w)
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # Second-order (if available)
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = feat_props[-2]  # propagated feature from two steps ago
                    if 'backward' in module_name:
                        # backward: flow from idx+2 -> idx+1, then idx+1 -> idx
                        flow_n2_raw = flows[:, flow_idx[i - 2]]  # (idx+2 -> idx+1)
                    else:
                        flow_n2_raw = flows[:, flow_idx[i - 2]]  # (idx-2 -> idx-1)

                    # Accumulate flow: flow_n2 = flow_n1 + warp(flow_n2_raw, flow_n1)
                    flow_n2 = flow_n1 + flow_warp(flow_n2_raw, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # Prepare inputs for deformable alignment
                # x: features to align (concat of current prop and n2)
                x = torch.cat([feat_prop, feat_n2], dim=1)  # (n, 2*mid, h, w)
                # extra_feat: condition = [warped prop, current feat, warped n2]
                extra_feat = torch.cat([cond_n1, feat_current, cond_n2], dim=1)  # (n, 3*mid, h, w)

                # Apply deformable alignment
                feat_prop = self.deform_align[module_name](x, extra_feat, flow_n1, flow_n2)
            else:
                # First frame: no propagation, just use current feature as initial prop
                feat_prop = feat_current.clone()

            # Gather other features (from previous propagation branches)
            other_feats = [feats[k][idx] for k in feats if k not in ['spatial', module_name]]
            feat_in = torch.cat([feat_current, feat_prop] + other_feats, dim=1)

            # Refine with backbone
            feat_prop = feat_prop + self.backbone[module_name](feat_in)
            feat_props.append(feat_prop.clone())
            propagated_features.append(feat_prop)

        if 'backward' in module_name:
            propagated_features.reverse()

        feats[module_name] = propagated_features
        return feats

    def upsample(self, lqs, feats):
        outputs = []
        t = lqs.size(1)
        
        # Extract HF features conditionally
        hf_features1, hf_features2 = [], []
        for i in range(t):
            d1, hf1 = self.down1(feats['spatial'][i])
            hf_features1.append(hf1)
            if self.scale == 4:
                d2, hf2 = self.down2(d1)
                hf_features2.append(hf2)
            else:
                hf_features2.append(None)  # placeholder

        for i in range(t):
            hr_in = torch.cat([feats[k][i] for k in feats.keys()], dim=1)
            hr = self.reconstruction(hr_in)
            
            if self.scale == 4:
                hr = self.lrelu(self.upsample1(hr))
                hf2_up = F.interpolate(hf_features2[i], size=hr.shape[2:], mode='bilinear', align_corners=False)
                hr = self.fusion1(torch.cat([hr, hf2_up], dim=1))
                hr = self.lrelu(self.upsample2(hr))
                hf1_up = F.interpolate(hf_features1[i], size=hr.shape[2:], mode='bilinear', align_corners=False)
                hr = self.fusion2(torch.cat([hr, hf1_up], dim=1))
            elif self.scale == 2:
                hr = self.lrelu(self.upsample1(hr))
                hf1_up = F.interpolate(hf_features1[i], size=hr.shape[2:], mode='bilinear', align_corners=False)
                hr = self.fusion2(torch.cat([hr, hf1_up], dim=1))

            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += self.img_upsample(lqs[:, i])
            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        
        feats_spatial = self.feat_extract(lqs.view(-1, c, h, w)).view(n, t, self.mid_channels, h, w)
        feats_spatial = [feats_spatial[:, i] for i in range(t)]
        
        flows_forward, flows_backward = self.compute_flow(lqs)
        
        feats = OrderedDict()
        feats['spatial'] = feats_spatial
        
        # Propagation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                flows = flows_backward if 'backward' in direction else flows_forward
                feats = self.propagate(feats, flows, module)
        
        return self.upsample(lqs, feats)


class VideoDiscriminator(nn.Module):
    """A simple video discriminator."""
    def __init__(self, in_channels=3, mid_channels=64):
        super().__init__()
        self.conv0_0 = nn.Conv3d(in_channels, mid_channels, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.conv0_1 = nn.Conv3d(mid_channels, mid_channels, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        
        self.conv1_0 = nn.Conv3d(mid_channels, mid_channels*2, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.conv1_1 = nn.Conv3d(mid_channels*2, mid_channels*2, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        
        self.conv2_0 = nn.Conv3d(mid_channels*2, mid_channels*4, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.conv2_1 = nn.Conv3d(mid_channels*4, mid_channels*4, (1, 3, 3), (1, 1, 1), (0, 1, 1))

        self.temporal_conv = nn.Conv3d(mid_channels*4, mid_channels*4, (3,1,1), (1,1,1), (1,0,0))
        
        self.final_conv = nn.Conv2d(mid_channels*4, 1, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # input: (n, t, c, h, w) -> (n, c, t, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.lrelu(self.conv0_0(x))
        x = self.lrelu(self.conv0_1(x))
        x = self.lrelu(self.conv1_0(x))
        x = self.lrelu(self.conv1_1(x))
        x = self.lrelu(self.conv2_0(x))
        x = self.lrelu(self.conv2_1(x))
        
        x = self.lrelu(self.temporal_conv(x))
        
        # Flatten time dimension
        n, c, t, h, w = x.shape
        x = x.permute(0,2,1,3,4).reshape(n*t, c, h, w)
        
        out = self.final_conv(x)
        return out
if __name__ == "__main__":    
    model = VideoGigaGAN_Generator(spynet_pretrained='../ckpt/spynet_20210409-c6c1bd09.pth',scale=4) 
    input_video = torch.randn(2, 10, 3, 64, 64)  # (B, T, 3, H, W)
    output = model(input_video)  
    #torch.save(model, "model.pth")
    print(output.shape)
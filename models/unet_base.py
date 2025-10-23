from typing import Union, Optional, List, Tuple, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.minkowskinet.resunet import (
    ResNetBase,
    NormType,
    ConvType,
    conv,
    conv_tr,
    get_norm,
    BasicBlock,
    Res16UNetBase,
    ResUNetConfig,
)
from models.minkowskinet.dense_to_sparse import dense_to_sparse
from models.time_embedding import get_timestep_embedding


def forward_block(block, input, time_emb):
    for layer in block:
        input = layer(input, time_emb)
    return input


@torch.no_grad()
def get_target(out, target_key, kernel_size=1):
    target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
    cm = out.coordinate_manager
    strided_target_key = cm.stride(
        target_key,
        out.tensor_stride[0],
    )
    kernel_map = cm.kernel_map(
        out.coordinate_map_key,
        strided_target_key,
        kernel_size=kernel_size,
        region_type=1,
    )
    # assert len(kernel_map) == 1
    if len(kernel_map) > 1:
        print(f"{out.coordinate_map_key} and {strided_target_key} has {len(kernel_map)} kernel map")

    indices_pred_list = []
    indices_gt_list = []
    for k, curr_in in kernel_map.items():
        # print(strided_target_key, curr_in.shape)
        # breakpoint()
        target[curr_in[0].long()] = 1
        indices_pred = curr_in[0].long()
        indices_gt = curr_in[1].long()

        if len(indices_pred) == 0:
            print("No indices pred")
        else:
            indices_pred_list.append(indices_pred)

        if len(indices_gt) == 0:
            print("No indices in GT")
        else:
            indices_gt_list.append(indices_gt)

    if len(indices_pred_list) == 0:
        indices_pred = None
    else:
        indices_pred = torch.cat(indices_pred_list, dim=0)

    if len(indices_gt_list) == 0:
        indices_gt = None
    else:
        indices_gt = torch.cat(indices_gt_list, dim=0)

    xyz_pred = out.C.clone()
    # DEBUG
    # indices_pred = None
    # indices_gt = None
    return target, indices_pred, indices_gt, xyz_pred


class DenseResBlock3D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,  # For backward compatibility
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(input_dim, output_dim, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(output_dim)
        self.conv2 = nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu(out)
        return out


class BasicBlockWithTime(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        time_emb_dim=0,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super(BasicBlockWithTime, self).__init__()

        self.conv1 = conv(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            conv_type=conv_type,
            D=D,
        )
        self.time_emb_dim = time_emb_dim
        if time_emb_dim > 0:
            self.time_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(time_emb_dim, planes),
            )

        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x, time_emb=None):
        residual = x

        out = self.conv1(x)
        if self.time_emb_dim > 0:
            assert time_emb is not None
            time_emb = self.time_proj(time_emb)
            time_emb = time_emb.expand(out.F.shape[0], -1)
            time_emb_sparse = ME.SparseTensor(
                features=time_emb,
                coordinate_map_key=out.coordinate_map_key,
                coordinate_manager=out.coordinate_manager,
            )
            out = out + time_emb_sparse

        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res16UNetWithTime(Res16UNetBase):
    BLOCK = BasicBlockWithTime
    PLANES = (64, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 64
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: ResUNetConfig,
        use_time_emb: bool = False,
        time_emb_dim: int = 64,
        D: int = 3,
        **kwargs,
    ):
        self.use_time_emb = use_time_emb
        self.temb_ch = time_emb_dim

        super().__init__(
            in_channels,
            out_channels,
            config,
            D,
        )

        if self.use_time_emb:
            self.temb = nn.Sequential(
                nn.Linear(self.temb_ch, self.temb_ch),
                nn.ReLU(),
                nn.Linear(self.temb_ch, self.temb_ch),
            )

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_type=NormType.BATCH_NORM,
        bn_momentum=0.1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D),
                get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                time_emb_dim=self.temb_ch if self.use_time_emb else 0,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    time_emb_dim=self.temb_ch if self.use_time_emb else 0,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D,
                )
            )
        return nn.Sequential(*layers)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)

        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr4p16s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr5p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr6p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr7p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[7],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, timestamp=None, memory_module=None):

        def forward_block(block, input, time_emb):
            for layer in block:
                input = layer(input, time_emb)
            return input

        if self.use_time_emb:
            assert timestamp is not None
            # print(get_timestep_embedding(timestamp, self.temb_ch))
            # breakpoint()
            time_emb = get_timestep_embedding(timestamp, self.temb_ch)
            time_emb = self.temb(time_emb)
        else:
            time_emb = None

        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)

        out_b1p2 = forward_block(self.block1, out, time_emb)
        # out_b1p2 = self.block1(out, time_emb)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = forward_block(self.block2, out, time_emb)
        # out_b2p4 = self.block2(out, time_emb)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = forward_block(self.block3, out, time_emb)
        # out_b3p8 = self.block3(out, time_emb)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = forward_block(self.block4, out, time_emb)
        # out = self.block4(out, time_emb)

        if memory_module is not None:
            # out.F = memory_module(out.F, timestamp)
            out = ME.SparseTensor(
                features=memory_module(out.F, timestamp),
                coordinate_map_key=out.coordinate_map_key,
                coordinate_manager=out.coordinate_manager,
            )

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = ME.cat(out, out_b3p8)
        out = forward_block(self.block5, out, time_emb)
        # out = self.block5(out, time_emb)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = ME.cat(out, out_b2p4)
        out = forward_block(self.block6, out, time_emb)
        # out = self.block6(out, time_emb)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = ME.cat(out, out_b1p2)
        out = forward_block(self.block7, out, time_emb)
        # out = self.block7(out, time_emb)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = ME.cat(out, out_p1)
        out = forward_block(self.block8, out, time_emb)
        # out = self.block8(out, time_emb)

        return self.final(out), out


class GenerativeRes16UNetCatBase(ResNetBase):
    BLOCK = BasicBlockWithTime
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config,
        use_time_emb: bool = False,
        time_emb_dim: int = 64,
        D: int = 3,
        dense_bottleneck: bool = False,
        num_dense_blocks: int = 2,
        **kwargs,
    ):
        self.use_time_emb = use_time_emb
        self.temb_ch = time_emb_dim
        self.dense_bottleneck = dense_bottleneck
        self.num_dense_blocks = num_dense_blocks
        if self.num_dense_blocks == 0:
            self.dense_bottleneck = False

        super().__init__(
            in_channels,
            out_channels,
            config,
            D,
        )

        if self.use_time_emb:
            self.temb = nn.Sequential(
                nn.Linear(self.temb_ch, self.temb_ch),
                nn.ReLU(),
                nn.Linear(self.temb_ch, self.temb_ch),
            )

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilation=1,
        norm_type=NormType.BATCH_NORM,
        bn_momentum=0.1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D),
                get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                time_emb_dim=self.temb_ch if self.use_time_emb else 0,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    time_emb_dim=self.temb_ch if self.use_time_emb else 0,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D,
                )
            )
        return nn.Sequential(*layers)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)

        self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        if self.dense_bottleneck:
            self.dense_blocks = nn.Sequential(
                *[DenseResBlock3D(self.PLANES[3], self.PLANES[3]) for _ in range(self.num_dense_blocks)]
            )

        self.convtr4p16s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.inplanes,
            out_channels=self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dimension=D,
            bias=False,
        )
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block5_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[4],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        self.convtr5p8s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.inplanes,
            out_channels=self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dimension=D,
            bias=False,
        )
        self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block6_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[5],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        self.convtr6p4s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.inplanes,
            out_channels=self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dimension=D,
            bias=False,
        )

        self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block7_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[6],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        self.convtr7p2s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.inplanes,
            out_channels=self.PLANES[7],
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dimension=D,
            bias=False,
        )
        self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block8_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[7],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )
        self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.pruning = ME.MinkowskiPruning()

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(
            target_key,
            out.tensor_stride[0],
        )
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def concat(self, x: ME.SparseTensor, y: ME.SparseTensor):
        """Hacky way to concatenate two sparse tensors:
            1. Expand the feature dimension of x and y to match each other
            2. Add the two tensors
        """
        x_feat = x.features
        y_feat = y.features
        x_dummy = torch.zeros(x_feat.shape[0], y_feat.shape[1], dtype=x_feat.dtype, device=x_feat.device)
        y_dummy = torch.zeros(y_feat.shape[0], x_feat.shape[1], dtype=y_feat.dtype, device=y_feat.device)
        x_feat = torch.cat([x_feat, x_dummy], axis=-1)
        y_feat = torch.cat([y_dummy, y_feat], axis=-1)

        # x._F = x_feat
        # y._F = y_feat
        # return x + y
        out_x = ME.SparseTensor(
            x_feat,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        out_y = ME.SparseTensor(
            y_feat,
            coordinate_map_key=y.coordinate_map_key,
            coordinate_manager=y.coordinate_manager,
        )
        return out_x + out_y

    def forward(
        self,
        x: ME.SparseTensor,
        gt_coords: torch.Tensor,
        anchor_coords: Optional[torch.Tensor] = None,
        ignore_coords: Optional[torch.Tensor] = None,
        timestamp: Optional[int] = None,
        threshold: float = 0.0,
        threshold_second_last: Optional[float] = None,
        threshold_last: Optional[float] = None,
        sample_n_last: Optional[int] = None,
        sample_last_strict: bool = False,
        sample_temperature: float = 1.0,
        max_gt_samples_last: Optional[int] = None,
        max_samples_second_last: Optional[int] = None,
        eval_randomness: bool = False,
        verbose: bool = False,
    ) -> Tuple[ME.SparseTensor, List[torch.Tensor], List[ME.SparseTensor]]:
        """
        Args:
            x: SparseTensor
            gt_coords (torch.Tensor):
                sparse coordinates (N, 4) for all the occupied voxels
            max_gt_samples_last (int): Maximum number of samples to keep from the last layer for training.
                                       Useful to reduce GPU memory usage during training.

        Returns:
            Output feature: ME.SparseTensor
            GT occupancy at different levels: List[torch.Tensor]
            Predicted occupancy at different levels: List[ME.SparseTensor]
        """

        if self.use_time_emb:
            assert timestamp is not None
            # print(get_timestep_embedding(timestamp, self.temb_ch))
            # breakpoint()
            time_emb = get_timestep_embedding(timestamp, self.temb_ch)
            time_emb = self.temb(time_emb)
        else:
            time_emb = None

        assert gt_coords.ndim == 2

        target_key, _ = x.coordinate_manager.insert_and_map(
            gt_coords,
            string_id="target",
        )

        if ignore_coords is not None:
            ignore_key, _ = x.coordinate_manager.insert_and_map(
                ignore_coords,
                string_id="ignore",
            )

        if anchor_coords is not None:
            anchor_key, _ = x.coordinate_manager.insert_and_map(
                anchor_coords,
                string_id="anchor",
            )

        if max_gt_samples_last is not None:
            # Uniformly sample the last sample_n_last points
            if max_gt_samples_last < gt_coords.shape[0]:
                indices = torch.randperm(gt_coords.shape[0])[:max_gt_samples_last]
                gt_coords_sub = gt_coords[indices].clone()
            else:
                gt_coords_sub = gt_coords.clone()
            subsample_key, _ = x.coordinate_manager.insert_and_map(
                gt_coords_sub,
                string_id="subsample",
            )

        targets = []
        ignores = []
        out_cls = []
        after_prune = []
        xyz_preds = []
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = forward_block(self.block1, out, time_emb)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = forward_block(self.block2, out, time_emb)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = forward_block(self.block3, out, time_emb)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = forward_block(self.block4, out, time_emb)

        ######## From sparse to dense (bottleneck) ########
        if self.dense_bottleneck:
            dense_tensor, min_coordinate, stride = out.dense(min_coordinate=torch.IntTensor([0, 0, 0]), contract_stride=True)
            dense_tensor = self.dense_blocks(dense_tensor)

            out1 = dense_to_sparse(
                dense_tensor,
                stride=out.tensor_stride[0],
                device=dense_tensor.device,
                coordinate_manager=out.coordinate_manager,
            )
            # print("Before denify", out.shape, "After densify", out1.shape)
            out = out1
        ###################################################

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = self.concat(out, out_b3p8)
        # out = self.block5(out)
        out = forward_block(self.block5, out, time_emb)

        # Prune the voxels at res 1/8
        out_cls5 = self.block5_cls(out)
        target, _, _, xyz_pred = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls5)
        xyz_preds.append(xyz_pred)

        if ignore_coords is not None:
            ignore, _, _, _ = get_target(out, ignore_key)
            ignores.append(ignore)

        keep5 = (out_cls5.F > threshold).squeeze()
        # if True:
        if self.training:
            if max_gt_samples_last is not None:
                target_sub, _, _, _ = get_target(out, subsample_key)
                if verbose:
                    print(f"Layer 5: {target_sub.sum()} samples from subsample, {target.sum()} samples from target, {keep5.sum()} thresholded, total {out.F.shape[0]}")
                keep5 += target_sub
            else:
                # keep = target      # DEBUG
                keep5 += target

        if anchor_coords is not None:
            anchor, _, _, _ = get_target(out, anchor_key)
            keep5 += anchor

        if verbose:
            print(f"Layer 5: keep5 {keep5.sum()} thresholded, total {out.F.shape[0]}")

        out = self.pruning(out, keep5)
        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 5 (1/8 resolution). It has {keep5.shape} voxels before")
        after_prune.append(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = self.concat(out, out_b2p4)
        # out = self.block6(out)
        out = forward_block(self.block6, out, time_emb)

        # Prune the voxels at res 1/4
        out_cls6 = self.block6_cls(out)
        target, _, _, xyz_pred = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls6)
        xyz_preds.append(xyz_pred)

        if ignore_coords is not None:
            ignore, _, _, _ = get_target(out, ignore_key)
            ignores.append(ignore)

        keep6 = (out_cls6.F > threshold).squeeze()

        if self.training:
            if max_gt_samples_last is not None:
                target_sub, _, _, _ = get_target(out, subsample_key)
                if verbose:
                    print(f"Layer 6: {target_sub.sum()} samples from subsample, {target.sum()} samples from target, {keep6.sum()} thresholded, total {out.F.shape[0]}")
                keep6 += target_sub
            else:
                # keep6 = target      # DEBUG
                keep6 += target

        if anchor_coords is not None:
            anchor, _, _, _ = get_target(out, anchor_key)
            keep6 += anchor

        if verbose:
            print(f"Layer 6: keep6 {keep6.sum()} thresholded, total {out.F.shape[0]}")

        out = self.pruning(out, keep6)
        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 6 (1/4 resolution). It has {keep6.shape} voxels before. Max threshold {out_cls6.F.max()}")
        after_prune.append(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = self.concat(out, out_b1p2)
        # out = self.block7(out)
        out = forward_block(self.block7, out, time_emb)

        # Prune the voxels at res 1/2
        out_cls7 = self.block7_cls(out)
        target, _, _, xyz_pred = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls7)
        xyz_preds.append(xyz_pred)

        if ignore_coords is not None:
            ignore, _, _, _ = get_target(out, ignore_key)
            ignores.append(ignore)

        keep7 = (out_cls7.F > threshold).squeeze()
        if anchor_coords is not None:
            anchor, _, _, _ = get_target(out, anchor_key)
            keep7 += anchor

        # ==============================
        # In big scenes, keep7 will generate too many samples after the next upsampling
        # To prevent GPU OOM, we can either:
        # Always subsample same numbers, not sure how much worse will in this case
        # Or simipling using higher thresholding for this level
        if threshold_second_last is not None:
            assert threshold_second_last >= 0 and threshold_second_last <= 1, f"Threshold_second_last should be in [0, 1], got {threshold_second_last}"
            prob7 = torch.sigmoid(out_cls7.F).squeeze()
            keep7 = (prob7 > threshold_second_last)

            if anchor_coords is not None:
                anchor, _, _, _ = get_target(out, anchor_key)
                keep7 += anchor

        elif max_samples_second_last is not None:
            with torch.no_grad():
                # keep7 = torch.zeros_like(out_cls7.F.squeeze(), dtype=torch.bool)
                # threshold_indices = (out_cls7.F > threshold).squeeze().nonzero().squeeze()
                threshold_indices = keep7.nonzero().squeeze()
                keep7 = torch.zeros_like(out_cls7.F.squeeze(), dtype=torch.bool)

                if threshold_indices.shape[0] > max_samples_second_last:
                    if self.training:
                        # uniformly sample max_samples_second_last from the thresholded indices
                        indices = torch.randperm(threshold_indices.shape[0])[:max_samples_second_last]
                        keep7[threshold_indices[indices]] = 1
                    else:
                        if eval_randomness:
                            # Randomly sample max_samples_second_last from the thresholded indices
                            indices = torch.randperm(threshold_indices.shape[0])[:max_samples_second_last]
                            keep7[threshold_indices[indices]] = 1
                        else:
                            # Choose the largest max_samples_second_last indices
                            indices = torch.topk(out_cls7.F.squeeze(), max_samples_second_last).indices
                            keep7[indices] = 1
                else:
                    keep7[threshold_indices] = 1
        # ==============================

        if self.training:
            if max_gt_samples_last is not None:
                target_sub, _, _, _ = get_target(out, subsample_key)
                if verbose:
                    print(f"Layer 7: {target_sub.sum()} samples from subsample, {target.sum()} samples from target, {keep7.sum()} thresholded, total {out.F.shape[0]}")
                keep7 += target_sub
            else:
                # keep7 = target      # DEBUG
                keep7 += target

        if verbose:
            print(f"Layer 7: keep7 {keep7.sum()} thresholded, total {out.F.shape[0]}")

        out = self.pruning(out, keep7)
        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 7 (1/2 resolution). It has {keep7.shape} voxels before. Max threshold {out_cls7.F.max()}")
        after_prune.append(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = self.concat(out, out_p1)
        # out = self.block8(out)
        out = forward_block(self.block8, out, time_emb)

        # Prune the voxels at orig
        out_cls8 = self.block8_cls(out)
        target, _, _, xyz_pred = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls8)
        xyz_preds.append(xyz_pred)

        if ignore_coords is not None:
            ignore, _, _, _ = get_target(out, ignore_key)
            ignores.append(ignore)

        if threshold_last is not None:
            assert threshold_last >= 0 and threshold_last <= 1, f"Threshold_last should be in [0, 1], got {threshold_last}"
            prob8 = torch.sigmoid(out_cls8.F).squeeze()
            keep8 = (prob8 > threshold_last)
        elif sample_n_last is not None:
            with torch.no_grad():
                if sample_n_last >= out_cls8.F.shape[0]:
                    # Simply keep all the samples
                    keep8 = torch.ones_like(out_cls8.F.squeeze(), dtype=torch.bool)
                else:
                    if self.training:
                        # Sample n voxels from the last layer based on the probability with temperature
                        prob8 = torch.sigmoid(out_cls8.F / sample_temperature).squeeze()

                        if sample_last_strict:
                            # The output of the last layer would be exactly "sample_n_last" samples
                            # TODO: Try prob8[target_sub] = 1 to increase the chance of generating new samples
                            # TODO: Try prob8[ignore_key] = 0.00001 to decrease the chance of generating new samples
                            target_sub, _, _, _ = get_target(out, subsample_key)
                            EPS = 1e-8
                            prob8[target_sub] = 1 - EPS
                            prob8[ignore] = EPS

                        keep8 = torch.zeros_like(prob8, dtype=torch.bool)
                        indices = torch.multinomial(prob8, sample_n_last, replacement=False)
                        keep8[indices] = 1
                    else:
                        if eval_randomness:
                            # Sample n voxels from the last layer based on the probability with temperature
                            prob8 = torch.sigmoid(out_cls8.F / sample_temperature).squeeze()
                            if sample_last_strict:
                                prob8[ignore] = 1e-8
                            indices = torch.multinomial(prob8, sample_n_last, replacement=False)
                            keep8 = torch.zeros_like(prob8, dtype=torch.bool)
                            keep8[indices] = 1
                        else:
                            # Pick the largest n voxels
                            prob8 = torch.sigmoid(out_cls8.F).squeeze()
                            if sample_last_strict:
                                prob8[ignore] = 0.0
                            # TODO: Try prob8[ignore_key] = 0.00001 to decrease the chance of generating new samples
                            indices = torch.topk(prob8, sample_n_last).indices
                            keep8 = torch.zeros_like(prob8, dtype=torch.bool)
                            keep8[indices] = 1
        else:
            keep8 = (out_cls8.F > threshold).squeeze()
        # TODO: Add an option to use keep8 = (out_cls8.F > threshold).squeeze() but ignore all the samples in ignore_key

        if self.training and not sample_last_strict:
            # TODO: Try removing this part
            if max_gt_samples_last is not None:
                target_sub, _, _, _ = get_target(out, subsample_key)
                if verbose:
                    print(f"Layer 8: {target_sub.sum()} samples from subsample, {target.sum()} samples from target, {keep8.sum()} thresholded, total {out.F.shape[0]}")
                keep8 += target_sub
            else:
                keep8 += target

        prob_last = out_cls8.F[keep8]

        if verbose:
            print(f"Layer 8: keep8 {keep8.sum()} thresholded, total {out.F.shape[0]}")
        out = self.pruning(out, keep8)
        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 8 (orig resolution). It has {keep8.shape} voxels before")
        # print("After pruning", out.F.shape)
        after_prune.append(out)

        out = self.final(out)
        _, indices_pred, indices_gt, _ = get_target(out, target_key)
        ignore_final, _, _, _ = get_target(out, ignore_key)

        return {
            "out": out,
            "targets": targets,
            "ignores": ignores,
            "out_cls": out_cls,
            # "after_prune": after_prune,
            "indices_pred": indices_pred,
            "indices_gt": indices_gt,
            "ignore_final": ignore_final,
            "xyz_preds": xyz_preds,
            "last_prob": prob_last,     # it's the probability logits of the last layer (before sigmoid)
        }

    def forward_encoder(self, x: ME.SparseTensor, timestamp: Optional[int] = None):
        if self.use_time_emb:
            assert timestamp is not None
            time_emb = get_timestep_embedding(timestamp, self.temb_ch)
            time_emb = self.temb(time_emb)
        else:
            time_emb = None

        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = forward_block(self.block1, out, time_emb)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = forward_block(self.block2, out, time_emb)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = forward_block(self.block3, out, time_emb)

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_b4p16 = forward_block(self.block4, out, time_emb)

        return time_emb, out_p1, out_b1p2, out_b2p4, out_b3p8, out_b4p16

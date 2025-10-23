from typing import Tuple
from dataclasses import dataclass
import os
from urllib.request import urlopen, Request

import torch
from MinkowskiEngine import MinkowskiReLU
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me

from .resnet import ResNetBase, get_norm
from .modules.common import ConvType, NormType, conv, conv_tr
from .modules.resnet_block import BasicBlock, Bottleneck


@dataclass
class ResUNetConfig:
    bn_momentum: float = 0.02
    conv1_kernel_size: int = 3
    dilations: Tuple[int, int, int, int] = (1, 1, 1, 1)


def load_pretrain_weights(model):
    """Download pretrained weights and load them into the model. If the weights are already downloaded, just load them."""
    PRETRAIN_MODEL_URL = {
        "res16unet34c": "https://kaldir.vc.in.tum.de/rozenberszki/language_grounded_semseg/Weights/34C/34C_CLIP_pretrain.ckpt",
        "res16unet34d": "https://kaldir.vc.in.tum.de/rozenberszki/language_grounded_semseg/Weights/34D/34D_CLIP_pretrain.ckpt",
    }
    if isinstance(model, Res16UNet34C):
        model_url = PRETRAIN_MODEL_URL["res16unet34c"]

    elif isinstance(model, Res16UNet34D):
        model_url = PRETRAIN_MODEL_URL["res16unet34d"]
    else:
        raise ValueError(f"Model {model} not supported")

    assert model.in_channels == 3, "The input channels must be 3 for the pretrained weights"
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "torch", "minkowski", model_url.split("/")[-1])
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Downloading {model_url} to {cache_path}")
        with open(cache_path, "wb") as f:
            f.write(urlopen(Request(model_url, headers={"User-Agent": "Mozilla/5.0"})).read())
    ckpt = torch.load(cache_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for key, val in state_dict.items():
        if key.startswith("model."):
            key = key[6:]
            new_state_dict[key] = val

    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded pretrain weights from {cache_path}")


class Res16UNetBase(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNetBase, self).__init__(in_channels, out_channels, config, D)

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
        self.relu = MinkowskiReLU(inplace=True)

    def forward(self, x):
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = me.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out), out


class GenerativeRes16UNetBase(ResNetBase):
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(GenerativeRes16UNetBase, self).__init__(in_channels, out_channels, config, D)

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
        # self.convtr4p16s2 = conv_tr(
        #     self.inplanes,
        #     self.PLANES[4],
        #     kernel_size=space_n_time_m(2, 1),
        #     upsample_stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     bias=False,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)

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

        # self.convtr5p8s2 = conv_tr(
        #     self.inplanes,
        #     self.PLANES[5],
        #     kernel_size=space_n_time_m(2, 1),
        #     upsample_stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     bias=False,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)
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
        # self.convtr6p4s2 = conv_tr(
        #     self.inplanes,
        #     self.PLANES[6],
        #     kernel_size=space_n_time_m(2, 1),
        #     upsample_stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     bias=False,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)

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
        # self.convtr7p2s2 = conv_tr(
        #     self.inplanes,
        #     self.PLANES[7],
        #     kernel_size=space_n_time_m(2, 1),
        #     upsample_stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     bias=False,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)
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
        self.relu = MinkowskiReLU(inplace=True)

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

    def forward(self, x, target_key):
        targets = []
        out_cls = []
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        breakpoint()
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # Prune the voxels at res 1/8
        out_cls5 = self.block5_cls(out)
        breakpoint()
        target = self.get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls5)
        keep5 = (out_cls5.F > 0).squeeze()
        if self.training:
            keep5 += target
        out = self.pruning(out, keep5)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # Prune the voxels at res 1/4
        out_cls6 = self.block6_cls(out)
        target = self.get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls6)
        keep6 = (out_cls6.F > 0).squeeze()
        if self.training:
            keep6 += target
        out = self.pruning(out, keep6)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # Prune the voxels at res 1/2
        out_cls7 = self.block7_cls(out)
        target = self.get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls7)
        keep7 = (out_cls7.F > 0).squeeze()
        if self.training:
            keep7 += target
        out = self.pruning(out, keep7)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = me.cat(out, out_p1)
        out = self.block8(out)

        # Prune the voxels at orig
        out_cls8 = self.block8_cls(out)
        target = self.get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls8)
        keep8 = (out_cls8.F > 0).squeeze()
        if self.training:
            keep8 += target
        out = self.pruning(out, keep8)

        return self.final(out), out


class Res16UNet14(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet50(Res16UNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
    LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
    LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class Res16UNet34D(Res16UNet34):  # For CLIP
    PLANES = (32, 64, 128, 256, 256, 256, 256, 512)


class Res16UNet34C200(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 200)

class Res16UNet34C100(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 128, 100)

class STRes16UNetBase(Res16UNetBase):
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
        super(STRes16UNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class STRes16UNet14(STRes16UNetBase, Res16UNet14):
    pass


class STRes16UNet14A(STRes16UNetBase, Res16UNet14A):
    pass


class STRes16UNet18(STRes16UNetBase, Res16UNet18):
    pass


class STRes16UNet34(STRes16UNetBase, Res16UNet34):
    pass


class STRes16UNet50(STRes16UNetBase, Res16UNet50):
    pass


class STRes16UNet101(STRes16UNetBase, Res16UNet101):
    pass


class STRes16UNet18A(STRes16UNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class STResTesseract16UNetBase(STRes16UNetBase):
    CONV_TYPE = ConvType.HYPERCUBE


class STResTesseract16UNet18A(STRes16UNet18A, STResTesseract16UNetBase):
    pass
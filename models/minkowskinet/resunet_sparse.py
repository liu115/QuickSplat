from typing import Union, Optional, List, Tuple

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from .resunet import ResNetBase, NormType, ConvType, conv, get_norm, BasicBlock


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
    for k, curr_in in kernel_map.items():
        target[curr_in[0].long()] = 1
    return target


def dense_to_sparse(
    x: torch.Tensor,
    stride: int,
    device: torch.device,
) -> ME.SparseTensor:
    """Convert torch 3D tensor to sparse tensor

    Args:
        x (torch.Tensor): shape (B, C, D, H, W)
        stride (int): How many downsampled times

    Returns:
        ME.SparseTensor that has the size (B * D * H * W, C)
    """
    assert x.ndim == 5
    with torch.no_grad():
        coords = torch.meshgrid(
            torch.arange(x.shape[0], device=device, dtype=torch.int32),
            torch.arange(x.shape[2], device=device, dtype=torch.int32),
            torch.arange(x.shape[3], device=device, dtype=torch.int32),
            torch.arange(x.shape[4], device=device, dtype=torch.int32),
            indexing="ij",
        )
        coords = torch.stack([c.flatten() for c in coords], dim=-1)
    coords[:, 1:] *= stride
    feats = x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])
    # print("before", x.shape)
    out = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        tensor_stride=stride,
        device=x.device,
    )
    # tmp = out.dense()
    # print("after", tmp[0].shape, tmp[1], tmp[2])
    return out


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
        super().__init__(in_channels, out_channels, config, D)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        # self.inplanes = self.INIT_DIM
        self.inplanes = self.PLANES[0]
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

        self.convtr4p16s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.inplanes,
            out_channels=self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dimension=D,
            bias=False,
        )
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[4]
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

        self.inplanes = self.PLANES[5]
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

        self.inplanes = self.PLANES[6]
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

        self.inplanes = self.PLANES[7]
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

    def forward(
        self,
        x: Union[torch.Tensor, ME.SparseTensor],
        gt_coords: torch.Tensor,
    ) -> Tuple[ME.SparseTensor, List[torch.Tensor], List[ME.SparseTensor]]:
        """
        Args:
            x (Union[torch.Tensor, ME.SparseTensor]):
                dense feature volume (B, C, H, W, D) or sparse tensor
            gt_coords (torch.Tensor):
                dense volume (B, H, W, D) with 0 or 1
                sparse coordinates (N, 4) for all the occupied voxels
        Returns:
            Output feature: ME.SparseTensor
            GT occupancy at different levels: List[torch.Tensor]
            Predicted occupancy at different levels: List[ME.SparseTensor]
        """

        if isinstance(x, torch.Tensor):
            # Convert dense tensor to sparse tensor
            x = dense_to_sparse(x, stride=1, device=gt_coords.device)

        if gt_coords.ndim == 4:
            # (B, H, W, D) -> (N, 4) where > 0
            gt_coords = torch.stack(
                torch.where(gt_coords > 0), dim=1
            ).int()
        assert gt_coords.ndim == 2

        target_key, _ = x.coordinate_manager.insert_and_map(
            gt_coords,
            string_id="target",
        )

        targets = []
        out_cls = []
        after_prune = []
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
        # out = me.cat(out, out_b3p8)
        out = out + out_b3p8
        out = self.block5(out)

        # Prune the voxels at res 1/8
        out_cls5 = self.block5_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls5)
        keep5 = (out_cls5.F > 0).squeeze()
        if self.training:
            keep5 += target
        out = self.pruning(out, keep5)
        after_prune.append(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        # out = me.cat(out, out_b2p4)
        out = out + out_b2p4
        out = self.block6(out)

        # Prune the voxels at res 1/4
        out_cls6 = self.block6_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls6)
        keep6 = (out_cls6.F > 0).squeeze()
        if self.training:
            keep6 += target
        out = self.pruning(out, keep6)
        after_prune.append(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        # out = me.cat(out, out_b1p2)
        out = out + out_b1p2
        out = self.block7(out)

        # Prune the voxels at res 1/2
        out_cls7 = self.block7_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls7)
        keep7 = (out_cls7.F > 0).squeeze()
        if self.training:
            keep7 += target
        out = self.pruning(out, keep7)
        after_prune.append(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        # out = me.cat(out, out_p1)
        out = out + out_p1
        out = self.block8(out)

        # Prune the voxels at orig
        out_cls8 = self.block8_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls8)
        keep8 = (out_cls8.F > 0).squeeze()
        if self.training:
            keep8 += target
        out = self.pruning(out, keep8)
        after_prune.append(out)

        out = self.final(out)

        return out, targets, out_cls, after_prune


class GenerativeRes16UNetCatBase(ResNetBase):
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
        super().__init__(in_channels, out_channels, config, D)

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
        x: Union[torch.Tensor, ME.SparseTensor],
        gt_coords: torch.Tensor,
    ) -> Tuple[ME.SparseTensor, List[torch.Tensor], List[ME.SparseTensor]]:
        """
        Args:
            x (Union[torch.Tensor, ME.SparseTensor]):
                dense feature volume (B, C, H, W, D) or sparse tensor
            gt_coords (torch.Tensor):
                dense volume (B, H, W, D) with 0 or 1
                sparse coordinates (N, 4) for all the occupied voxels
        Returns:
            Output feature: ME.SparseTensor
            GT occupancy at different levels: List[torch.Tensor]
            Predicted occupancy at different levels: List[ME.SparseTensor]
        """

        if isinstance(x, torch.Tensor):
            # Convert dense tensor to sparse tensor
            x = dense_to_sparse(x, stride=1, device=gt_coords.device)

        if gt_coords.ndim == 4:
            # (B, H, W, D) -> (N, 4) where > 0
            gt_coords = torch.stack(
                torch.where(gt_coords > 0), dim=1
            ).int()
        assert gt_coords.ndim == 2

        target_key, _ = x.coordinate_manager.insert_and_map(
            gt_coords,
            string_id="target",
        )

        targets = []
        out_cls = []
        after_prune = []
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
        out = self.concat(out, out_b3p8)
        out = self.block5(out)

        # Prune the voxels at res 1/8
        out_cls5 = self.block5_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls5)
        keep5 = (out_cls5.F > 0).squeeze()
        if self.training:
            keep5 += target

        out = self.pruning(out, keep5)
        after_prune.append(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = self.concat(out, out_b2p4)
        out = self.block6(out)

        # Prune the voxels at res 1/4
        out_cls6 = self.block6_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls6)
        keep6 = (out_cls6.F > 0).squeeze()
        if self.training:
            keep6 += target
        out = self.pruning(out, keep6)
        after_prune.append(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = self.concat(out, out_b1p2)
        out = self.block7(out)

        # Prune the voxels at res 1/2
        out_cls7 = self.block7_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls7)
        keep7 = (out_cls7.F > 0).squeeze()
        if self.training:
            keep7 += target
        out = self.pruning(out, keep7)
        after_prune.append(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = self.concat(out, out_p1)
        out = self.block8(out)

        # Prune the voxels at orig
        out_cls8 = self.block8_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls8)
        keep8 = (out_cls8.F > 0).squeeze()
        if self.training:
            keep8 += target
        out = self.pruning(out, keep8)
        after_prune.append(out)

        out = self.final(out)

        return out, targets, out_cls, after_prune


def get_norm_dense(norm_type, channels, bn_momentum, num_groups=32):
    if norm_type == NormType.BATCH_NORM:
        return nn.BatchNorm3d(channels, momentum=bn_momentum, affine=True)
    elif norm_type == NormType.INSTANCE_NORM:
        return nn.InstanceNorm3d(channels, affine=True)
    elif norm_type == NormType.GROUP_NORM:
        return nn.GroupNorm(num_groups, channels)
    else:
        raise ValueError(f"Invalid norm type: {norm_type}")


class DenseBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        norm_type=NormType.BATCH_NORM,
        # conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = get_norm_dense(norm_type, planes, bn_momentum=bn_momentum)

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )

        self.norm2 = get_norm_dense(norm_type, planes, bn_momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class D2SRes16UNet34Base(ResNetBase):
    """
    Use dense 3D convolutions as the encoder and sparse 3D convolutions as the decoder
    """
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
        super().__init__(in_channels, out_channels, config, D)

    def _make_layer_dense(
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
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                get_norm_dense(norm_type, planes * block.expansion, bn_momentum=bn_momentum),
            )
        layers = []
        layers.append(
            # block(
            #     self.inplanes,
            #     planes,
            #     stride=stride,
            #     dilation=dilation,
            #     downsample=downsample,
            #     conv_type=self.CONV_TYPE,
            #     D=self.D)
            DenseBasicBlock(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                # block(
                #     self.inplanes,
                #     planes,
                #     stride=1,
                #     dilation=dilation,
                #     conv_type=self.CONV_TYPE,
                #     D=self.D)
                DenseBasicBlock(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    norm_type=norm_type,
                    bn_momentum=bn_momentum,
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
        self.conv0p1s1 = nn.Conv3d(
            in_channels,
            self.inplanes,
            kernel_size=config.conv1_kernel_size,
            stride=1,
            dilation=1,
            padding=1,
            bias=False,
        )
        # self.conv0p1s1 = conv(
        #     in_channels,
        #     self.inplanes,
        #     kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
        #     stride=1,
        #     dilation=1,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)

        self.bn0 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)

        # self.conv1p1s2 = conv(
        #     self.inplanes,
        #     self.inplanes,
        #     kernel_size=space_n_time_m(2, 1),
        #     stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)
        self.conv1p1s2 = nn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        self.bn1 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)

        self.block1 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        # self.conv2p2s2 = conv(
        #     self.inplanes,
        #     self.inplanes,
        #     kernel_size=space_n_time_m(2, 1),
        #     stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)
        self.conv2p2s2 = nn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        self.bn2 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block2 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        # self.conv3p4s2 = conv(
        #     self.inplanes,
        #     self.inplanes,
        #     kernel_size=space_n_time_m(2, 1),
        #     stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)
        self.conv3p4s2 = nn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )

        self.bn3 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block3 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        # self.conv4p8s2 = conv(
        #     self.inplanes,
        #     self.inplanes,
        #     kernel_size=space_n_time_m(2, 1),
        #     stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)
        self.conv4p8s2 = nn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )

        self.bn4 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block4 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

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
        self.dense_relu = nn.ReLU(inplace=True)

        self.pruning = ME.MinkowskiPruning()

    def concat(self, x_sparse: ME.SparseTensor, y_dense: torch.Tensor) -> ME.SparseTensor:
        """Hacky way to concatenate two sparse tensors:
            1. Expand the feature dimension of x and y to match each other
            2. Add the two tensors
        """
        # x_feat = x_sparse.features

        # Query y_features using the coordinates of x_sparse
        x_coords = x_sparse.coordinates.clone()   # (N, 4) which has the order (batch, x, y, z)
        stride = torch.tensor(x_sparse.tensor_stride, dtype=int, device=x_coords.device)
        x_coords[:, 1:] = x_coords[:, 1:] // stride
        y_feat = y_dense[x_coords[:, 0], :, x_coords[:, 1], x_coords[:, 2], x_coords[:, 3]]

        y_sparse = ME.SparseTensor(
            y_feat,
            coordinate_map_key=x_sparse.coordinate_map_key,
            coordinate_manager=x_sparse.coordinate_manager,
        )
        return ME.cat(x_sparse, y_sparse)

    def forward(
        self,
        x: torch.Tensor,
        gt_coords: torch.Tensor,
    ) -> Tuple[ME.SparseTensor, List[torch.Tensor], List[ME.SparseTensor]]:
        """
        Args:
            x (torch.Tensor): dense feature volume (B, C, H, W, D)
            gt_coords (torch.Tensor):
                dense volume (B, H, W, D) with 0 or 1
                sparse coordinates (N, 4) for all the occupied voxels
        Returns:
            Output feature: ME.SparseTensor
            GT occupancy at different levels: List[torch.Tensor]
            Predicted occupancy at different levels: List[ME.SparseTensor]
        """
        if gt_coords.ndim == 4:
            # (B, H, W, D) -> (N, 4) where > 0
            gt_coords = torch.stack(
                torch.where(gt_coords > 0), dim=1
            ).int()
        assert gt_coords.ndim == 2, "gt_coords must be (N, 4) where N is the number of occupied voxels"

        targets = []
        out_cls = []
        after_prune = []
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.dense_relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.dense_relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.dense_relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.dense_relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.dense_relu(out)
        out = self.block4(out)

        # Dense-to-sparse
        out = dense_to_sparse(out, stride=2**4, device=gt_coords.device)
        target_key, _ = out.coordinate_manager.insert_and_map(
            gt_coords.int(),
            string_id="target",
        )

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = self.concat(out, out_b3p8)
        out = self.block5(out)

        # Prune the voxels at res 1/8
        out_cls5 = self.block5_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls5)
        keep5 = (out_cls5.F > 0).squeeze()
        if self.training:
            keep5 += target
        else:
            assert keep5.sum() > 0, out_cls5.F.min()
        out = self.pruning(out, keep5)
        after_prune.append(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = self.concat(out, out_b2p4)
        out = self.block6(out)

        # Prune the voxels at res 1/4
        out_cls6 = self.block6_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls6)
        keep6 = (out_cls6.F > 0).squeeze()
        if self.training:
            keep6 += target
        else:
            assert keep6.sum() > 0, out_cls6.F.min()
        out = self.pruning(out, keep6)
        after_prune.append(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = self.concat(out, out_b1p2)
        out = self.block7(out)

        # Prune the voxels at res 1/2
        out_cls7 = self.block7_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls7)
        keep7 = (out_cls7.F > 0).squeeze()
        if self.training:
            keep7 += target
        else:
            assert keep7.sum() > 0, out_cls7.F.min()
        out = self.pruning(out, keep7)
        after_prune.append(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = self.concat(out, out_p1)
        out = self.block8(out)

        # Prune the voxels at orig
        out_cls8 = self.block8_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls8)
        keep8 = (out_cls8.F > 0).squeeze()
        if self.training:
            keep8 += target
        else:
            assert keep8.sum() > 0, out_cls8.F.min()
        out = self.pruning(out, keep8)
        after_prune.append(out)

        out = self.final(out)

        return out, targets, out_cls, after_prune


class GenerativeRes16UNet34(GenerativeRes16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 128, 256, 128, 64, 32, 32)
    # dwn_r =( 2,  4,  8,   16,   8,  4,   2,  1)


class GenerativeRes16UNetCat34(GenerativeRes16UNetCatBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 128, 256, 128, 128, 96, 96)
    # dwn_r =( 2,  4,  8,   16,   8,  4,   2,  1)


class D2SRes16UNet34(D2SRes16UNet34Base):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 128, 256, 128, 128, 96, 96)
    # dwn_r =( 2,  4,  8,   16,   8,  4,   2,  1)


class D2SRes16UNet34Up(D2SRes16UNet34Base):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 128, 256, 128, 128, 96, 96)

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super().__init__(in_channels, out_channels, config, D)

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
        self.conv0p1s1 = nn.Conv3d(
            in_channels,
            self.inplanes,
            kernel_size=config.conv1_kernel_size,
            stride=1,
            dilation=1,
            padding=1,
            bias=False,
        )
        self.bn0 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)

        self.conv1p1s2 = nn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        self.bn1 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)

        self.block1 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv2p2s2 = nn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        self.bn2 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block2 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv3p4s2 = nn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )

        self.bn3 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block3 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv4p8s2 = nn.Conv3d(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )

        self.bn4 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block4 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

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
        self.dense_relu = nn.ReLU(inplace=True)

        self.pruning = ME.MinkowskiPruning()
        bn_momentum = config.bn_momentum
        self.convtr8p2s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.PLANES[7],
            out_channels=self.PLANES[7],
            kernel_size=2,
            stride=2,
            dimension=D,
            bias=False,
        )
        self.bntr8 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

        self.block9 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=1,
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum
        )
        self.block9_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[7],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

    def forward(
        self,
        x: torch.Tensor,
        gt_coords: torch.Tensor,
    ) -> Tuple[ME.SparseTensor, List[torch.Tensor], List[ME.SparseTensor]]:
        """
        Args:
            x (torch.Tensor): dense feature volume (B, C, H, W, D)
            gt_coords (torch.Tensor):
                dense volume (B, H, W, D) with 0 or 1
                sparse coordinates (N, 4) for all the occupied voxels
        Returns:
            Output feature: ME.SparseTensor
            GT occupancy at different levels: List[torch.Tensor]
            Predicted occupancy at different levels: List[ME.SparseTensor]
        """
        if gt_coords.ndim == 4:
            # (B, H, W, D) -> (N, 4) where > 0
            gt_coords = torch.stack(
                torch.where(gt_coords > 0), dim=1
            ).int()
        assert gt_coords.ndim == 2

        targets = []
        out_cls = []
        after_prune = []
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.dense_relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.dense_relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.dense_relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.dense_relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.dense_relu(out)
        out = self.block4(out)

        # Dense-to-sparse
        out = dense_to_sparse(out, stride=2**5, device=gt_coords.device)
        target_key, _ = out.coordinate_manager.insert_and_map(
            gt_coords.int(),
            string_id="target",
        )

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = self.concat(out, out_b3p8)
        out = self.block5(out)

        # Prune the voxels at res 1/8
        out_cls5 = self.block5_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls5)
        keep5 = (out_cls5.F > 0).squeeze()
        if self.training:
            keep5 += target
        else:
            assert keep5.sum() > 0, out_cls5.F.min()
        out = self.pruning(out, keep5)
        after_prune.append(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = self.concat(out, out_b2p4)
        out = self.block6(out)

        # Prune the voxels at res 1/4
        out_cls6 = self.block6_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls6)
        keep6 = (out_cls6.F > 0).squeeze()
        if self.training:
            keep6 += target
        else:
            assert keep6.sum() > 0, out_cls6.F.min()
        out = self.pruning(out, keep6)
        after_prune.append(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = self.concat(out, out_b1p2)
        out = self.block7(out)

        # Prune the voxels at res 1/2
        out_cls7 = self.block7_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls7)
        keep7 = (out_cls7.F > 0).squeeze()
        if self.training:
            keep7 += target
        else:
            assert keep7.sum() > 0, out_cls7.F.min()
        out = self.pruning(out, keep7)
        after_prune.append(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = self.concat(out, out_p1)
        out = self.block8(out)

        # Prune the voxels at orig
        out_cls8 = self.block8_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls8)
        keep8 = (out_cls8.F > 0).squeeze()
        if self.training:
            keep8 += target
        else:
            assert keep8.sum() > 0, out_cls8.F.min()
        out = self.pruning(out, keep8)
        after_prune.append(out)

        # orig -> 2x
        # up_stride 2, dilation 1
        out = self.convtr8p2s2(out)
        out = self.bntr8(out)
        out = self.relu(out)

        out = self.block9(out)
        out_cls9 = self.block9_cls(out)

        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls9)
        keep9 = (out_cls9.F > 0).squeeze()
        if self.training:
            keep9 += target
        else:
            assert keep9.sum() > 0, out_cls9.F.min()
        out = self.pruning(out, keep9)
        after_prune.append(out)

        out = self.final(out)
        return out, targets, out_cls, after_prune


class D2SRes4UNet16(D2SRes16UNet34Base):
    """
    Use dense 3D convolutions as the encoder and sparse 3D convolutions as the decoder
    """
    BLOCK = BasicBlock
    LAYERS = (0, 2, 4, 3, 2, 1)
    # PLANES = (128, 256, 512, 256, 128, 128)
    PLANES = (128, 256, 256, 256, 128, 128)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 128
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super().__init__(in_channels, out_channels, config, D)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.PLANES[0]
        self.conv0p1s1 = nn.Conv3d(
            in_channels,
            self.inplanes,
            kernel_size=config.conv1_kernel_size,
            stride=1,
            dilation=1,
            padding=1,
            bias=False,
        )
        self.bn0 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)

        self.conv1p1s2 = nn.Conv3d(
            self.inplanes,
            self.PLANES[1],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )

        self.inplanes = self.PLANES[1]
        self.bn1 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block1 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv2p2s2 = nn.Conv3d(
            self.inplanes,
            self.PLANES[2],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        self.inplanes = self.PLANES[2]
        self.bn2 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block2 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        # self.conv3p4s2 = conv(
        #     self.inplanes,
        #     self.inplanes,
        #     kernel_size=space_n_time_m(2, 1),
        #     stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)

        self.convtr3p4s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.inplanes,
            out_channels=self.PLANES[3],
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dimension=D,
            bias=False,
        )
        self.bntr3 = get_norm(self.NORM_TYPE, self.PLANES[3], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[3] + self.PLANES[1] * self.BLOCK.expansion
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block3_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[3],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        self.convtr4p8s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.inplanes,
            out_channels=self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dimension=D,
            bias=False,
        )
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)
        self.inplanes = self.PLANES[4] + self.PLANES[0] * self.BLOCK.expansion
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block4_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[4],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum
        )
        self.final = conv(self.PLANES[4], out_channels, kernel_size=1, stride=1, bias=True, D=D)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.dense_relu = nn.ReLU(inplace=True)

        self.pruning = ME.MinkowskiPruning()

    def forward(
        self,
        x: torch.Tensor,
        gt_coords: torch.Tensor,
    ) -> Tuple[ME.SparseTensor, List[torch.Tensor], List[ME.SparseTensor]]:
        """
        Args:
            x (torch.Tensor): dense feature volume (B, C, H, W, D)
            gt_coords (torch.Tensor):
                dense volume (B, H, W, D) with 0 or 1
                sparse coordinates (N, 4) for all the occupied voxels
        Returns:
            Output feature: ME.SparseTensor
            GT occupancy at different levels: List[torch.Tensor]
            Predicted occupancy at different levels: List[ME.SparseTensor]
        """
        if gt_coords.ndim == 4:
            # (B, H, W, D) -> (N, 4) where > 0
            gt_coords = torch.stack(
                torch.where(gt_coords > 0), dim=1
            ).int()
        assert gt_coords.ndim == 2

        targets = []
        out_cls = []
        after_prune = []
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.dense_relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.dense_relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.dense_relu(out)
        out = self.block2(out)

        # Dense-to-sparse
        out = dense_to_sparse(out, stride=2**2, device=gt_coords.device)
        target_key, _ = out.coordinate_manager.insert_and_map(
            gt_coords.int(),
            string_id="target",
        )

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr3p4s2(out)
        out = self.bntr3(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = self.concat(out, out_b1p2)
        out = self.block3(out)

        # Prune the voxels at res 1/2
        out_cls3 = self.block3_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls3)
        keep3 = (out_cls3.F > 0).squeeze()
        if self.training:
            keep3 += target
        else:
            assert keep3.sum() > 0, out_cls3.F.min()
        out = self.pruning(out, keep3)
        after_prune.append(out)

        # 1/2 resolution -> 1
        # up_stride 2, dilation 1
        out = self.convtr4p8s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/1
        out = self.concat(out, out_p1)
        out = self.block4(out)

        # Prune the voxels at res 1/4
        out_cls4 = self.block4_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls4)
        keep4 = (out_cls4.F > 0).squeeze()
        if self.training:
            keep4 += target
        else:
            assert keep4.sum() > 0, out_cls4.F.min()
        out = self.pruning(out, keep4)
        after_prune.append(out)

        out = self.block5(out)

        out = self.final(out)
        return out, targets, out_cls, after_prune


class D2SRes8UNet(D2SRes16UNet34Base):
    BLOCK = BasicBlock
    LAYERS = (0, 2, 2, 3, 2, 2, 2, 1)
    PLANES = (128, 128, 256, 256, 256, 128, 128, 128)
    # PLANES = (64, 64, 128, 256, 128, 128, 96, 96)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 128
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.PLANES[0]
        self.conv0p1s1 = nn.Conv3d(
            in_channels,
            self.inplanes,
            kernel_size=config.conv1_kernel_size,
            stride=1,
            dilation=1,
            padding=1,
            bias=False,
        )
        self.bn0 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)

        self.conv1p1s2 = nn.Conv3d(
            self.inplanes,
            self.PLANES[1],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )

        self.inplanes = self.PLANES[1]
        self.bn1 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block1 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv2p2s2 = nn.Conv3d(
            self.inplanes,
            self.PLANES[2],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        self.inplanes = self.PLANES[2]
        self.bn2 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block2 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv3p4s2 = nn.Conv3d(
            self.inplanes,
            self.PLANES[3],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
        )
        self.inplanes = self.PLANES[3]
        self.bn3 = get_norm_dense(self.NORM_TYPE, self.inplanes, bn_momentum=bn_momentum)
        self.block3 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        # self.conv3p4s2 = conv(
        #     self.inplanes,
        #     self.inplanes,
        #     kernel_size=space_n_time_m(2, 1),
        #     stride=space_n_time_m(2, 1),
        #     dilation=1,
        #     conv_type=self.NON_BLOCK_CONV_TYPE,
        #     D=D)

        # 1/8 resolution -> 1/4
        self.convtr4p4s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=self.inplanes,
            out_channels=self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dimension=D,
            bias=False,
        )
        self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block4_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[4],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        # 1/4 resolution -> 1/2
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
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block5_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[5],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        # 1/2 resolution -> 1
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
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.block6_cls = ME.MinkowskiConvolution(
            in_channels=self.PLANES[6],
            out_channels=1,
            kernel_size=1,
            bias=True,
            dimension=D,
        )

        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum
        )
        self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.dense_relu = nn.ReLU(inplace=True)

        self.pruning = ME.MinkowskiPruning()

    def forward(
        self,
        x: torch.Tensor,
        gt_coords: torch.Tensor,
    ) -> Tuple[ME.SparseTensor, List[torch.Tensor], List[ME.SparseTensor]]:
        """
        Args:
            x (torch.Tensor): dense feature volume (B, C, H, W, D)
            gt_coords (torch.Tensor):
                dense volume (B, H, W, D) with 0 or 1
                sparse coordinates (N, 4) for all the occupied voxels
        Returns:
            Output feature: ME.SparseTensor
            GT occupancy at different levels: List[torch.Tensor]
            Predicted occupancy at different levels: List[ME.SparseTensor]
        """
        if gt_coords.ndim == 4:
            # (B, H, W, D) -> (N, 4) where > 0
            gt_coords = torch.stack(
                torch.where(gt_coords > 0), dim=1
            ).int()
        assert gt_coords.ndim == 2

        targets = []
        out_cls = []
        after_prune = []
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.dense_relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.dense_relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.dense_relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.dense_relu(out)
        out = self.block3(out)

        # Dense-to-sparse
        out = dense_to_sparse(out, stride=2**3, device=gt_coords.device)
        target_key, _ = out.coordinate_manager.insert_and_map(
            gt_coords.int(),
            string_id="target",
        )

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr4p4s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = self.concat(out, out_b2p4)
        out = self.block4(out)

        # Prune the voxels at res 1/4
        out_cls4 = self.block4_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls4)
        keep4 = (out_cls4.F > 0).squeeze()
        if self.training:
            keep4 += target
        else:
            assert keep4.sum() > 0, out_cls4.F.min()
        out = self.pruning(out, keep4)
        after_prune.append(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = self.concat(out, out_b1p2)
        out = self.block5(out)

        # Prune the voxels at res 1/2
        out_cls5 = self.block5_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls5)
        keep5 = (out_cls5.F > 0).squeeze()
        if self.training:
            keep5 += target
        else:
            assert keep5.sum() > 0, out_cls5.F.min()

        out = self.pruning(out, keep5)
        after_prune.append(out)

        # 1/2 resolution -> 1
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with orig
        out = self.concat(out, out_p1)
        out = self.block6(out)

        # Prune the voxels at orig
        out_cls6 = self.block6_cls(out)
        target = get_target(out, target_key)
        targets.append(target)
        out_cls.append(out_cls6)
        keep6 = (out_cls6.F > 0).squeeze()
        if self.training:
            keep6 += target
        else:
            assert keep6.sum() > 0, out_cls6.F.min()
        out = self.pruning(out, keep6)
        after_prune.append(out)

        out = self.block7(out)
        out = self.final(out)
        return out, targets, out_cls, after_prune


if __name__ == "__main__":
    from models.encoder.minkowskinet.resunet import ResUNetConfig
    # model = D2SRes4UNet16(in_channels=64, out_channels=3, config=ResUNetConfig(), D=3)
    model = D2SRes8UNet(in_channels=64, out_channels=3, config=ResUNetConfig(), D=3)
    print(model)

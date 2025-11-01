from typing import Union, Optional, List, Tuple, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.minkowskinet.resunet import ResNetBase, NormType, ConvType, conv, get_norm, BasicBlock
from models.minkowskinet.dense_to_sparse import dense_to_sparse
from models.time_embedding import get_timestep_embedding
from models.unet_base import (
    get_target,
    GenerativeRes16UNetCatBase,
    ResUNetConfig,
)


class DenseResBlock3DWithTime(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        time_emb_dim: int = 0,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(input_dim, output_dim, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(output_dim)
        self.conv2 = nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.time_emb_dim = time_emb_dim
        if time_emb_dim > 0:
            self.time_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(time_emb_dim, output_dim),
            )

        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(input_dim, output_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(output_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb=None):
        residual = self.shortcut(x)

        out = self.conv1(x)
        if self.time_emb_dim > 0:
            assert time_emb is not None
            out = out + self.time_proj(time_emb)[None, :, None, None, None]

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


# class DenseUNet(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         num_blocks: int,
#         time_emb_dim: int = 0,
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.relu = nn.ReLU(inplace=True)
#         self.conv1s2 = nn.Conv3d(
#             input_dim,
#             int(input_dim * 1.5),
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             bias=False,
#         )
#         self.bn1 = nn.BatchNorm3d(int(input_dim * 1.5))
#         self.conv2s2 = nn.Conv3d(
#             int(input_dim * 1.5),
#             input_dim * 2,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             bias=False,
#         )
#         self.bn2 = nn.BatchNorm3d(int(input_dim * 2))

#         self.blocks = nn.ModuleList()
#         for i in range(num_blocks):
#             self.blocks.append(DenseResBlock3DWithTime(input_dim * 2, input_dim * 2, bias=False, time_emb_dim=time_emb_dim))

#         self.conv3tr2 = nn.ConvTranspose3d(
#             input_dim * 2 + input_dim * 2,  # skip + out
#             input_dim * 2,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             output_padding=1,
#             bias=False,
#         )
#         self.bn3 = nn.BatchNorm3d(int(input_dim * 2))
#         self.conv4tr2 = nn.ConvTranspose3d(
#             input_dim * 2 + int(input_dim * 1.5),
#             self.output_dim,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             output_padding=1,
#             bias=False,
#         )
#         self.bn4 = nn.BatchNorm3d(self.output_dim)
#         self.final = nn.Conv3d(self.output_dim, self.output_dim, kernel_size=1)

#     def forward(self, x, time_emb=None):
#         print("x", x.shape)
#         x1 = self.relu(self.bn1(self.conv1s2(x)))
#         print("x1", x1.shape)
#         x2 = self.relu(self.bn2(self.conv2s2(x1)))
#         print("x2", x2.shape)

#         x = x2
#         for block in self.blocks:
#             x = block(x, time_emb=time_emb)
#         print(x.shape)

#         x = self.relu(self.bn3(self.conv3tr2(torch.cat([x, x2], dim=1))))
#         print("x up1", x.shape)
#         x = self.relu(self.bn4(self.conv4tr2(torch.cat([x, x1], dim=1))))
#         print("x up2", x.shape)
#         x = self.final(x)
#         return x


class DenseUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int,
        time_emb_dim: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU(inplace=True)
        self.conv1s2 = nn.Conv3d(
            input_dim,
            input_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(input_dim * 2)

        self.blocks1 = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks1.append(DenseResBlock3DWithTime(input_dim * 2, input_dim * 2, bias=False, time_emb_dim=time_emb_dim))

        self.conv2tr2 = nn.ConvTranspose3d(
            input_dim * 2 + input_dim * 2,  # skip + out
            input_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(input_dim * 2)
        self.block2 = DenseResBlock3DWithTime(input_dim * 2, input_dim, bias=False, time_emb_dim=time_emb_dim)
        self.final = nn.Conv3d(input_dim, output_dim, kernel_size=1)

    def forward(self, x, time_emb=None):
        x1 = self.relu(self.bn1(self.conv1s2(x)))

        x = x1
        for block in self.blocks1:
            x = block(x, time_emb=time_emb)

        x = self.relu(self.bn2(self.conv2tr2(torch.cat([x, x1], dim=1))))
        x = self.block2(x, time_emb=time_emb)
        x = self.final(x)
        return x


class Initializer(GenerativeRes16UNetCatBase):
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

            self.dense_unet = DenseUNet(
                self.PLANES[3],
                self.PLANES[3],
                self.num_dense_blocks,
            )

            self.block4_cls = ME.MinkowskiConvolution(
                in_channels=self.PLANES[3],
                out_channels=1,
                kernel_size=1,
                bias=True,
                dimension=D,
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

    def forward(
        self,
        x: ME.SparseTensor,
        gt_coords: torch.Tensor,
        ignore_coords: Optional[torch.Tensor] = None,
        known_coords: Optional[torch.Tensor] = None,
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

        def forward_block(block, input, time_emb):
            for layer in block:
                input = layer(input, time_emb)
            return input

        if self.use_time_emb:
            assert timestamp is not None
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

        if known_coords is not None:
            if max_gt_samples_last is not None and max_gt_samples_last < known_coords.shape[0]:
                indices = torch.randperm(known_coords.shape[0])[:max_gt_samples_last]
                known_coords = known_coords[indices]

            known_key, _ = x.coordinate_manager.insert_and_map(
                known_coords,
                string_id="known",
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
        out_b4p16 = forward_block(self.block4, out, time_emb)

        ######## From sparse to dense (bottleneck) ########
        if self.dense_bottleneck:
            dense_tensor, min_coordinate, stride = out_b4p16.dense(min_coordinate=torch.IntTensor([0, 0, 0]), contract_stride=True)

            # Pad dense_tensor if the shape is not divisible by 4
            # dense_tensor.shape = (B, C, D, H, W)
            pad = [0, 0, 0]
            for i in range(3):
                pad[i] = 4 - dense_tensor.shape[i + 2] % 4

            if sum(pad) > 0:
                dense_tensor = F.pad(dense_tensor, (0, pad[2], 0, pad[1], 0, pad[0]), mode="constant", value=0)

            dense_tensor = self.dense_unet(dense_tensor, time_emb)

            out = dense_to_sparse(
                dense_tensor,
                stride=out_b4p16.tensor_stride[0],
                device=dense_tensor.device,
                coordinate_manager=out_b4p16.coordinate_manager,
            )
            # print("Before denify", out.shape, "After densify", out1.shape)
            out_cls4 = self.block4_cls(out)
            target, _, _, xyz_pred = get_target(out, target_key)
            targets.append(target)
            out_cls.append(out_cls4)
            xyz_preds.append(xyz_pred)

            if ignore_coords is not None:
                ignore, _, _, _ = get_target(out, ignore_key)
                ignores.append(ignore)

            keep4 = (out_cls4.F > threshold).squeeze()

            if self.training:
                if known_coords is not None:
                    known, _, _, _ = get_target(out, known_key)
                    keep4 += known

                if max_gt_samples_last is not None:
                    target_sub, _, _, _ = get_target(out, subsample_key)
                    if verbose:
                        print(f"Layer 4: {target_sub.sum()} samples from subsample, {target.sum()} samples from target, {keep4.sum()} thresholded, total {out.F.shape[0]}")
                    keep4 += target_sub
                else:
                    # keep4 = target      # DEBUG
                    keep4 += target
            if verbose:
                print(f"Layer 4: keep4 {keep4.sum()} thresholded, total {out.F.shape[0]}")

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
            if known_coords is not None:
                known, _, _, _ = get_target(out, known_key)
                keep5 += known

            if max_gt_samples_last is not None:
                target_sub, _, _, _ = get_target(out, subsample_key)
                if verbose:
                    print(f"Layer 5: {target_sub.sum()} samples from subsample, {target.sum()} samples from target, {keep5.sum()} thresholded, total {out.F.shape[0]}")
                keep5 += target_sub
            else:
                # keep = target      # DEBUG
                keep5 += target

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
            if known_coords is not None:
                known, _, _, _ = get_target(out, known_key)
                keep6 += known
                if verbose:
                    print(f"Layer 7: add known {known.sum()} samples, total {keep6.sum()}")

            if max_gt_samples_last is not None:
                target_sub, _, _, _ = get_target(out, subsample_key)
                if verbose:
                    print(f"Layer 6: {target_sub.sum()} samples from subsample, {target.sum()} samples from target, {keep6.sum()} thresholded, total {out.F.shape[0]}")
                keep6 += target_sub
            else:
                # keep6 = target      # DEBUG
                keep6 += target
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
        # ==============================
        # In big scenes, keep7 will generate too many samples after the next upsampling
        # To prevent GPU OOM, we can either:
        # Always subsample same numbers, not sure how much worse will in this case
        # Or simipling using higher thresholding for this level
        if threshold_second_last is not None:
            assert threshold_second_last >= 0 and threshold_second_last <= 1, f"Threshold_second_last should be in [0, 1], got {threshold_second_last}"
            prob7 = torch.sigmoid(out_cls7.F).squeeze()
            keep7 = (prob7 > threshold_second_last)
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
            if known_coords is not None:
                known, _, _, _ = get_target(out, known_key)
                keep7 += known
                if verbose:
                    print(f"Layer 7: add known {known.sum()} samples, total {keep7.sum()}")

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

        if self.training and not sample_last_strict:
            if max_gt_samples_last is not None:
                target_sub, _, _, _ = get_target(out, subsample_key)
                if verbose:
                    print(f"Layer 8: {target_sub.sum()} samples from subsample, {target.sum()} samples from target, {keep8.sum()} thresholded, total {out.F.shape[0]}")
                keep8 += target_sub
            else:
                keep8 += target

        # prob_last = out_cls8.F[keep8]

        if verbose:
            print(f"Layer 8: keep8 {keep8.sum()} thresholded, total {out.F.shape[0]}")
        out = self.pruning(out, keep8)
        prob_last = self.pruning(out_cls8, keep8)

        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 8 (orig resolution). It has {keep8.shape} voxels before")
        # print("After pruning", out.F.shape)
        after_prune.append(out)

        out = self.final(out)
        _, indices_pred, indices_gt, _ = get_target(out, target_key)

        if ignore_coords is not None:
            ignore_final, _, _, _ = get_target(out, ignore_key)
        else:
            ignore_final = None

        return {
            "out": out,
            "targets": targets,
            "ignores": ignores,
            "out_cls": out_cls,
            "indices_pred": indices_pred,
            "indices_gt": indices_gt,
            "ignore_final": ignore_final,
            "xyz_preds": xyz_preds,
            "last_prob": prob_last,     # it's the probability logits of the last layer (before sigmoid)
        }


class Initializer18(Initializer):
    INIT_DIM = 64
    LAYERS = (2, 3, 3, 3, 2, 2, 2, 2)
    PLANES = (64, 96, 128, 128, 128, 96, 64, 64)


class Initializer34(Initializer):
    INIT_DIM = 64
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 96, 128, 128, 128, 96, 64, 64)

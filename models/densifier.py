from typing import Union, Optional, List, Tuple, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.minkowskinet.resunet import ResNetBase, NormType, ConvType, conv, get_norm, BasicBlock
from models.minkowskinet.dense_to_sparse import dense_to_sparse
from models.time_embedding import get_timestep_embedding
from models.unet_base import ResUNetConfig, GenerativeRes16UNetCatBase, get_target, forward_block
from modules.mlp import MLP
from models.scaffold_gs import inverse_sigmoid


class DensifierUNet(GenerativeRes16UNetCatBase):
    def network_initialization(self, in_channels, out_channels, config, D):
        super().network_initialization(in_channels, out_channels, config, D)
        # self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.inplanes = self.PLANES[7]
        self.block9 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=self.DILATIONS[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=config.bn_momentum,
        )

    def forward(
        self,
        x: ME.SparseTensor,
        timestamp: Optional[int] = None,
        gt_coords: Optional[torch.Tensor] = None,
        anchor_coords: Optional[torch.Tensor] = None,
        ignore_coords: Optional[torch.Tensor] = None,
        max_anchor_samples: Optional[int] = None,
        threshold: float = 0.5,
        threshold_second_last: Optional[float] = None,
        threshold_last: Optional[float] = None,
        max_samples_second_last: Optional[int] = None,
        sample_n_last: Optional[int] = None,
        sample_temperature: float = 1.0,
        eval_randomness: bool = False,
        ignore_second_last: bool = False,
        verbose: bool = False,
    ):
        """
        x: input sparse tensor
        timestamp: timestamp for the time embedding. If None, the time embedding is not used
        gt_coords: ground truth coordinates for getting the labels for each level
        ignore_coords: ignoring selecting them as samples at the last level
        anchor_coords: the coordinates that will be insert for each level
        """
        assert threshold >= 0 and threshold <= 1, f"Threshold should be in [0, 1], got {threshold}"
        threshold = inverse_sigmoid(threshold)
        if threshold_second_last is not None:
            assert threshold_second_last >= 0 and threshold_second_last <= 1, f"Threshold_second_last should be in [0, 1], got {threshold_second_last}"
            threshold_second_last = inverse_sigmoid(threshold_second_last)
        else:
            threshold_second_last = threshold

        if threshold_last is not None:
            assert threshold_last >= 0 and threshold_last <= 1, f"Threshold_last should be in [0, 1], got {threshold_last}"
            threshold_last = inverse_sigmoid(threshold_last)
        else:
            threshold_last = threshold

        targets = []
        out_cls = []
        ignores = []

        time_emb, out_p1, out_b1p2, out_b2p4, out_b3p8, out_b4p16 = self.forward_encoder(x, timestamp)

        if self.dense_bottleneck:
            dense_tensor, min_coordinate, stride = out_b4p16.dense(min_coordinate=torch.IntTensor([0, 0, 0]), contract_stride=True)
            dense_tensor = self.dense_blocks(dense_tensor)

            out = dense_to_sparse(
                dense_tensor,
                stride=out_b4p16.tensor_stride[0],
                device=dense_tensor.device,
                coordinate_manager=out_b4p16.coordinate_manager,
            )
        else:
            out = out_b4p16

        if gt_coords is not None:
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
            if max_anchor_samples is not None:
                indices = torch.randperm(anchor_coords.shape[0])[:max_anchor_samples]
                anchor_coords = anchor_coords[indices]

            anchor_key, _ = x.coordinate_manager.insert_and_map(
                anchor_coords,
                string_id="anchor",
            )

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

        if ignore_coords is not None:
            ignore, _, _, _ = get_target(out, ignore_key)
            ignores.append(ignore)

        keep5 = (out_cls5.F > threshold).squeeze()
        if anchor_coords is not None:
            anchor, _, _, _ = get_target(out, anchor_key)
            keep5 += anchor

        if verbose:
            num_keep5 = keep5.sum()
            print(f"Prune layer 5: keep5 {num_keep5} thresholded, total {out.F.shape[0]}, ratio {num_keep5 / out.F.shape[0]:.2f}")

        out = self.pruning(out, keep5)

        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 5 (1/8 resolution). It has {keep5.shape} voxels before")

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

        if ignore_coords is not None:
            ignore, _, _, _ = get_target(out, ignore_key)
            ignores.append(ignore)

        keep6 = (out_cls6.F > threshold).squeeze()
        if anchor_coords is not None:
            anchor, _, _, _ = get_target(out, anchor_key)
            keep6 += anchor

        if verbose:
            num_keep6 = keep6.sum()
            print(f"Prune layer 6: keep6 {num_keep6} thresholded, total {out.F.shape[0]}, ratio {num_keep6 / out.F.shape[0]:.2f}")

        out = self.pruning(out, keep6)
        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 6 (1/4 resolution). It has {keep6.shape} voxels before")

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

        if ignore_coords is not None:
            ignore, _, _, _ = get_target(out, ignore_key)
            ignores.append(ignore)

        # NOTE: Use "threshold_second_last" instead of "threshold"

        logits7 = out_cls7.F.clone().squeeze()

        if ignore_coords is not None and ignore_second_last:
            # ignore_coords are less likely to be sampled
            # TODO: handle the problem when anchor is added back
            ignore, _, _, _ = get_target(out, ignore_key)
            logits7[ignore] = inverse_sigmoid(1e-8)

        keep7 = (logits7 > threshold_second_last)

        if max_samples_second_last is not None:
            with torch.no_grad():
                threshold_indices = keep7.nonzero()

                if threshold_indices.shape[0] > max_samples_second_last:
                    keep7 = torch.zeros_like(out_cls7.F.squeeze(), dtype=torch.bool)

                    if not self.training and not eval_randomness:
                        # During evaluation and not using randomness, we will keep the largest max_samples_second_last indices
                        indices = torch.topk(logits7, max_samples_second_last).indices
                        keep7[indices] = 1
                    else:
                        indices = torch.randperm(threshold_indices.shape[0])[:max_samples_second_last]
                        keep7[threshold_indices[indices]] = 1

        if anchor_coords is not None and not ignore_second_last:
            anchor, _, _, _ = get_target(out, anchor_key)
            keep7 += anchor

        if verbose:
            num_keep7 = keep7.sum()
            print(f"Prune layer 7: keep7 {num_keep7} thresholded, total {out.F.shape[0]}, ratio {num_keep7 / out.F.shape[0]:.2f}")

        out = self.pruning(out, keep7)
        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 7 (1/2 resolution). It has {keep7.shape} voxels before")

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

        bxyz_pred8 = xyz_pred
        targets.append(target)
        out_cls.append(out_cls8)

        if ignore_coords is not None:
            ignore, _, _, _ = get_target(out, ignore_key)
            ignores.append(ignore)

        # NOTE: Use "threshold_last" instead of "threshold"
        keep8 = (out_cls8.F > threshold_last).squeeze()

        if sample_n_last is not None:
            EPS = 1e-8
            # Sample n voxels from the last layer based on the probability with temperature
            with torch.no_grad():
                if sample_n_last > keep8.shape[0]:
                    keep8 = torch.ones_like(out_cls8.F.squeeze(), dtype=torch.bool)

                elif not self.training and not eval_randomness:
                    # During evaluation and not using randomness, we will keep the largest n indices
                    prob8 = torch.sigmoid(out_cls8.F.squeeze())

                    if ignore_coords is not None:
                        # Make sure that ignore points are unlikely to be sampled
                        ignore, _, _, _ = get_target(out, ignore_key)
                        prob8[ignore] = EPS

                    indices = torch.topk(prob8, sample_n_last).indices
                    keep8 = torch.zeros_like(out_cls8.F.squeeze(), dtype=torch.bool)
                    keep8[indices] = 1

                else:
                    prob8 = torch.sigmoid(out_cls8.F / sample_temperature).squeeze()
                    if self.training and anchor_coords is not None:
                        # Insert anchor points
                        anchor, _, _, _ = get_target(out, anchor_key)
                        prob8[anchor] = 1 - EPS

                    if ignore_coords is not None:
                        # Ignore the points
                        ignore, _, _, _ = get_target(out, ignore_key)
                        prob8[ignore] = EPS

                    indices = torch.multinomial(prob8, sample_n_last, replacement=False)
                    keep8 = torch.zeros_like(prob8, dtype=torch.bool)
                    keep8[indices] = 1

        logits_last = out_cls8.F[keep8]

        if verbose:
            num_keep8 = keep8.sum()
            print(f"Prune layer 8: keep8 {num_keep8} thresholded, total {out.F.shape[0]}, ratio {num_keep8 / out.F.shape[0]:.2f}")

        out = self.pruning(out, keep8)
        if out.F.shape[0] == 0:
            raise ValueError(f"Pruning resulted in empty tensor at block 8 (orig resolution). It has {keep8.shape} voxels before")

        # Concat with orig
        # out = self.concat(out, out_p1)
        # out = self.block8(out)
        out = forward_block(self.block9, out, time_emb)
        out = self.final(out)

        if ignore_coords is not None:
            ignore_final, _, _, _ = get_target(out, ignore_key)

        outputs = {
            "out": out,
            "targets": targets,
            "out_cls": out_cls,
            "last_prob": logits_last,
            "bxyz_pred8": bxyz_pred8,
        }
        if ignore_coords is not None:
            outputs["ignores"] = ignores
            outputs["ignore_final"] = ignore_final

        return outputs


class DensifierUNet18(DensifierUNet):
    INIT_DIM = 64
    LAYERS = (2, 3, 3, 3, 2, 2, 2, 2)
    PLANES = (64, 96, 128, 128, 128, 96, 64, 64)


class Densifier(nn.Module):
    def __init__(
        self,
        config,
        in_dim: int,
        out_dim: int,
        occupancy_as_opacity: bool = True,
    ):
        super().__init__()
        self.config = config
        self.occupancy_as_opacity = occupancy_as_opacity

        self.compress = MLP(
            in_dim=in_dim * 2,
            layer_width=(in_dim * 3) // 2,
            num_layers=2,
            out_dim=in_dim,
        )

        if self.occupancy_as_opacity:
            out_dim -= 1

        # dense18 by default
        self.unet = DensifierUNet18(
            in_channels=self.config.MODEL.SCAFFOLD.hidden_dim,
            out_channels=self.config.MODEL.SCAFFOLD.hidden_dim,
            config=ResUNetConfig(),
            use_time_emb=True,
            time_emb_dim=64,
            dense_bottleneck=False,
        )

    def forward(
        self,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        bxyz: torch.Tensor,
        bxyz_gt: torch.Tensor,
        timestamp: Optional[float] = None,
        threshold_last: Optional[float] = None,
        threshold_second_last: Optional[float] = None,
        sample_n_last: Optional[int] = None,
        sample_temperature: float = 1.0,
        max_gt_samples_training: Optional[int] = None,
        max_samples_second_last: Optional[int] = None,
    ):
        x = torch.cat([params["latent"], grads["latent"]], dim=1)
        x = self.compress(x)
        timestamp = torch.tensor([timestamp], device=x.device)
        sparse_tensor = ME.SparseTensor(
            coordinates=bxyz,
            features=x,
        )
        # outputs = self.unet(
        #     sparse_tensor,
        #     timestamp=timestamp,
        #     gt_coords=bxyz_gt,
        #     ignore_coords=bxyz,
        #     anchor_coords=bxyz,
        #     # TODO: max_anchor_samples=max_gt_samples_training,
        #     threshold_second_last=threshold_second_last,
        #     threshold_last=threshold_last,
        #     sample_n_last=sample_n_last,
        #     # Make sure the output of the last layer has exactly "sample_n_last" points
        #     sample_last_strict=True,                            # TODO: remove this
        #     sample_temperature=sample_temperature,
        #     max_samples_second_last=max_samples_second_last,    # TODO: Carefully use this
        #     eval_randomness=True,       # TODO: Try False
        #     max_gt_samples_last=max_gt_samples_training,        # TODO: remove this
        #     verbose=self.config.debug,
        # )
        # TODO
        if self.training:
            bxyz_anchor = torch.cat([bxyz, bxyz_gt], dim=0)
        else:
            bxyz_anchor = bxyz

        # Avoid sampling the original points during densification
        bxyz_ignore = bxyz
        if self.config.MODEL.DENSIFIER.dilate_ignore > 0:
            with torch.no_grad():
                # Better to set unique=True if dilate_ignore > 1
                bxyz_ignore = dilation_sparse(bxyz_ignore, self.config.MODEL.DENSIFIER.dilate_ignore, unique=False)

        outputs = self.unet(
            sparse_tensor,
            timestamp=timestamp,
            gt_coords=bxyz_gt,
            ignore_coords=bxyz_ignore,
            anchor_coords=bxyz_anchor,
            max_anchor_samples=max_gt_samples_training,
            max_samples_second_last=max_samples_second_last,
            sample_n_last=sample_n_last,
            sample_temperature=sample_temperature,
            eval_randomness=self.config.MODEL.DENSIFIER.eval_randomness,
            ignore_second_last=self.config.MODEL.DENSIFIER.ignore_second_last,       # TODO: Test this
            verbose=self.config.debug,
        )

        output_sparse = outputs["out"]

        if self.occupancy_as_opacity:
            # Opacity is the probability of the voxel being occupied of the last pruning layer
            # This enable backpropagation from rendering loss to the SGNN
            # Put prob into the outputs["out"] at 10:11 dimension
            prob = outputs["last_prob"]
            feat_new = torch.cat([
                output_sparse.F[..., :10],
                prob,
                output_sparse.F[..., 10:],
            ], dim=1)
        else:
            feat_new = output_sparse.F

        output_sparse_new = ME.SparseTensor(
            features=feat_new,
            coordinate_map_key=output_sparse.coordinate_map_key,
            coordinate_manager=output_sparse.coordinate_manager,
        )
        outputs["out"] = output_sparse_new
        return outputs

from typing import Union, Optional, List, Tuple

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from .resunet import ResNetBase, NormType, ConvType, conv, get_norm, BasicBlock
from .resunet_sparse import D2SRes8UNet, get_norm_dense


class ResSparseDecoder(D2SRes8UNet):
    LAYERS = (0, 2, 2, 3, 2, 2, 2, 1)
    PLANES = (128, 128, 256, 256, 256, 128, 128, 128)

    def network_initialization(self, in_channels, out_channels, config, D):
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

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
        self.block1 = self._make_layer_dense(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum,
        )


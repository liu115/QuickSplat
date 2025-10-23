from typing import Optional, Dict, Literal

import torch
from torch import nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.unet_base import (
    Res16UNetWithTime,
    ResUNetConfig,
)


class Res16UNet18AWithTime(Res16UNetWithTime):
    INIT_DIM = 64
    PLANES = (64, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18CWithTime(Res16UNetWithTime):
    INIT_DIM = 64
    PLANES = (64, 64, 128, 256, 256, 128, 96, 96)


class Res16UNet34CWithTime(Res16UNetWithTime):
    INIT_DIM = 64
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (64, 64, 128, 256, 256, 128, 96, 96)


class UNetOptimizer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        backbone: Literal["16unetA", "16unetC", "16unet34C"] = "16unetA",
        timestamp_norm: bool = False,       # Deprecated
        output_norm: bool = False,
        grad_residual: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.grad_residual = grad_residual
        if self.grad_residual:
            # output = input_grad * a + b
            out_dim = out_dim * 2

        if self.backbone == "16unetA":
            self.encoder = Res16UNet18AWithTime(
                in_dim,
                out_dim,
                ResUNetConfig(),
                use_time_emb=True,
                time_emb_dim=64,
            )
        elif self.backbone == "16unetC":
            self.encoder = Res16UNet18CWithTime(
                in_dim,
                out_dim,
                ResUNetConfig(),
                use_time_emb=True,
                time_emb_dim=64,
            )
        else:
            self.encoder = Res16UNet34CWithTime(
                in_dim,
                out_dim,
                ResUNetConfig(),
                use_time_emb=True,
                time_emb_dim=64,
            )
        # self.timestamp_norm = timestamp_norm
        self.output_norm = output_norm

    def get_param_groups(self):
        return {
            "model": list(self.encoder.parameters()),
        }

    def forward(
        self,
        bxyz: torch.Tensor,
        gaussian_params: Dict[str, torch.Tensor],
        gaussian_grads: Dict[str, torch.Tensor],
        additional_input: Optional[torch.Tensor] = None,
        timestamp: Optional[float] = None,
        remap_mapping: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        input_tensor = torch.cat([
            gaussian_params["latent"],
            gaussian_grads["latent"],
        ], dim=1)   # (N, hidden_dim * 2)
        if additional_input is not None:
            input_tensor = torch.cat([input_tensor, additional_input], dim=1)

        input_sparse = ME.SparseTensor(
            features=input_tensor,
            coordinates=bxyz.int(),
        )

        time_stamp = torch.tensor([timestamp], device=input_tensor.device)    # (1,)
        output_sparse, _ = self.encoder(input_sparse, time_stamp)
        output = output_sparse.F
        # output.nan_to_num_()

        if self.grad_residual:
            # output = input_grad * a + b
            output = torch.nn.functional.sigmoid(output)
            output_a = output[:, :output.shape[1] // 2]
            output_b = output[:, output.shape[1] // 2:] * 2 - 1
            output = -gaussian_grads["latent"] * output_a + output_b

        elif self.output_norm:
            output = torch.nn.functional.sigmoid(output) * 2 - 1

        out_dict = {}
        out_dict["latent"] = output
        out_dict["bxyz"] = output_sparse.C.clone()

        if input_sparse.C.shape != output_sparse.C.shape:
            print("input_tensor.C.shape", input_tensor.C.shape)
            print("output_sparse.C.shape", output_sparse.C.shape)
            # print("Input tensor and output tensor have different shapes")
            torch.save({
                "input_bxyz": input_tensor.C,
                "output_bxyz": output_sparse.C,
            }, "input_output_bxyz.pt")
            raise ValueError("Input tensor and output tensor have different shapes")
        return out_dict

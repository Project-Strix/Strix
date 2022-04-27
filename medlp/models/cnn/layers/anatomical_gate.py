import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import get_conv_layer


class AnatomicalAttentionGate(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        featmap1_inchn: int,
        featmap2_inchn: int,
        mode: str = "sum",  # 'sum', 'cat', 'weighted', 'mode2', 'mode3'
        act: str = "sigmoid",
    ):
        super().__init__()
        self.mode = mode
        self.spatial_dims = spatial_dims
        assert featmap1_inchn == featmap2_inchn, "channel num must be same!"
        self.conv1 = get_conv_layer(
            spatial_dims,
            featmap1_inchn + featmap2_inchn,
            featmap1_inchn,
            kernel_size=1,
            stride=1,
            act=act,
            norm=None,
            bias=True,
            conv_only=False,
        )

        self.conv2 = get_conv_layer(
            spatial_dims,
            featmap1_inchn + featmap2_inchn,
            featmap2_inchn,
            kernel_size=1,
            stride=1,
            act=act,
            norm=None,
            bias=True,
            conv_only=False,
        )

        if self.mode == "cat":
            self.final_conv = get_conv_layer(
                spatial_dims,
                featmap1_inchn + featmap2_inchn,
                featmap1_inchn,
                kernel_size=1,
                stride=1,
                act="relu",
                norm="batch",
                bias=True,
                conv_only=False,
            )
        elif self.mode == "weighted":
            self.w = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        elif self.mode == "sum":
            self.w = 1
        elif self.mode in ["mode2", "mode3"]:
            pass
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

    def forward(self, x1, x2):
        concat_featmap = torch.cat([x1, x2], dim=1)
        weighted_featmap1 = self.conv1(concat_featmap) * x1
        if self.mode == "mode3":
            dims = list(range(2, 2 + self.spatial_dims))
            weighted_featmap2 = self.conv2(concat_featmap)
            weights = weighted_featmap2.mean(dims, keepdim=True)
            norm_weights = nn.functional.sigmoid(weights)
            return weighted_featmap1 * norm_weights

        weighted_featmap2 = self.conv2(concat_featmap) * x2

        if self.mode == "cat":
            return self.final_conv(
                torch.cat([weighted_featmap1, weighted_featmap2], dim=1)
            )
        elif self.mode == "weighted" or self.mode == "sum":
            return weighted_featmap1 + self.w * weighted_featmap2
        elif self.mode == "mode2":
            dims = list(range(2, 2 + self.spatial_dims))
            weights = weighted_featmap2.mean(dims, keepdim=True)
            norm_weights = nn.functional.sigmoid(weights)
            return weighted_featmap1 * norm_weights


class AnatomicalAttentionGate2(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        featmap1_inchn: int,
        featmap2_inchn: int,
        use_W: bool = True,
        use_W_bn: bool = True,
        mode: str = "concatenation_range_normalise",
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.mode = mode
        assert featmap1_inchn == featmap2_inchn, "channel num must be same!"
        self.conv1 = get_conv_layer(
            spatial_dims,
            featmap1_inchn,
            featmap1_inchn,
            kernel_size=1,
            stride=1,
            act=None,
            norm=None,
            bias=True,
            conv_only=True,
        )

        self.conv2 = get_conv_layer(
            spatial_dims,
            featmap2_inchn,
            featmap2_inchn,
            kernel_size=1,
            stride=1,
            act=None,
            norm=None,
            bias=True,
            conv_only=True,
        )

        self.psi = get_conv_layer(
            spatial_dims, featmap2_inchn, 1, kernel_size=1, stride=1, bias=True
        )
        self.nonlinear = lambda x: nn.functional.relu(x, inplace=True)

        if use_W:
            self.W = get_conv_layer(
                spatial_dims,
                featmap1_inchn,
                featmap1_inchn,
                kernel_size=1,
                stride=1,
                act=None,
                norm="batch" if use_W_bn else None,
                bias=True,
                conv_only=False,
            )
        else:
            self.W = nn.Identity()

    def forward(self, x1, x2):
        theta_x = self.conv1(x1)
        phi_g = self.conv2(x2)

        f = self.nonlinear(theta_x + phi_g)
        psi_f = self.psi(f)

        ############################################
        # normalisation -- scale compatibility score
        #  psi^T . f -> (b, 1, t/s1, h/s2, w/s3)
        dims = list(range(2, 2 + self.spatial_dims))
        if self.mode == "concatenation_mean_flow":
            psi_f_min = psi_f.amin(dims, keepdim=True)
            psi_f_sum = psi_f.sum(dims, keepdim=True)
            sigm_psi_f = (psi_f - psi_f_min) / psi_f_sum
        elif self.mode == "concatenation_range_normalise":
            psi_f_min = psi_f.amin(dims, keepdim=True)
            psi_f_max = psi_f.amax(dims, keepdim=True)
            sigm_psi_f = (psi_f - psi_f_min) / (psi_f_max - psi_f_min)
        elif self.mode == "concatenation_sigmoid":
            sigm_psi_f = nn.functional.sigmoid(psi_f)
        else:
            raise NotImplementedError

        y = sigm_psi_f * x1
        W_y = self.W(y)

        return W_y

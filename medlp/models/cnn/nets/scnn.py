import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import vgg16_bn


class SCNN(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            input_size,
            ms_ks=9,
            is_deconv=False,
            use_dilated_conv=True,
            pretrained=True
    ):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.ms_ks = ms_ks
        self.pretrained = pretrained
        self.is_deconv = is_deconv
        self.use_dilated_conv = use_dilated_conv

        self.net_init()
        if not pretrained:
            self.weight_init()

        self.scale_background = 0.01
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([self.scale_background]+[1]*(self.out_channels-1)))
        #self.bce_loss = nn.BCELoss()

        if not self.is_deconv:
            self.upsample = None
        else:
            self.upsample = nn.ConvTranspose2d(self.out_channels, self.out_channels,
                                               kernel_size=3, stride=8,
                                               padding=1, output_padding=1)

    def forward(self, img):
        x = self.backbone(img)
        x = self.layer1(x)
        x = self.message_passing_forward(x)
        x = self.layer2(x)

        if self.upsample is None:
            seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            seg_pred = self.upsample(x)

        # x = self.layer3(x)
        # x = x.view(-1, self.fc_input_feature)
        # exist_pred = self.fc(x)

        # if seg_gt is not None and exist_gt is not None:
        #     loss_seg = self.ce_loss(seg_pred, seg_gt)
        #     loss_exist = self.bce_loss(exist_pred, exist_gt)
        #     loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        # else:
        #     loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
        #     loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
        #     loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def net_init(self):
        input_w, input_h = self.input_size
        self.fc_input_feature = 5 * int(input_w/16) * int(input_h/16)
        self.backbone = vgg16_bn(pretrained=self.pretrained, in_channels=self.in_channels).features

        # ----------------- process backbone -----------------
        if self.use_dilated_conv:
            for i in [34, 37, 40]:
                conv = self.backbone._modules[str(i)]
                dilated_conv = nn.Conv2d(
                    conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                    padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
                )
                dilated_conv.load_state_dict(conv.state_dict())
                self.backbone._modules[str(i)] = dilated_conv
            self.backbone._modules.pop('33')
            self.backbone._modules.pop('43')

        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, self.ms_ks), padding=(0, self.ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, self.ms_ks), padding=(0, self.ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(128, 128, (self.ms_ks, 1), padding=(self.ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(128, 128, (self.ms_ks, 1), padding=(self.ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, self.out_channels, 1)  # get (nB, 5, 36, 100)
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )

        # for multi-classes classification
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_channels-1),
            nn.Sigmoid()
        )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()
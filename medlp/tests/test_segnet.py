import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from medlp.models.cnn.nets.segnet import SegNet
# from tests.utils import test_script_save

# CASES_1D = []
# for mode in ["pixelshuffle", "nontrainable", "deconv", None]:
#     kwargs = {
#         "dim": 1,
#         "in_channels": 5,
#         "out_channels": 8,
#     }
#     if mode is not None:
#         kwargs["upsample"] = mode  # type: ignore
#     CASES_1D.append(
#         [
#             kwargs,
#             (10, 5, 17),
#             (10, 8, 17),
#         ]
#     )

CASES_2D = []
for depth in [3, 4, 5]:
    # d1, d2 = 32, 32
    for d1 in [32, 64, 92]:
        for d2 in [32, 64]:
            in_channels, out_channels = 2, 3
            CASES_2D.append(
                [
                    {
                        "dim": 2,
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "n_depth": depth,
                    },
                    (2, in_channels, d1, d2),
                    (2, out_channels, d1, d2),
                ]
            )

CASES_3D = [
    [  # single channel 3D, batch 2
        {
            "dim": 3,
            "in_channels": 1,
            "out_channels": 2,
            "n_depth": 4,
        },
        (2, 1, 32, 32, 16),
        (2, 2, 32, 32, 16),
    ],
    [  # 2-channel 3D, batch 3
        {
            "dim": 3,
            "in_channels": 2,
            "out_channels": 7,
            "n_depth": 3,
        },
        (3, 2, 16, 18, 20),
        (3, 7, 16, 18, 20),
    ]
]


class TestSegNET(unittest.TestCase):
    @parameterized.expand(CASES_2D+CASES_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(input_param)
        net = SegNet(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
        self.assertEqual(result.shape, expected_shape)

    # def test_script(self):
    #     net = BasicUNet(dim=2, in_channels=1, out_channels=3)
    #     test_data = torch.randn(16, 1, 32, 32)
    #     test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch
from network_architecture.generic_UNet import Generic_UNet
from network_architecture.initialization import InitWeights_He
from training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from network_architecture.custom_modules.helperModules import MyGroupNorm
from utilities.nd_softmax import softmax_helper
from torch import nn
import numpy as np
from typing import Tuple
from utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from typing import Tuple
from typing import Union, Tuple, List


class nnUNetTrainerV2_GN(nnUNetTrainerV2):
    def initialize_network(self):
        """
        changed deep supervision to False
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = MyGroupNorm

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = MyGroupNorm

        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'num_groups': 8}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    # def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
    #                                                      mirror_axes: Tuple[int] = None,
    #                                                      use_sliding_window: bool = True, step_size: float = 0.5,
    #                                                      use_gaussian: bool = True, pad_border_mode: str = 'constant',
    #                                                      pad_kwargs: dict = None, all_in_gpu: bool = False,
    #                                                      verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     We need to wrap this because we need to enforce self.network.do_ds = False for prediction
    #     """
    #     ds = self.network.do_ds
    #     self.network.do_ds = False
    #     # data = self.data[1]
    #     print("DATA SHAPE",data.shape)
    #     ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
    #                                                                    do_mirroring=do_mirroring,
    #                                                                    mirror_axes=mirror_axes,
    #                                                                    use_sliding_window=use_sliding_window,
    #                                                                    step_size=step_size, use_gaussian=use_gaussian,
    #                                                                    pad_border_mode=pad_border_mode,
    #                                                                    pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
    #                                                                    verbose=verbose,
    #                                                                    mixed_precision=mixed_precision)
    #     self.network.do_ds = ds
    #     return ret

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!

        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        print("DATA SHAPE",x.shape)
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float).cuda(self.get_device(), non_blocking=True)
        print("result_torch SHAPE",result_torch.shape)

        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, ))))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch
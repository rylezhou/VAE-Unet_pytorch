#    Copyright 2021 Stanford School of Medicine Department of Oncology Ryle Zhou 
#    rylezhou@stanford.edu Stanford University, Stanford CA United States
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
import numpy as np
from network_architecture.generic_UNet_VAE import Generic_UNet_VAE
from network_architecture.initialization import InitWeights_He
from training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from network_architecture.custom_modules.helperModules import MyGroupNorm
from utilities.nd_softmax import softmax_helper
from torch import nn
from training.loss_functions.dice_loss import DC_and_CE_KL_Loss

from utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from typing import Tuple
from typing import Union, Tuple, List

from torch.cuda.amp import autocast

# INIT_WEIGHT = 1e-5
INIT_WEIGHT = 1e-2
class nnUNetTrainerV2_GN_VAE(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DC_and_CE_KL_Loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

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
        self.network = Generic_UNet_VAE(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(INIT_WEIGHT),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        ## DEBUG:([2, 1, 80, 224, 128])
        data = data_dict['data'] ## input
        # print("INPUT_SHAPE",data.shape)
        target = data_dict['target'] ##seg target
        # print([t.shape for t in target])
      

        data = maybe_to_torch(data)
    
        target = maybe_to_torch(target)

        target = [[t, data] for t in target] ## now target has vae 


        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, output, target):  ## add this for vae
        """
        """
        output = [o[0] for o in output]
        target = [t[0] for t in target]
        return super().run_online_evaluation(output, target)
    

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        print(x.shape)
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!

        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float).cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        x, _,_ = self(x)
        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(x)
                # pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(torch.flip(x, (4, )))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (4, ))))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(torch.flip(x, (3, )))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(torch.flip(x, (4, 3)))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(torch.flip(x, (2, )))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(torch.flip(x, (4, 2)))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(torch.flip(x, (3, 2)))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(torch.flip(x, (4, 3, 2)))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    
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
import numpy as np
from network_architecture.generic_UNet_VAE import Generic_UNet_VAE
# from network_architecture.generic_modular_residual_UNet import Generic_UNet
from network_architecture.initialization import InitWeights_He
from training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from network_architecture.custom_modules.helperModules import MyGroupNorm
from utilities.nd_softmax import softmax_helper
from torch import nn
from training.loss_functions.dice_loss import DC_and_CE_KL_Loss

from utilities.to_torch import maybe_to_torch, to_cuda
# from network_architecture.generic_UNet import Generic_UNet
# from network_architecture.initialization import InitWeights_He
# from network_architecture.neural_network import SegmentationNetwork
# from training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
#     get_patch_size, default_3D_augmentation_params
# from training.dataloading.dataset_loading import unpack_dataset
# from training.network_training.nnUNetTrainer import nnUNetTrainer
# from utilities.nd_softmax import softmax_helper
# from sklearn.model_selection import KFold
# from torch import nn
from torch.cuda.amp import autocast

INIT_WEIGHT = 1e-5
# INIT_WEIGHT = 1e-2
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
        data = data_dict['data']
        target = data_dict['target']
        # print([t.shape for t in target])
      

        data = maybe_to_torch(data)
    
        target = maybe_to_torch(target)

        target = torch.cat((target[0], data), dim=1)


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
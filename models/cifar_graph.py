# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """Random graph models for CIFAR-10"""

    def __init__(self, plan, initializer, outputs=10):
        super(Model, self).__init__()
        
        layers = []
        filters = 3
        layer_types, node_num, p, channels, graph_mode, seed = plan
        for i, spec in enumerate(layer_types):
            if layer_type == 'cv':
                layer = nn.Sequential(
                    nn.Conv2d(in_channels=filters, out_channels=channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    # Daniel: There is an extra ReLU() here in the last layer compared to leaderj1001's implementation. 
                    # As far as I can tell, this is because his RandWire Unit() implements:
                    #     ReLU -> Conv -> BatchNorm -> Dropout
                    # So this ReLU() is actually an extra ReLU(). But I don't think it matters, ReLU^2 = ReLU anyway. 
                    nn.ReLU()
                )
            elif layer_type == 'rw':
                layer = RandWire(node_num, p, filters, channels[i], graph_mode, True, name=f"RandWire_{i}", graph_seed = seed)
            else:
                raise ValueError("layer_type must be one of 'cv' or 'rw'")
            filters = channels
            layers.append(layer)
                
        # Top off with a 1x1 convolution
        layers.append(nn.Sequential(
            nn.Conv2d(filters, 1280, kernel_size=1),
            nn.BatchNorm2d(1280)
        ))
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(1280, outputs)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(initializer)

    def forward(self, x):
        x = self.layers(x)
        batch, channels, height, width = x.size()
        x = F.avgpool2d(x, kernel_size=[height_width])
        x = torch.squeeze(x)
        x = self.fc(x)
        return x

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        def isfloat(s):
            try:
                float(s)
                return True
            except:
                return False
        
        components = model_name.split('_')
        
        return (components[0] == "cifar" and
                components[1] == "graph" and
                components[2] in ['er', 'ws', 'ba'] and
                components[3].isdigit() and
                isfloat(components[4]) and (0 < float(components[4]) < 1) and
                components[5].isdigit() and
                components[6].isdigit())

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=10):
        """ 
        Model should be named as cifar_graph_{graph_mode}_{node_num}_{p}_{base_channels}_{graph_seed}
        e.g. cifar_graph_er_32_0.75_64_1
        """
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        outputs = outputs or 10

        components = model_name.split('_')
        _, _, graph_mode, node_num, p, base_channels, graph_seed = components 
        layer_types = ['cv', 'cv', 'rw', 'rw']
        channels = [base_channels, base_channels, base_channels * 2, base_channels * 4]
        plan = (layer_types, node_num, p, channels, graph_mode, graph_seed)
        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_graph_er_32_0.75_64_1',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep'
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight'
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)

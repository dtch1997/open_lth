# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Random Local Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    """
    Random local pruning. 
    Prunes each layer randomly by pruning_fraction. 
    """
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        if pruning_hparams.pruning_layers_to_ignore:
            prunable_tensors -= set(pruning_hparams.pruning_layers_to_ignore.split(','))

        # Daniel: Make a copy of the mask so that the old mask isn't changed. 
        # I don't know if this is strictly necessary but it's better to be safe. 
        new_mask_dict = {k: np.copy(v) for k,v in current_mask.items()}
        for k, v in new_mask_dict.items():
            if k in prunable_tensors: continue
            
            # Determine the number of remaining prunable weights
            number_of_remaining_weights = np.sum(v)
            # Determine the number of weights that need to be pruned.
            number_of_weights_to_prune = np.ceil(
                pruning_hparams.pruning_fraction * number_of_remaining_weights).astype(int)
            weight_indices_to_prune = np.random.choice(number_of_remaining_weights, number_of_weights_to_prune, replace=False)
            prune_indices = lambda indices: tuple(idx[weight_indices_to_prune] for idx in indices)
            v[prune_indices(v.nonzero())] = 0
        new_mask = Mask(new_mask_dict)
        return new_mask

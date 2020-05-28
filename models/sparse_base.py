# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import models
import pruning

class SparseModel(abc.ABC, models.base.Model):
    """
    Base class for constructing sparse models. 
    Subclasses can use the given default_strategy, or override it.
    """
    
    default_strategy = pruning.registry.registered_strategies['sparse_global']
    
    @property
    @abc.abstractmethod
    def sparsify(self, strategy: pruning.base.Strategy = None, 
                 pruning_hparams: pruning.base.PruningHparams = None, 
                 mask : pruning.mask.Mask = None):
        """
        Returns a new PrunedModel without changing the current model.  
        """
        if strategy is None:
            strategy = self.__class__.default_strategy
        if pruning_hparams is None:
            pruning_hparams = strategy.get_pruning_hparams()
        assert isinstance(strategy, pruning.base.Strategy)
        assert isinstance(pruning_hparams, pruning.base.PruningHparams)
        mask = strategy.prune(pruning_hparams, self, mask)
        return pruning.pruned_model.prunedModel(self, mask) 
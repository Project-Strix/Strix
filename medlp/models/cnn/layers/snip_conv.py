from torch import nn

class PrunableWeights():
    """Intended to be inherited along with a nn.Module"""

    def set_pruning_mask(self, mask):
        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero

        self.weight.data[mask == 0.] = 0.

        def hook(grads):
            return grads * mask

        self.weight.register_hook(hook)


class PrunableLinear(nn.Linear, PrunableWeights):
    pass

class PrunableConv3d(nn.Conv3d, PrunableWeights):
    pass

class PrunableDeconv3d(nn.ConvTranspose3d, PrunableWeights):
    pass
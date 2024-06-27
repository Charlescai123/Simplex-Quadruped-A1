"""Taylor Parameters"""

class TaylorParams:
    def __init__(self):
        self.dense_dims = [10, 10]  # dim for hidden layers, not including input dim and output dim
        self.aug_order = [1, 1, 0]  # augmentation order for all hidden layers. 2 will lead to 3rd order
        self.initializer_w = 'tn'
        self.initializer_b = 'uniform'
        #self.activations = ['null', 'null']  # activations for hidden layers, not including output
        self.activations = ['relu', 'relu']  # activations for hidden layers, not including output
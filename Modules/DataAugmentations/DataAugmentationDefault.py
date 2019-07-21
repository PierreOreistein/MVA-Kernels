class DataAugmentationDefault(object):
    """Default class for data_augmentation function."""

    def __init__(self, data_augmentation=None, hp=None):
        """Define the data_augmentation function with the given hyperparameters hp."""

        self.data_augmentation = data_augmentation
        self.hp = hp
        self.name = data_augmentation.__name__

    def call(self, X, Y):
        """Call the function data_augmentation with the given hp."""

        return self.data_augmentation(X, Y, **self.hp)

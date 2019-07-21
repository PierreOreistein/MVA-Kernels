class KernelDefault(object):
    """Default class for kernel function."""

    def __init__(self, kernel=None, hp=None):
        """Define the kernel function with the given hyperparameters hp."""

        self.kernel = kernel
        self.hp = hp
        self.name = kernel.__name__

    def call(self, X, Y):
        """Call the function kernel with the given hp."""

        return self.kernel(X, Y, **self.hp)

class EmbeddingDefault(object):
    """Default class for embedding function."""

    def __init__(self, embedding=None, hp=None):
        """Define the embedding function with the given hyperparameters hp."""

        self.embedding = embedding
        self.hp = hp
        try:
            self.name = embedding.__name__
        except AttributeError:
            self.name = None

    def call(self, X, train=True, X_train=None):
        """Call the function embedding with the given hp."""

        return self.embedding(X, train=train, X_train=X_train, **self.hp)

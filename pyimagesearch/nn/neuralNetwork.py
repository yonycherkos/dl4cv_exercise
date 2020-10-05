import numpy as np


class NeuralNetwork():
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []

        # STEP - 1: WEIGHT INITIALIZATION        
        for layer in np.arange(0, len(layers) - 2):
            # add bias term to the weight
            w = np.random.randn(layers[layer] + 1, layers[layer + 1] + 1)
            self.W.append(w / np.sqrt(layers[layer]))

        # don't add bias term to the last layer
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(layer) for layer in self.layers))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivation(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, verbose=100):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        # STEP - 2: FEADFORWARD
        # compute activation for each layer
        A = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.layers) - 1):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        # STEP - 3: COMPUTE LOSS
        error = A[-1] - y

        # STEP - 4: BACKPROBAGATION
        D = [error * self.sigmoid_derivation(A[-1])]

        for layer in np.arange(len(self.layers) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_derivation(A[layer])
            D.append(delta)

        # reverse the order of the deltas
        D = D[::-1]

        # STEP - 5: WEIGHT UPDATE
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(p.dot(self.W[layer]))
        
        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        preds = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((preds - targets) ** 2)
        return loss


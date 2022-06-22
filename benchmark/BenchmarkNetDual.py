from DualTensor import DualTensor
from benchmark.BenchmarkNetTorch import BenchmarkNetTorch
import numpy as np


class BenchmarkNetDual:

    def __init__(self, depth: int, layer_size: int):
        self.depth = depth
        self.layer_size = layer_size
        self.layers = []
        for i in range(depth):
            self.layers.append(
                [
                    DualTensor(np.random.normal(0, np.sqrt(1 / layer_size), (layer_size, layer_size)).astype(np.float)),
                    DualTensor(np.random.normal(0, np.sqrt(1 / layer_size), layer_size).astype(np.float))
                ]
            )
        self.layers.append(
            [
                DualTensor(np.random.normal(0, np.sqrt(1 / layer_size), (layer_size, 1)).astype(np.float)),
                DualTensor(np.random.normal(0, np.sqrt(1 / layer_size), 1).astype(np.float))
            ]
        )

    def clone_weights(self, net: BenchmarkNetTorch):
        assert len(self.layers) == len(net.layers)
        for l, n in zip(self.layers, net.layers):
            assert l[0].shape == (n.in_features, n.out_features)
            assert l[1].shape == (n.out_features,)
            l[0] = DualTensor(n.weight.detach().numpy().T)
            l[1] = DualTensor(n.bias.detach().numpy())

    def fullpass(self, x):
        x = DualTensor(x)
        for (w, b) in self.layers:
            x = x @ w + b
        return x

    def forward(self, x):
        return self.fullpass(x).re()

    def vectorgrad(self, x, vec):
        for i in range(len(self.layers)):
            (w, b) = self.layers[i]
            (gw, gb) = vec[i]
            self.layers[i] = [DualTensor(w.re(), gw), DualTensor(b.re(), gb)]
        return self.fullpass(x).im().sum()

    def fullgrad(self, x):
        grads = [(np.zeros(w.shape), np.zeros(b.shape)) for w, b in self.layers]
        vec = [(np.zeros(w.shape), np.zeros(b.shape)) for w, b in self.layers]
        for i in range(len(self.layers)):
            w, b = self.layers[i]
            for idx in np.ndindex(w.shape):
                vec[i][0][idx] = 1
                grads[i][0][idx] = self.vectorgrad(x, vec).sum()
                vec[i][0][idx] = 0
            for idx in np.ndindex(b.shape):
                vec[i][1][idx] = 1
                grads[i][1][idx] = self.vectorgrad(x, vec).sum()
                vec[i][1][idx] = 0
        return grads


if __name__ == '__main__':
    net = BenchmarkNetDual(3, 4)
    print(net.forward(np.array([[1, 2, 3, 4], [3, 2, 6, 1]])))

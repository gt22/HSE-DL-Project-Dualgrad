import torch
from torch.nn import Sequential, Linear


class BenchmarkNetTorch:

    def __init__(self, depth: int, layer_size: int):
        self.depth = depth
        self.layer_size = layer_size
        self.layers = []
        for i in range(depth):
            self.layers.append(Linear(layer_size, layer_size))
        self.layers.append(Linear(layer_size, 1))
        self.net = Sequential(*self.layers)

    def forward(self, x):
        return self.net(torch.tensor(x))

    def fullgrad(self, x):
        t = self.forward(x)
        t.sum().backward()
        grads = []
        for l in self.layers:
            grads.append((l.weight.grad.clone(), l.bias.grad.clone()))
        self.net.zero_grad()
        return grads

    def vectorgrad(self, x, vec):
        grad = self.fullgrad(x)
        ret = 0
        for (gw, gb), (vw, vb) in zip(grad, vec):
            ret += (gw.T * vw).sum() + (gb * vb).sum()
        return ret


if __name__ == '__main__':
    net = BenchmarkNetTorch(3, 4)
    print(net.fullgrad(torch.tensor([[1, 2, 3, 4], [3, 2, 6, 1]], dtype=torch.float)))
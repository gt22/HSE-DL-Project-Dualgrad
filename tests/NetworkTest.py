from benchmark.BenchmarkNetDual import BenchmarkNetDual as DualNet
from benchmark.BenchmarkNetTorch import BenchmarkNetTorch as TorchNet
from benchmark.Benchmark import random_gradvec
import torch
import numpy as np


def rand(*size):
    return torch.normal(0, 5, size=size).numpy()


def setup(use_vec=False):
    torch.random.manual_seed(6741)
    depth = 4
    size = 8
    batch_size = 5
    tnet = TorchNet(depth, size)
    dnet = DualNet(depth, size)
    dnet.clone_weights(tnet)
    x = rand(batch_size, size)
    if use_vec:
        vec = random_gradvec(depth, size, rand)
        return dnet, tnet, x, vec
    return dnet, tnet, x


def tensor_eq(a, b):
    with torch.no_grad():
        return ((np.array(a) - np.array(b)) < 1e-5).all()


def test_forward():
    with torch.no_grad():
        dnet, tnet, x = setup()
        assert tensor_eq(dnet.forward(x), tnet.forward(x))


def test_fullgrad():
    dnet, tnet, x = setup()
    dgrad = dnet.fullgrad(x)
    tgrad = tnet.fullgrad(x)
    assert len(dgrad) == len(tgrad)
    for (dw, db), (tw, tb) in zip(dgrad, tgrad):
        assert tensor_eq(dw, tw.T)
        assert tensor_eq(db, tb)


def test_vectorgrad():
    dnet, tnet, x, vec = setup(True)
    assert (dnet.vectorgrad(x, vec) - tnet.vectorgrad(x, vec)) < 1e-5

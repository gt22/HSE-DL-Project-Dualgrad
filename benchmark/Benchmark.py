import timeit
import torch
from benchmark.BenchmarkNetTorch import BenchmarkNetTorch as TorchNet
from benchmark.BenchmarkNetDual import BenchmarkNetDual as DualNet
from memory_profiler import memory_usage

def random_gradvec(depth, size, rnd):
    vec = []
    for i in range(depth):
        vec.append((
            rnd(size, size),
            rnd(size)
        ))
    vec.append((
        rnd(size, 1),
        rnd(1)
    ))
    return vec


def rand(*size):
    return torch.normal(0, 5, size=size).numpy()


def benchmark_vectorgrad(net, batch_size):
    print("vectorgrad")
    vec = random_gradvec(net.depth, net.layer_size, rand)
    x = rand(batch_size, net.layer_size)
    return timeit.repeat(lambda: net.vectorgrad(x, vec), repeat=100, number=1000)


def benchmark_fullgrad(net, batch_size):
    print("fullgrad")
    x = rand(batch_size, net.layer_size)
    return timeit.repeat(lambda: net.fullgrad(x), repeat=50, number=100)


def run_benchmark_(name, net, batch_size):
    for v in benchmark_vectorgrad(net, batch_size):
        yield (name, "vector", net.depth, net.layer_size, batch_size, v)
    for f in benchmark_fullgrad(net, batch_size):
        yield (name, "full", net.depth, net.layer_size, batch_size, f)


def run_benchmark(depth, layer_size, batch_size):
    yield from run_benchmark_("dual", DualNet(depth, layer_size), batch_size)
    yield from run_benchmark_("auto", TorchNet(depth, layer_size), batch_size)


def memory_vectorgrad(net, batch_size):
    print("vectorgrad")
    vec = random_gradvec(net.depth, net.layer_size, rand)
    x = rand(batch_size, net.layer_size)
    return memory_usage((net.vectorgrad, (x, vec)), max_usage=True)


def memory_fullgrad(net, batch_size):
    print("fullgrad")
    x = rand(batch_size, net.layer_size)
    return memory_usage((net.fullgrad, (x,)), max_usage=True)


def run_memory_(name, net, batch_size):
    yield (name, "vector", net.depth, net.layer_size, batch_size, memory_vectorgrad(net, batch_size))
    yield (name, "full", net.depth, net.layer_size, batch_size, memory_fullgrad(net, batch_size))


def run_memory(depth, layer_size, batch_size):
    yield from run_memory_("dual", DualNet(depth, layer_size), batch_size)
    yield from run_memory_("auto", TorchNet(depth, layer_size), batch_size)


if __name__ == '__main__':
    for depth in range(4, 11):
        with open('benchmark.csv', 'a') as f:
            # f.write('nettype,gradtype,depth,layer_size,batch_size,memory\n')
            for b in run_memory(depth, 10, 20):
                f.write(','.join(map(str, b)))
                f.write('\n')
                f.flush()

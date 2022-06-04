import pytest
import numpy as np
from DualTensor import DualTensor
from typing import Callable, Iterable


def generate(n, shape) -> Iterable[DualTensor]:
    a = np.random.random((n, *shape)) * 5
    b = np.random.random((n, *shape)) * 5
    for x, y in zip(a, b):
        yield DualTensor(x, y)


def gradient_test(f: Callable[[DualTensor, DualTensor], DualTensor], shape1, shape2, eps=1e-5, rtol=1e-5):
    a = generate(100, shape1)
    b = generate(100, shape2)
    for x, y in zip(a, b):
        res_shape = f(x, y).shape
        for idx in np.ndindex(res_shape):
            z1 = f(x - eps, y).re()
            z2 = f(x + eps, y).re()
            assert z1.shape == z2.shape
            num_grad = ((z2 - z1) / (2 * eps))[idx]
            exact_grad_te = f(x.grad_target(idx), y.grad_nontarget()).im()
            assert exact_grad_te.shape == z1.shape
            exact_grad = exact_grad_te[idx]
            if not np.isnan(exact_grad):
                rel_tol = abs(num_grad - exact_grad) / (1 + np.minimum(abs(num_grad), abs(exact_grad)))
                assert rel_tol < rtol


def value_test(f: Callable[[DualTensor, DualTensor], DualTensor], f_real: Callable[[np.ndarray, np.ndarray], np.ndarray], shape1, shape2, rtol=1e-10):
    a = generate(100, shape1)
    b = generate(100, shape2)
    for x, y in zip(a, b):
        dual = f(x, y).re()
        real = f_real(x.re(), y.re())

        rel_tol = abs(dual - real) / (1 + np.minimum(dual, real))
        assert rel_tol.max() < rtol


def run_test(f, shape1, shape2=None, run_grad=True):
    if shape2 is None:
        shape2 = shape1
    np.random.seed(6741)
    value_test(f, f, shape1, shape2)
    if run_grad:
        gradient_test(f, shape1, shape2)


def test_add():
    run_test(lambda x, y: y + x, (10, 10))


def test_radd():
    run_test(lambda x, y: y.__radd__(x), (10, 10))


def test_sub():
    run_test(lambda x, y: y - x, (10, 10))


def test_rsub():
    run_test(lambda x, y: y.__rsub__(x), (10, 10))
    

def test_neg():
    run_test(lambda x, _: -x, (10, 10))


def test_abs():
    run_test(lambda x, _: abs(x), (10, 10))


def test_mul():
    run_test(lambda x, y: y * x, (10, 10))


def test_rmul():
    run_test(lambda x, y: y.__rmul__(x), (10, 10))


def test_truediv():
    run_test(lambda x, y: x / y, (10, 10))


def test_rtruediv():
    run_test(lambda x, y: y.__rtruediv__(x), (10, 10))


def test_floordiv():
    run_test(lambda x, y: x // y, (10, 10), run_grad=False)


def test_rfloordiv():
    run_test(lambda x, y: y.__rfloordiv__(x), (10, 10), run_grad=False)


def test_mod():
    run_test(lambda x, y: x % y, (10, 10), run_grad=False)


def test_rmod():
    run_test(lambda x, y: y.__rmod__(x), (10, 10), run_grad=False)


def test_pow():
    run_test(lambda x, y: x ** y, (10, 10), (1, ))


def test_rpow():
    run_test(lambda x, y: y.__rpow__(x), (10, 10), (1, ))

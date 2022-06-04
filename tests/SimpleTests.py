import pytest
import numpy as np
from DualNumber import DualNumber
from typing import Callable


def generate(n):
    a = np.random.random(n) * 20
    b = np.random.random(n) * 20
    for x, y in zip(a, b):
        yield DualNumber(x, y)


def gradient_test(f: Callable[[DualNumber, DualNumber], DualNumber], eps=1e-5, rtol=1e-5):
    a = generate(100)
    b = generate(100)
    for x, y in zip(a, b):
        z1 = f(x - eps, y).re()
        z2 = f(x + eps, y).re()

        num_grad = (z2 - z1) / (2 * eps)
        exact_grad = f(x.grad_target(), y.grad_nontarget()).im()
        if not np.isnan(exact_grad):
            rel_tol = abs(num_grad - exact_grad) / (1 + min(abs(num_grad), abs()))
            assert rel_tol < rtol


def value_test(f: Callable[[DualNumber, DualNumber], DualNumber], f_real: Callable[[float, float], float], rtol=1e-10):
    a = generate(100)
    b = generate(100)
    for x, y in zip(a, b):
        dual = f(x, y).re()
        real = f_real(x.re(), y.re())

        rel_tol = abs(dual - real) / (1 + min(dual, real))
        assert rel_tol < rtol


def run_test(f, run_grad=True):
    np.random.seed(6741)
    value_test(f, f)
    if run_grad:
        gradient_test(f)


def test_add():
    run_test(lambda x, y: y + x)


def test_radd():
    run_test(lambda x, y: y.__radd__(x))


def test_sub():
    run_test(lambda x, y: y - x)


def test_rsub():
    run_test(lambda x, y: y.__rsub__(x))
    

def test_neg():
    run_test(lambda x, _: -x)


def test_abs():
    run_test(lambda x, _: abs(x))


def test_mul():
    run_test(lambda x, y: y * x)


def test_rmul():
    run_test(lambda x, y: y.__rmul__(x))


def test_truediv():
    run_test(lambda x, y: x / y)


def test_rtruediv():
    run_test(lambda x, y: y.__rtruediv__(x))


def test_floordiv():
    run_test(lambda x, y: x // y, run_grad=False)


def test_rfloordiv():
    run_test(lambda x, y: y.__rfloordiv__(x), run_grad=False)


def test_mod():
    run_test(lambda x, y: x % y, run_grad=False)


def test_rmod():
    run_test(lambda x, y: y.__rmod__(x), run_grad=False)


def test_pow():
    run_test(lambda x, y: x ** y)


def test_rpow():
    run_test(lambda x, y: y.__rpow__(x))

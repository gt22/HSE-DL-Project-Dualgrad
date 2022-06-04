from __future__ import annotations
import numpy as np
from typing import Optional


class DualTensor:
    __slots__ = 'a', 'b', 'shape'

    a: np.ndarray
    b: np.ndarray

    def __init__(self, a: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None):
        self.shape = a.shape if a is not None else (b.shape if b is not None else (1,))
        self.a = a if a is not None else np.zeros(self.shape)
        self.b = b if b is not None else np.zeros(self.shape)

    def re(self) -> np.ndarray:
        return self.a

    def im(self) -> np.ndarray:
        return self.b

    def grad_target(self, idx):
        b = np.zeros(self.shape)
        b[idx] = 1
        return DualTensor(self.a, b)

    def grad_nontarget(self):
        return DualTensor(self.a, np.zeros(self.shape))

    def __repr__(self):
        return f'DualTensor({self.a}, {self.b})'

    def __str__(self):
        return f'({self.a}) + ({self.b})ε'

    @staticmethod
    def check_shape(x, expected_shape):
        if expected_shape is None:
            return
        if x.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {x.shape}")

    @staticmethod
    def normalize(x, expected_shape=None) -> DualTensor:
        if isinstance(x, DualTensor):
            DualTensor.check_shape(x, expected_shape)
            return x
        if isinstance(x, np.ndarray):
            DualTensor.check_shape(x, expected_shape)
            return DualTensor(x)
        return DualTensor(np.full(expected_shape if expected_shape is not None else (1,), x))

    def __add__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return DualTensor(self.a + other.a, self.b + other.b)

    def __radd__(self, other):
        return DualTensor.normalize(other, self.shape) + self

    def __sub__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return DualTensor(self.a - other.a, self.b - other.b)

    def __rsub__(self, other):
        return DualTensor.normalize(other, self.shape) - self

    def __neg__(self):
        return DualTensor(-self.a, -self.b)

    def __abs__(self):
        sgn = (self.a > 0) * 2 - 1  # like np.sign but return 1 on x==0
        return DualTensor(abs(self.a), self.b * sgn)

    def __mul__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return DualTensor(self.a * other.a, self.a * other.b + other.a * self.b)

    def __rmul__(self, other):
        return DualTensor.normalize(other, self.shape) * self

    def __truediv__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return DualTensor(self.a / other.a, (self.b * other.a - self.a * other.b) / (other.a ** 2))

    def __rtruediv__(self, other):
        return DualTensor.normalize(other, self.shape) / self

    def __floordiv__(self, other):
        print("WARNING: Using Dual floordiv, no gradient")
        other = DualTensor.normalize(other, self.shape)
        return DualTensor(self.a // other.a, np.zeros(self.shape))

    def __rfloordiv__(self, other):
        return DualTensor.normalize(other, self.shape) // self

    def __mod__(self, other):
        print("WARNING: Using Dual mod, no gradient")
        other = DualTensor.normalize(other, self.shape)
        return DualTensor(self.a % other.a, np.zeros(self.shape))

    def __rmod__(self, other):
        return DualTensor.normalize(other, self.shape) % self

    def __pow__(self, other):
        other = DualTensor.normalize(other, (1,))
        if other.a == 1:
            return self
        if (self.a == 0).any():
            raise NotImplementedError("0 to power for tensors - not implemented, tricky gradient")
        real_power = self.a ** other.a
        im_adjust = self.b * other.a / self.a
        if other.b != 0:
            im_adjust += other.b * np.log(self.a)

        return DualTensor(real_power, real_power * im_adjust)

    def __rpow__(self, other):
        return DualTensor.normalize(other) ** self

    def __lshift__(self, other: int):
        print("WARNING: Using Dual bitshift, unoptimized")
        return self * (pow(2, other))

    def __rshift__(self, other: int):
        print("WARNING: Using Dual bitshift, unoptimized")
        return self / (pow(2, other))

    def __eq__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return (self.a == other.a) and (self.b == other.b)

    def __lt__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return self.a < other.a

    def __le__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return self.a <= other.a

    def __gt__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return self.a > other.a

    def __ge__(self, other):
        other = DualTensor.normalize(other, self.shape)
        return self.a >= other.a

    def __hash__(self):
        return hash((self.a, self.b))

    def __bool__(self):
        return bool(self.a) or bool(self.b)

    def copy(self):
        return DualTensor(np.array(self.a, copy=True), np.array(self.b, copy=True))

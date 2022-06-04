from __future__ import annotations
import numpy as np


class DualNumber:
    __slots__ = 'a', 'b'

    a: float
    b: float

    def __init__(self, a: float = 0., b: float = 0.):
        self.a = a
        self.b = b

    def re(self) -> float:
        return self.a

    def im(self) -> float:
        return self.b

    def grad_target(self) -> DualNumber:
        return DualNumber(self.a, 1)

    def grad_nontarget(self) -> DualNumber:
        return DualNumber(self.a, 0)

    def __repr__(self):
        return f'DualNumber({self.a}, {self.b})'

    def __str__(self):
        return f'({self.a}) + ({self.b})ε'

    @staticmethod
    def normalize(x) -> DualNumber:
        if isinstance(x, DualNumber):
            return x
        return DualNumber(float(x))

    def __add__(self, other):
        other = DualNumber.normalize(other)
        return DualNumber(self.a + other.a, self.b + other.b)

    def __radd__(self, other):
        return DualNumber.normalize(other) + self

    def __sub__(self, other):
        other = DualNumber.normalize(other)
        return DualNumber(self.a - other.a, self.b - other.b)

    def __rsub__(self, other):
        return DualNumber.normalize(other) - self

    def __neg__(self):
        return DualNumber(-self.a, -self.b)

    def __abs__(self):
        return DualNumber(abs(self.a), self.b if self.a > 0 else -self.b)

    def __mul__(self, other):
        other = DualNumber.normalize(other)
        return DualNumber(self.a * other.a, self.a * other.b + other.a * self.b)

    def __rmul__(self, other):
        return DualNumber.normalize(other) * self

    def __truediv__(self, other):
        other = DualNumber.normalize(other)
        return DualNumber(self.a / other.a, (self.b * other.a - self.a * other.b) / (other.a ** 2))

    def __rtruediv__(self, other):
        return DualNumber.normalize(other) / self

    def __floordiv__(self, other):
        print("WARNING: Using Dual floordiv, no gradient")
        other = DualNumber.normalize(other)
        return DualNumber(self.a // other.a, 0)

    def __rfloordiv__(self, other):
        return DualNumber.normalize(other) // self

    def __mod__(self, other):
        print("WARNING: Using Dual mod, no gradient")
        other = DualNumber.normalize(other)
        return DualNumber(self.a % other.a, 0)

    def __rmod__(self, other):
        return DualNumber.normalize(other) % self

    def __pow__(self, other):
        other = DualNumber.normalize(other)
        if other.a == 1:
            return self
        if self.a == 0:
            if other.a == 0:
                return DualNumber(1, float('nan'))
            if other.a < 0:
                return DualNumber(float('nan'), float('nan'))
            if other.b != 0:
                if other.a == 0:
                    return DualNumber(0, other.b * float('-inf'))
                else:
                    return self ** DualNumber(other.a, 0)
            if other.a < 1:
                return DualNumber(0, float('nan'))
            return DualNumber(0, 0)
        real_power = self.a ** other.a
        im_adjust = self.b * other.a / self.a
        if other.b != 0:
            if self.a < 0:
                return DualNumber(real_power, float('nan'))
            im_adjust += other.b * np.log(self.a)

        return DualNumber(real_power, real_power * im_adjust)

    def __rpow__(self, other):
        return DualNumber.normalize(other) ** self

    def __lshift__(self, other: int):
        print("WARNING: Using Dual bitshift, unoptimized")
        return self * (pow(2, other))

    def __rshift__(self, other: int):
        print("WARNING: Using Dual bitshift, unoptimized")
        return self / (pow(2, other))

    def __eq__(self, other):
        other = DualNumber.normalize(other)
        return (self.a == other.a) and (self.b == other.b)

    def __lt__(self, other):
        other = DualNumber.normalize(other)
        return self.a < other.a

    def __le__(self, other):
        other = DualNumber.normalize(other)
        return self.a <= other.a

    def __gt__(self, other):
        other = DualNumber.normalize(other)
        return self.a > other.a

    def __ge__(self, other):
        other = DualNumber.normalize(other)
        return self.a >= other.a

    def __hash__(self):
        return hash((self.a, self.b))

    def __bool__(self):
        return bool(self.a) or bool(self.b)


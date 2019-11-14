from st.expr import Expr
from typing import List, Tuple
from st.grid import Grid


class Buffer:
    iteration: List[Tuple[int, int]]
    rhs: Expr
    name: str
    grid: Grid

    def __init__(self, rhs):
        self.rhs = rhs
        self.depends = list()
        self.iteration = list()
        self.name = None
        self.grid = None

    def ref_name(self):
        return self.name


class BufferRead(Expr):
    buf: Buffer

    def __init__(self, buf):
        super().__init__()
        self.buf = buf


class Shift(Expr):
    _children = ['subexpr']
    subexpr: Expr
    shifts: List[int]

    def __init__(self, shifts, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.shifts = shifts[:]

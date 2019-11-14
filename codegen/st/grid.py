from st.expr import Expr, conv_expr
from typing import List


class Grid:
    def __init__(self, src_name: str, dims: int):
        self.name = src_name
        self.dims = dims
        self.out = None

    def __call__(self, *args, **kwargs):
        if self.out is not None:
            return self.out[1]
        return GridRef(self, list(args))


def eval_offset(idx_expr: Expr):
    import st.expr
    if isinstance(idx_expr, st.expr.Index):
        return idx_expr, 1j
    if isinstance(idx_expr, st.expr.BinOp):
        lhs_idx, lhs_val = eval_offset(idx_expr.lhs)
        rhs_idx, rhs_val = eval_offset(idx_expr.rhs)
        if lhs_idx is None:
            idx = rhs_idx
        elif rhs_idx is None:
            idx = lhs_idx
        else:
            raise ValueError("Using more than one index")
        val = eval(repr(lhs_val) + idx_expr.operator.value + repr(rhs_val))
        return idx, val
    if isinstance(idx_expr, st.expr.UnOp):
        idx, val = eval_offset(idx_expr.subexpr)
        return idx, eval(idx_expr.operator.value + repr(val))
    if isinstance(idx_expr, st.expr.IntLiteral):
        return None, idx_expr.val
    raise ValueError("Wrong format")


class GridRef(Expr):
    _attr = {'atomic': True}

    def __init__(self, grid: Grid, indices: List):
        super().__init__()
        self.grid = grid
        if len(indices) != grid.dims:
            raise ValueError("Index list not consistent with dimensions")
        self.children = []
        self.indices = []
        self.offsets = []
        for idx in indices:
            self.children.append(idx)
            dim, offset = eval_offset(idx)
            if offset.imag != 1:
                raise ValueError("Wrong scaling of the index")
            self.indices.append(dim)
            self.offsets.append(int(offset.real))

    def assign(self, rhs):
        self.grid.out = (self, conv_expr(rhs))

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        ret = ""
        for idx in range(len(self.indices)):
            if idx:
                ret += "|"
            ret += "{}:{}".format(self.indices[idx].n, self.offsets[idx])
        ret = "{}:[{}]".format(self.grid.name, ret)
        return ret

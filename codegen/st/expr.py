"""Here are the AST nodes contained for a simple stencil language"""

from st.expr_meta import ExprMeta
import st.alop
from typing import List, Dict


def conv_expr(input):
    if isinstance(input, int):
        return IntLiteral(input)
    if isinstance(input, float):
        return FloatLiteral(input)
    if isinstance(input, str):
        return ConstRef(input)
    if isinstance(input, Expr):
        return input
    raise ValueError(
        "Cannot convert to expression, {}".format(repr(input)))


class Expr(object, metaclass=ExprMeta):
    """Generic AST node

    Contains a list of children and forms a multiway tree.

    Attributes:
        children List[Node]: The list of children.
        scope (optional): The current scope additions.
        attr: record information attached to the node, can be initialized for all classes using _attr
    """
    _children = []
    _arg_sig = None
    _attr = dict()
    attr: Dict

    def __init__(self, *args, **kwargs):
        bound = self._arg_sig.bind(*args, **kwargs)
        self.children = [None] * len(self._children)
        for name, val in bound.arguments.items():
            setattr(self, name, val)
        self.attr = dict(self._attr)
        self.parent = None

    def visit(self, init, func):
        """Preorder traversal"""
        init, recurse = func(init, self)
        if recurse:
            for child in self.children:
                init = child.visit(init, func)
        return init

    def mk_child(self, child):
        """ Make one node a child
            This does not append the child but rather fixes the parent-child
            relationship
        """
        child.parent = self

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return ""

    def get_attr(self, attr):
        if attr in self.attr:
            return self.attr[attr]
        return None

    def __str__(self):
        ret = "({} [{}] ".format(self.__class__.__name__, self.str_attr())
        for child in self.children:
            ret += str(child)
        ret += ")"
        return ret

    # Arithmetic operators
    def __add__(self, other):
        return BinOp(st.alop.BinaryOperators.Add,
                     self, conv_expr(other))

    def __radd__(self, other):
        return conv_expr(other).__add__(self)

    def __sub__(self, other):
        return BinOp(st.alop.BinaryOperators.Sub,
                     self, conv_expr(other))

    def __rsub__(self, other):
        return conv_expr(other).__sub__(self)

    def __mul__(self, other):
        return BinOp(st.alop.BinaryOperators.Mul,
                     self, conv_expr(other))

    def __rmul__(self, other):
        return conv_expr(other).__mul__(self)

    def __truediv__(self, other):
        return BinOp(st.alop.BinaryOperators.Div,
                     self, conv_expr(other))

    def __rtruediv__(self, other):
        return conv_expr(other).__truediv__(self)

    def __mod__(self, other):
        return BinOp(st.alop.BinaryOperators.Mod,
                     self, conv_expr(other))

    def __rmod__(self, other):
        return conv_expr(other).__mod__(self)

    def __and__(self, other):
        return BinOp(st.alop.BinaryOperators.BitAnd,
                     self, conv_expr(other))

    def __rand__(self, other):
        return conv_expr(other).__and__(self)

    def __xor__(self, other):
        return BinOp(st.alop.BinaryOperators.BitXor,
                     self, conv_expr(other))

    def __rxor__(self, other):
        return conv_expr(other).__xor__(self)

    def __or__(self, other):
        return BinOp(st.alop.BinaryOperators.BitOr,
                     self, conv_expr(other))

    def __ror__(self, other):
        return conv_expr(other).__or__(self)

    def __lshift__(self, other):
        return BinOp(st.alop.BinaryOperators.BitSHL,
                     self, conv_expr(other))

    def __rlshift__(self, other):
        return conv_expr(other).__lshift__(self)

    def __rshift__(self, other):
        return BinOp(st.alop.BinaryOperators.BitSHR,
                     self, conv_expr(other))

    def __rrshift__(self, other):
        return conv_expr(other).__rshift__(self)

    def __neg__(self):
        return UnOp(st.alop.UnaryOperators.Neg, self)

    # Comparison operators
    def __lt__(self, other):
        return BinOp(st.alop.BinaryOperators.Lt,
                     self, conv_expr(other))

    def __le__(self, other):
        return BinOp(st.alop.BinaryOperators.Leq,
                     self, conv_expr(other))

    def __eq__(self, other):
        return BinOp(st.alop.BinaryOperators.Eq,
                     self, conv_expr(other))

    def __ne__(self, other):
        return BinOp(st.alop.BinaryOperators.Neq,
                     self, conv_expr(other))

    def __gt__(self, other):
        return BinOp(st.alop.BinaryOperators.Gt,
                     self, conv_expr(other))

    def __ge__(self, other):
        return BinOp(st.alop.BinaryOperators.Geq,
                     self, conv_expr(other))

    # Logical operators
    def logical_and(self, other):
        return BinOp(st.alop.BinaryOperators.And,
                     self, conv_expr(other))

    def logical_or(self, other):
        return BinOp(st.alop.BinaryOperators.Or,
                     self, conv_expr(other))

    def logical_not(self):
        return UnOp(st.alop.UnaryOperators.Not, self)

    def __hash__(self):
        return id(self)


class Index(Expr):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def genericName(self):
        return "axis{}".format(self.n)

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return str(self.n)


class ReductionOp(Expr):
    def __init__(self, operator: st.alop.BinaryOperators, terms: List[Expr]):
        super().__init__()
        self.children = terms[:]
        self.op = operator


class If(Expr):
    _children = ['cnd', 'thn', 'els']
    cnd: Expr
    thn: Expr
    els: Expr

    def __init__(self, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)


class BinOp(Expr):
    _children = ['lhs', 'rhs']
    lhs: Expr
    rhs: Expr

    def __init__(self, operator: st.alop.BinaryOperators = None,
                 *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.operator = operator

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return str(self.operator)


class UnOp(Expr):
    _children = ['subexpr']
    subexpr: Expr

    def __init__(self, operator: st.alop.UnaryOperators = None,
                 *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.operator = operator

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return str(self.operator)


class IntLiteral(Expr):
    _attr = {'num_literal': True, 'num_const': True, 'atomic': True}

    def __init__(self, v: int):
        super().__init__()
        self.val = v

    def __int__(self):
        return self.val

    def __float__(self):
        return float(self.val)


class FloatLiteral(Expr):
    _attr = {'num_literal': True, 'num_const': True, 'atomic': True}

    def __init__(self, v: float):
        super().__init__()
        self.val = v

    def __float__(self):
        return self.val


class ConstRef(Expr):
    _attr = {'num_const': True, 'atomic': True}

    def __init__(self, v: str):
        super().__init__()
        self.val = v

    def str_attr(self):
        return self.val

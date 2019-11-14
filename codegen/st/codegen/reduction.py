from st.expr import Expr
from st.alop import BinaryOperators


class Reduction(Expr):
    op = {
        BinaryOperators.Add: BinaryOperators.Add,
        BinaryOperators.Sub: BinaryOperators.Add,
        BinaryOperators.Mul: BinaryOperators.Mul,
        BinaryOperators.Div: BinaryOperators.Mul,
        BinaryOperators.And: BinaryOperators.And,
        BinaryOperators.Or: BinaryOperators.Or,
    }
    identity = {
        BinaryOperators.Add: 0,
        BinaryOperators.Mul: 1,
        BinaryOperators.And: 1,
        BinaryOperators.Or: 0,
    }

    def __init__(self, op: BinaryOperators):
        super().__init__()
        self.operator = op
        self.terms_op = []

    def str_attr(self):
        """ Extra attributes that should be printed in str representation """
        return str(self.operator)

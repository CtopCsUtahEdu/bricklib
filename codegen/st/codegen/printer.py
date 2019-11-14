import st.expr
import st.func
from functools import singledispatch
from st.grid import GridRef
from io import StringIO


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class PrinterBase(metaclass=Singleton):
    def __init__(self):
        self.print = singledispatch(self._print)

    def print_str(self, node, prec=255):
        stream = StringIO()
        self.print(node, stream, prec)
        return stream.getvalue()

    def _print(self, node, stream: StringIO, prec=255):
        raise ValueError("Unhandled type {}".format(type(node).__name__))


class Printer(PrinterBase):
    def __init__(self):
        from st.alop import BinaryOperators as Bop
        from st.alop import UnaryOperators as Uop
        super().__init__()
        # https://en.cppreference.com/w/cpp/language/operator_precedence
        self.precedence = {
            # Binary operators
            Bop.Add: 6,
            Bop.Sub: 6,
            Bop.Mul: 5,
            Bop.Div: 5,
            Bop.Mod: 5,
            Bop.Eq: 10,
            Bop.Gt: 9,
            Bop.Lt: 9,
            Bop.Geq: 9,
            Bop.Leq: 9,
            Bop.Neq: 10,
            Bop.Or: 15,
            Bop.And: 14,
            Bop.BitAnd: 11,
            Bop.BitOr: 13,
            Bop.BitSHL: 7,
            Bop.BitSHR: 7,
            Bop.BitXor: 12,

            # Unary Operators
            Uop.Neg: 3,
            Uop.Pos: 3,
            Uop.Inc: 3,
            Uop.Dec: 3,
            Uop.Not: 3,
            Uop.BitNot: 3
        }
        self.print.register(st.expr.BinOp, self._print_binOp)
        self.print.register(st.expr.FloatLiteral, self._print_floatLiteral)
        self.print.register(st.expr.IntLiteral, self._print_intLiteral)
        self.print.register(st.expr.UnOp, self._print_unOp)
        self.print.register(st.expr.ConstRef, self._print_constRef)
        self.print.register(st.expr.If, self._print_if)
        self.print.register(st.func.CallExpr, self._print_callExpr)
        self.print.register(GridRef, self._print_gridref)

    def _print_binOp(self, node: st.expr.BinOp, stream: StringIO, prec=255):
        mprec = self.precedence[node.operator]
        if mprec > prec:
            stream.write("(")
        self.print(node.lhs, stream, mprec)
        stream.write(" {} ".format(node.operator.value))
        self.print(node.rhs, stream, mprec - 1)
        if mprec > prec:
            stream.write(")")

    def _print_floatLiteral(self, node: st.expr.FloatLiteral, stream: StringIO, prec=255):
        stream.write(str(node.val))

    def _print_intLiteral(self, node: st.expr.IntLiteral, stream: StringIO, prec=255):
        stream.write(str(node.val))

    def _print_unOp(self, node: st.expr.UnOp, stream: StringIO, prec=255):
        mprec = self.precedence[node.operator]
        if mprec > prec:
            stream.write("(")
        stream.write(node.operator.value)
        self.print(node.subexpr, stream, mprec - 1)
        if mprec > prec:
            stream.write(")")

    def _print_constRef(self, node: st.expr.ConstRef, stream: StringIO, prec=255):
        stream.write(node.val)

    def _print_if(self, node: st.expr.If, stream: StringIO, prec=255):
        mprec = 16
        if mprec > prec:
            stream.write("(")
        self.print(node.cnd, stream, mprec)
        stream.write(" ? ")
        self.print(node.thn, stream, mprec - 1)
        stream.write(" : ")
        self.print(node.els, stream, mprec - 1)
        if mprec > prec:
            stream.write(")")

    def _print_gridref(self, node: GridRef, stream: StringIO, prec=255):
        stream.write(node.grid.name)

    def _print_callExpr(self, call: st.func.CallExpr, stream: StringIO, prec=255):
        stream.write(call.callee.name)
        stream.write("(")
        for idx, c in enumerate(call.children):
            if idx > 0:
                stream.write(", ")
            self.print(c, stream, 255)
        stream.write(")")

if __name__ == '__main__':
    p = Printer()
    print(p.print(1 - (-st.expr.IntLiteral(1) + 1)))

"""Collection of algebra operators"""

from enum import Enum


class BinaryOperators(Enum):
    """Enumeration for binary operators in the AST

    Variations of assign if allowed are de-sugared before AST
    """
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    Mod = "%"
    Assign = "="
    Eq = "=="
    Gt = ">"
    Lt = "<"
    Geq = ">="
    Leq = "<="
    Neq = "!="
    Or = "||"
    And = "&&"
    BitAnd = "&"
    BitOr = "|"
    BitXor = "^"
    BitSHL = "<<"
    BitSHR = ">>"


class UnaryOperators(Enum):
    """Enumeration for unary operators in the AST"""
    Neg = "-"
    Pos = "+"
    Inc = "++"
    Dec = "--"
    Not = "!"
    BitNot = "~"

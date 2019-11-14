"""Here are the AST nodes contained for a simple stencil language"""

from inspect import Parameter, Signature


class ExprMeta(type):
    """Meta class for a generic AST node.
    Handles children generation and initialization.
    """

    def __new__(cls, name, bases, namespace):
        """Attach attribute to a new class from _children and _arg_sig"""
        clsobj = type.__new__(cls, name, bases, dict(namespace))
        children = getattr(clsobj, '_children', list())
        params = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, default=None)
                  for name in children]
        sig = Signature(params)
        setattr(clsobj, '_arg_sig', sig)

        def make_property(pos):
            def getter(self):
                return self.children[pos]

            def setter(self, child):
                self.children[pos] = child
                if child is not None:
                    child.parent = self

            return property(getter, setter)

        for idx, field in enumerate(children):
            setattr(clsobj, field, make_property(idx))

        return clsobj

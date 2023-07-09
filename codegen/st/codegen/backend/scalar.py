from st.codegen.backend.base import CodeBlock, Backend, Buffer, PrinterRed
from st.grid import Grid, GridRef
from st.expr import Expr
from typing import List

class PrinterScalar(PrinterRed):
    def __init__(self):
        super().__init__()
        self.print.register(GridRef, self._print_gridref)

    def _print_gridref(self, node: GridRef, stream, prec=255):
        import st.expr
        if self.codegen.ALIGNED:
            read_off = [o1 + o2 for o1, o2 in zip(node.offsets, self.shift)]
            stream.write("{}".format(Backend.vecreg_name(node.grid, read_off)))
        else:
            stream.write("{}(".format(node.grid.name))
            for n, (i, idx) in enumerate(zip(node.offsets, node.indices)):
                if n > 0:
                    stream.write(", ")
                comp = self.codegen.layout.TILE_NAME[idx.n]
                if self.dimrels and self.dimrels[-idx.n - 1]:
                    comp = comp + st.expr.ConstRef(self.dimrels[-idx.n - 1])
                offset = [0] * len(self.offset)
                for oid, o in zip(node.indices, node.offsets):
                    offset[oid.n] += o
                off = self.offset[idx.n] + self.shift[idx.n] + offset[idx.n]
                if off:
                    comp = comp + st.expr.IntLiteral(off)
                self.print(comp, stream)
            stream.write(")")

class BackendScalar(Backend):
    def __init__(self):
        super().__init__()
        self.VECLEN = 1
    
    def setCodeGen(self, codegen):
        self.codegen = codegen
        self.printer = PrinterScalar()
        self.printer.codegen = codegen

    def setLayout(self, layout):
        super().setLayout(layout)
        print("Using scalar backend, setting fold to 1")
        self.layout.fold = [1]

    def declare_buf(self, buf: Buffer, block: CodeBlock):
        space = 1
        for a, b in buf.iteration:
            space *= b - a
        block.append("bElem {}[{}];".format(buf.name, space))
        return buf.name

    def declare_gridref(self, grid: Grid, block: CodeBlock):
        name = self.gridref_name(grid)
        block.append("bElem *{} = &{};".format(
            name, self.layout.elem(grid, [0] * len(self.codegen.TILE_DIM))))
        block.append("__builtin_assume_aligned({}, 64);".format(name))
        return name

    def genVectorLoop(self, group: CodeBlock):
        return group

    def genStoreLoop(self, group: CodeBlock):
        group.append("for (long sti = 0; sti < {}; ++sti)".format(self.codegen.TILE_SIZE))

    def genStoreTileLoop(self, group: CodeBlock, dims):
        subblock = CodeBlock()
        group.append(subblock)
        subblock.append("long rel = 0;")
        for d in range(dims - 1, 0, -1):
            idx_name = self.layout.rel_name(d)
            subblock.append(
                "for (long {} = {}; {} < {}; {} += {})".format(
                    idx_name, 0, idx_name, self.codegen.TILE_DIM[d], idx_name, 1))
            newlevel = CodeBlock()
            subblock.append(newlevel)
            subblock = newlevel

        subblock.append("{} = 0;".format(self.layout.rel_name(0)))
        subblock.append("for (long sti = 0; sti < {}; ++sti, ++rel)".format(self.codegen.TILE_DIM[0]))
        return subblock

    def gen_lhs(self, buf: Buffer, offset: List[int], rel=None, dimrels=None):
        import st.expr
        roff = 0
        for idx, o in enumerate(reversed(offset)):
            idx = len(self.codegen.TILE_DIM) - idx - 1
            if idx < len(self.codegen.FOLD):
                roff *= self.codegen.TILE_DIM[idx] // self.codegen.FOLD[idx]
                roff += o // self.codegen.FOLD[idx]
            else:
                roff *= self.codegen.TILE_DIM[idx]
                roff += o
        roff *= self.VECLEN
        comp = st.expr.IntLiteral(roff)
        if rel:
            comp += rel
        ref = "{}[{}]".format(buf.name, self.printer.print_str(comp))
        return ref

    def gen_rhs(self, comp: Expr, shift: List[int], offset: List[int], rel=None, dimrels=None):
        """
        :param comp: The expression to print
        :param shift: The shift of scatter
        :param offset: Scattered from
        :param rel: Added offset when using loops
        :return:
        """
        self.printer.shift = shift[:]
        self.printer.offset = offset[:]
        self.printer.rel = None
        self.printer.dimrels = dimrels
        if rel:
            self.printer.rel = rel
        return self.printer.print_str(comp)

    def declare_reg(self, name, block: CodeBlock):
        block.append("bElem {};".format(name))

    def declare_vec(self, name, block: CodeBlock):
        block.append("bElem {};".format(name))

    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        block.append("{} = {};".format(reg_name, vecbuf_name))

    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        raise ValueError("No merge with scalar")

    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        if rel is not None:
            rel = [rel]
        block.append("{} = {};".format(
            name, self.layout.elem(grid, offset, rel)))

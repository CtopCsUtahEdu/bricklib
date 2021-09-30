from typing import List
from st.codegen.backend.base import Backend, PrinterRed, CodeBlock
from st.codegen.buffer import Buffer, BufferRead
from st.grid import Grid, GridRef
from st.expr import Expr


class BackendCUDA(Backend):
    LID = "threadIdx.x"

    def __init__(self, VECLEN=32, LID=None, ocl=False):
        super().__init__()
        self.VECLEN = VECLEN
        self.ocl = ocl
        if LID is not None:
            self.LID = LID

    def checkConfig(self):
        layout = self.layout
        tot = 1
        for l in layout.fold:
            tot *= l
        if self.VECLEN != tot:
            raise ValueError("Fold and vector length mismatch")
        num_vec = 1
        self.STRIDE = []
        for i, d in enumerate(layout.dim):
            self.STRIDE.append(num_vec)
            if i < len(layout.fold):
                if layout.fold[i] > layout.dim[i]:
                    raise ValueError("Fold and vector length mismatch")
                num_vec = num_vec * d // layout.fold[i]
            else:
                num_vec = num_vec * d

    def setCodeGen(self, codegen):
        self.codegen = codegen
        self.printer = PrinterCUDA(self.LID)
        self.printer.codegen = codegen

    def genVectorLoop(self, group: CodeBlock):
        return group

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
        self.printer.rel = rel
        self.printer.dimrels = dimrels
        return self.printer.print_str(comp)

    def declare_reg(self, name, block: CodeBlock):
        block.append("bElem {};".format(name))

    def declare_vec(self, name, block: CodeBlock):
        block.append("bElem {};".format(name))

    def declare_gridref(self, grid: Grid, block: CodeBlock):
        name = self.gridref_name(grid)
        block.append(("__global " if self.ocl else "") + "bElem *{} = &{};".format(
            name, self.layout.elem(grid, [0] * len(self.codegen.TILE_DIM))))
        return name

    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        block.append("{} = {};".format(reg_name, vecbuf_name))

    def declare_buf(self, buf: Buffer, block: CodeBlock):
        space = 1
        for a, b in buf.iteration:
            space *= b - a
        block.append("bElem {}[{}];".format(buf.name, space // self.VECLEN))
        return buf.name

    def genStoreLoop(self, group: CodeBlock):
        group.append("for (long sti = 0; sti < {}; ++sti)".format(self.codegen.TILE_SIZE // self.VECLEN))

    def storeTile(self, buf: Buffer, group: CodeBlock):
        dims = buf.grid.dims
        dimrels = [self.index_name(i) for i in reversed(range(dims))]
        group.append("{} = {}[rel];".format(self.layout.elem(buf.grid, [0] * dims, dimrels), buf.name))

    def genStoreTileLoop(self, group: CodeBlock, dims):
        subblock = CodeBlock()
        group.append(subblock)
        subblock.append("long rel = 0;")
        for d in range(dims - 1, 0, -1):
            idx_name = self.index_name(d)
            subblock.append(
                "for (long {} = {}; {} < {}; {} += {})".format(
                    idx_name, 0, idx_name, self.codegen.TILE_DIM[d], idx_name, 1))
            newlevel = CodeBlock()
            subblock.append(newlevel)
            subblock = newlevel

        rel = self.index_name(0)
        subblock.append("for (long {} = {}; {} < {}; {} += {}, ++rel)".format(
            rel, self.LID, rel, self.codegen.TILE_DIM[0], rel, self.VECLEN))
        return subblock

    def store(self, buf: Buffer, group: CodeBlock):
        group.append("{}[sti * {} + {}] = {}[sti];".format(
            self.gridref_name(buf.grid), self.VECLEN, self.LID, buf.name))

    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        import st.expr
        ref = [None] * len(offset)
        ref[-1] = st.expr.ConstRef(self.LID)
        if isinstance(rel, list):
            nrel = rel[:]
            if nrel[-1]:
                nrel[-1] += ref[-1]
            else:
                nrel[-1] = ref[-1]
            ref = nrel
        elif rel:
            ref[-1] = ref[-1] + rel * self.VECLEN
        block.append("{} = {};".format(
            name, self.layout.elem(grid, offset, ref)))

    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        block.append("// merge{} {} ,{}, {} -> {}".format(dim, regl, regr, shift, rego))
        l = 1
        for i in range(dim):
            l *= self.codegen.FOLD[i]
        ll = l * self.codegen.FOLD[dim]
        if ll == self.VECLEN:
            lid = self.LID
        else:
            lid = "{} & {}".format(self.LID, ll - 1)
        block.append("dev_shl({}, {}, {}, {}, {}, {});".format(
            rego, regl, regr, (self.codegen.FOLD[dim] - shift) * l, l * self.codegen.FOLD[dim], lid))


class PrinterCUDA(PrinterRed):
    def __init__(self, LID):
        super().__init__()
        self.LID = LID
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
                if idx.n == 0:
                    comp = comp + self.LID
                off = self.offset[idx.n] + self.shift[idx.n] + node.offsets[idx.n]
                if off:
                    comp = comp + st.expr.IntLiteral(off)
                self.print(comp, stream)
            stream.write(")")

    def _print_bufferread(self, node: BufferRead, stream, prec=255):
        import st.expr
        real_off = [o1 + o2 for o1, o2 in zip(self.offset, self.shift)]
        voff = 0
        for idx, o in enumerate(reversed(real_off)):
            idx = len(real_off) - idx - 1
            if idx < len(self.codegen.FOLD):
                voff *= (self.codegen.TILE_DIM[idx] // self.codegen.FOLD[idx])
                voff += o // self.codegen.FOLD[idx]
            else:
                voff *= self.codegen.TILE_DIM[idx]
                voff += o
        comp = st.expr.IntLiteral(voff)
        if self.rel:
            comp += self.rel
        stream.write("{}[".format(node.buf.name))
        self.print(comp, stream)
        stream.write("]")


class BackendCuFlex(BackendCUDA):
    def __init__(self):
        super().__init__()
        # Dummy
        self.VECLEN = 32

    def setLayout(self, layout):
        super().setLayout(layout)
        print("Using cuflex layout for tiled format, setting VECLEN to the size of last dimension")
        self.VECLEN = layout.dim[0]
        self.layout.fold = [self.VECLEN]

    def checkConfig(self):
        super().checkConfig()

    def declare_vec(self, name, block: CodeBlock):
        raise RuntimeError("No vector with flex")

    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        raise RuntimeError("No vector with flex")

    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        raise RuntimeError("No vector with flex")

    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        raise RuntimeError("No vector with flex")

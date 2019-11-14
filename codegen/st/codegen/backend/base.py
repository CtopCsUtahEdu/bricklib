from typing import List
from abc import abstractmethod
from st.codegen.printer import Printer
from st.codegen.buffer import Buffer, BufferRead
from st.codegen.reduction import Reduction
from st.grid import Grid, GridRef
from st.expr import Expr
from st.alop import BinaryOperators
import io

""" A generic (AVX512) backend.

A backend code generator should include:
* A code/codeblock recorder and printer
* A vector code declarator and blender
* A datalayout
    * changes only the read and write
* A printer
"""


class Code:
    @abstractmethod
    def to_str(self, ret: io.TextIOBase, n_ind: int):
        pass


class CodeBlock(Code):
    lines: List[Code]
    indent = "  "

    def __init__(self):
        self.lines = list()

    def to_str(self, ret: io.TextIOBase, n_ind: int):
        ret.write(CodeBlock.indent * n_ind + '{\n')
        for line in self.lines:
            line.to_str(ret, n_ind + 1)
        ret.write(CodeBlock.indent * n_ind + '}\n')

    def __str__(self):
        ret = io.StringIO()
        self.to_str(ret, 0)
        return ret.getvalue()

    def append(self, code):
        if isinstance(code, str):
            code = LineCode(code)
        self.lines.append(code)


class LineCode(Code):
    code: str

    def __init__(self, code=""):
        self.code = code

    def to_str(self, ret: io.TextIOBase, n_ind: int):
        ret.write(CodeBlock.indent * n_ind + self.code + '\n')


def genmask(fold, l, shift, scale, veclen):
    len = 1
    for dim in range(l):
        len *= fold[dim]

    zeros = shift * len * scale
    ones = fold[l] * len * scale
    repeat = (1 << ones) - (1 << zeros)
    len *= fold[l]
    mask = 0
    for i in range(veclen // len):
        mask = (mask << ones) + repeat
    return mask


class Brick:
    def __init__(self, *, fold=None, dim=None, prec=1, brick_idx="b", cstruct=False):
        self.backend = None
        self.codegen = None
        self.fold = fold
        self.dim = dim
        self.prec = prec
        self.BRICK_IDX = brick_idx
        self.cstruct = cstruct

    def setCodeGen(self, codegen):
        self.codegen = codegen

    def setBackend(self, backend):
        self.backend = backend

    def prologue(self, toplevel: CodeBlock):
        codegen = self.backend.codegen
        if self.cstruct:
            dims = len(codegen.TILE_DIM)
            neighbor = [-1] * dims
            while True:
                toplevel.append(
                    "unsigned {} = {};".format(self.neighbor(neighbor), self.neighbor_idx_cstruct(codegen.grids[0].name, self.BRICK_IDX, neighbor)))
                cur = 0
                while cur < dims and neighbor[cur] == 1:
                    neighbor[cur] = -1
                    cur += 1
                if cur < dims:
                    neighbor[cur] += 1
                else:
                    break
        else:
            dims = len(codegen.TILE_DIM)
            neighbor = [-1] * dims
            toplevel.append("auto *binfo = {}.bInfo;".format(codegen.grids[0].name))
            while True:
                toplevel.append(
                    "long {} = {};".format(self.neighbor(neighbor), self.neighbor_idx("binfo", self.BRICK_IDX, neighbor)))
                cur = 0
                while cur < dims and neighbor[cur] == 1:
                    neighbor[cur] = -1
                    cur += 1
                if cur < dims:
                    neighbor[cur] += 1
                else:
                    break

    def neighbor(self, offset: List[int]):
        val = self.neighbor_val(offset)
        return "neighbor{}".format(val)

    def neighbor_idx_cstruct(self, binfo, bidx, neighbor: List[int]):
        tot = int(3 ** len(self.backend.codegen.TILE_DIM))
        val = self.neighbor_val(neighbor)
        if val == tot // 2:
            return bidx
        else:
            return "{}.adj[{} * {} + {}]".format(binfo, bidx, tot, val)

    def neighbor_idx(self, binfo, bidx, neighbor: List[int]):
        tot = int(3 ** len(self.backend.codegen.TILE_DIM))
        val = self.neighbor_val(neighbor)
        if val == tot // 2:
            return bidx
        else:
            return "{}->adj[{}][{}]".format(binfo, bidx, val)

    def neighbor_val(self, offset: List[int]):
        if offset is not None:
            val = 0
            for o in reversed(offset):
                val = val * 3 + (o + 1)
        else:
            tot = int(3 ** len(self.backend.codegen.TILE_DIM))
            val = tot // 2
        return val

    def vecstart(self, grid: Grid, offset: List[int], rel=None):
        codegen = self.backend.codegen
        neighbor = [o // t for o, t in zip(offset, codegen.TILE_DIM)]
        nei_idx = [o % t for o, t in zip(offset, codegen.TILE_DIM)]
        roff = 0
        for idx, o in enumerate(reversed(nei_idx)):
            idx = len(codegen.TILE_DIM) - idx - 1
            if idx < len(codegen.FOLD):
                roff *= codegen.TILE_DIM[idx] // codegen.FOLD[idx]
                roff += o // codegen.FOLD[idx]
            else:
                roff *= codegen.TILE_DIM[idx]
                roff += o
        roff *= self.backend.VECLEN
        nname = self.neighbor(neighbor)
        import st.expr
        comp = st.expr.ConstRef("{} * {}.step".format(nname, grid.name))
        if roff != 0:
            comp += roff
        if rel:
            comp += rel[-1]

        return self.backend.printer.print_str(comp)

    def checkConfig(self):
        for idx, i in enumerate(self.fold):
            if self.dim[idx] % i != 0:
                raise ValueError("Wrong combination of fold and tile")
        pass

    def elem(self, grid: Grid, offset: List[int], rel=None):
        return "{}.dat[{}]".format(grid.name, self.vecstart(grid, offset, rel))


class Tiled(Brick):
    def __init__(self, *, tile_iter=None, aligned=True, **kwargs):
        super().__init__(**kwargs)
        self.TILE_NAME = tile_iter
        self.aligned = aligned

    def setCodeGen(self, codegen):
        super().setCodeGen(codegen)
        codegen.ALIGNED = self.aligned
        codegen.TILE = True

    def setBackend(self, backend):
        super().setBackend(backend)
        if self.fold is not None:
            print("Fold is set to VECLEN for Tiled")
        self.fold = [backend.VECLEN]
        backend.ALIGNED = self.aligned

    def prologue(self, toplevel: CodeBlock):
        from st.expr import ConstRef
        self.TILE_NAME = [ConstRef(i) for i in self.TILE_NAME]
        self.backend.codegen.LAYOUTREL = True

    def checkConfig(self):
        if len(self.fold) > 1:
            raise ValueError("Cannot perform vector folding with regular array format")

    def elem(self, grid: Grid, offset: List[int], rel=None):
        from io import StringIO
        stream = StringIO()
        stream.write("{}(".format(grid.name))
        from st.expr import ConstRef
        for idx, pos in enumerate(offset):
            npos = self.TILE_NAME[idx]
            if pos > 0:
                npos += pos
            elif pos < 0:
                npos -= abs(pos)
            if rel is not None and rel[-idx - 1] is not None:
                npos += rel[-idx - 1]
            if idx > 0:
                stream.write(", ")
            self.backend.printer.print(npos, stream)
        stream.write(")")
        return stream.getvalue()

    def stride(self, dim):
        if dim == 0:
            return self.backend.VECLEN
        else:
            return 1


class Backend:
    preffix = "_cg"

    def __init__(self):
        self.printer = None
        self.codegen = None
        self.prec = 1
        self.layout = None
        self.STRIDE = []
        self.VECLEN = 1
        self.ALIGNED = True

    def setCodeGen(self, codegen):
        self.codegen = codegen
        self.printer = PrinterRed()
        self.printer.codegen = codegen

    def setLayout(self, layout):
        self.layout = layout
        self.prec = self.layout.prec

    def prequel(self, toplevel):
        self.layout.prologue(toplevel)

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
            self.STRIDE.append(num_vec * self.VECLEN)
            if i < len(layout.fold):
                if layout.fold[i] > layout.dim[i]:
                    raise ValueError("Fold and vector length mismatch")
                num_vec = num_vec * d // layout.fold[i]
            else:
                num_vec = num_vec * d

    @abstractmethod
    def declare_buf(self, buf: Buffer, block: CodeBlock):
        pass

    @abstractmethod
    def declare_gridref(self, grid: Grid, block: CodeBlock):
        pass

    @staticmethod
    def gridref_name(grid):
        import re
        return re.sub(r'[\[\]() \n\t-]', '_', grid.name) + "_ref"

    @staticmethod
    def vectmp_name(idx):
        return "{}_vectmp{}".format(Backend.preffix, idx)

    @staticmethod
    def vecbuf_name(grid, vec_shift):
        import re
        name = grid.name
        name += str(vec_shift[0])
        for s in vec_shift[1:]:
            name += str(s)
        name = re.sub(r'[\[\]() \n\t-]', '_', name) + "_vecbuf"
        return "{}_{}".format(Backend.preffix, name)

    @staticmethod
    def vecreg_name(grid, vec_shift):
        import re
        name = grid.name
        name += str(vec_shift[0])
        for s in vec_shift[1:]:
            name += str(s)
        name = re.sub(r'[\[\]() \n\t-]', '_', name) + "_reg"
        return "{}_{}".format(Backend.preffix, name)

    @staticmethod
    def index_name(idx):
        return "{}_idx{}".format(Backend.preffix, idx)

    @staticmethod
    def rel_name(idx=None):
        if idx is None:
            return "rel"
        return "{}_rel{}".format(Backend.preffix, idx)

    def stride(self, dim):
        return self.STRIDE[dim]

    def declare_vecbuf(self, grid: Grid, vec_shift, block: CodeBlock):
        self.declare_vec(self.vecbuf_name(grid, vec_shift), block)

    @abstractmethod
    def genVectorLoop(self, group: CodeBlock):
        pass

    @abstractmethod
    def genStoreLoop(self, group: CodeBlock):
        pass

    def store(self, buf: Buffer, group: CodeBlock):
        group.append("{}[sti] = {}[sti];".format(self.gridref_name(buf.grid), buf.name))

    def storeTile(self, buf: Buffer, group: CodeBlock):
        dims = buf.grid.dims
        dimrels = [self.index_name(i) for i in reversed(range(dims))]
        group.append("{} = {}[rel + vit];".format(
            self.genStoreLoc(buf.grid, [0] * dims, [0] * dims, None, dimrels), buf.name))

    def genStoreLoc(self, grid: Grid, shift, offset, rel, dimrels):
        from st.grid import GridRef
        from st.expr import Index
        dims = grid.dims
        return self.gen_rhs(GridRef(grid, [Index(i) for i in range(dims)]), shift, offset, rel, dimrels)

    def genStoreTileLoop(self, group: CodeBlock, dims):
        subblock = CodeBlock()
        group.append(subblock)
        subblock.append("long rel = 0;")
        for d in range(dims - 1, 0, -1):
            idx_name = self.index_name(d)
            if d == 1:
                subblock.append(
                    "for (long {} = 0; {} < {}; {} += {}, rel += {})".format(
                        idx_name, idx_name, self.codegen.TILE_DIM[d], idx_name, 1, self.codegen.TILE_DIM[0]))
            else:
                subblock.append(
                    "for (long {} = 0; {} < {}; {} += {})".format(
                        idx_name, idx_name, self.codegen.TILE_DIM[d], idx_name, 1, self.codegen.TILE_DIM[0]))
            newlevel = CodeBlock()
            subblock.append(newlevel)
            subblock = newlevel

        subblock.append("long {} = 0;".format(self.index_name(0)))
        subblock.append("#pragma omp simd")
        subblock.append("for (long vit = 0; vit < {}; ++vit)".format(self.codegen.TILE_DIM[0]))
        return subblock

    @abstractmethod
    def gen_lhs(self, buf: Buffer, offset: List[int], rel=None, dimrels=None):
        pass

    @abstractmethod
    def gen_rhs(self, comp: Expr, shift: List[int], offset: List[int], rel=None, dimrels=None):
        """
        :param comp: The expression to print
        :param shift: The shift of scatter
        :param offset: Scattered from
        :param rel: Added offset when using loops
        :return:
        """
        pass

    @abstractmethod
    def declare_reg(self, name, block: CodeBlock):
        pass

    @abstractmethod
    def declare_vec(self, name, block: CodeBlock):
        pass

    @abstractmethod
    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        pass

    @abstractmethod
    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        pass

    @abstractmethod
    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        pass


class PrinterRed(Printer):
    def __init__(self):
        super().__init__()
        self.print.register(Reduction, self._print_reduction)
        self.print.register(GridRef, self._print_gridref)
        self.print.register(BufferRead, self._print_bufferread)
        self.shift = []
        self.offset = []
        self.rel = None
        self.dimrels = None
        self.codegen = None

    def _print_reduction(self, node: Reduction, stream, prec=255):
        mprec = self.precedence[node.operator]
        if mprec > prec:
            stream.write("(")
        if node.terms_op[0] == BinaryOperators.Div:
            stream.write("1 / ")
        if node.terms_op[0] == BinaryOperators.Sub:
            stream.write("-")
        self.print(node.children[0], stream, mprec)
        for idx, child in enumerate(node.children[1:]):
            stream.write(" {} ".format(node.terms_op[idx + 1].value))
            self.print(child, stream, mprec - 1)
        if mprec > prec:
            stream.write(")")

    def _print_gridref(self, node: GridRef, stream, prec=255):
        import st.expr
        if self.codegen.ALIGNED:
            read_off = [o1 + o2 for o1, o2 in zip(node.offsets, self.shift)]
            stream.write("{}[".format(Backend.vecreg_name(node.grid, read_off)))
            comp = st.expr.ConstRef('vit')
            self.print(comp, stream)
            stream.write("]")
        else:
            stream.write("{}(".format(node.grid.name))
            for n, (i, idx) in enumerate(zip(node.offsets, node.indices)):
                if n > 0:
                    stream.write(", ")
                comp = self.codegen.layout.TILE_NAME[idx.n]
                if self.dimrels and self.dimrels[-idx.n - 1]:
                    comp = comp + st.expr.ConstRef(self.dimrels[-idx.n - 1])
                if idx.n == 0:
                    comp = comp + st.expr.ConstRef('vit')
                offset = [0] * len(self.offset)
                for oid, o in zip(node.indices, node.offsets):
                    offset[oid.n] += o
                off = self.offset[idx.n] + self.shift[idx.n] + offset[idx.n]
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
        voff *= self.codegen.backend.VECLEN
        comp = st.expr.IntLiteral(voff)
        if self.rel:
            comp += self.rel
        stream.write("{}[".format(node.buf.name))
        self.print(comp, stream)
        stream.write("]")

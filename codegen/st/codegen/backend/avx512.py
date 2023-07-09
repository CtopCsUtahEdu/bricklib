from st.grid import Grid
from st.codegen.backend.base import PrinterRed, CodeBlock, Backend
from st.codegen.buffer import Buffer
from st.expr import Expr
from typing import List


def genmask(fold, l, shift, scale, veclen):
    len = 1
    for dim in range(l):
        len *= fold[dim]

    ones = (fold[l] - shift) * len * scale
    rep = fold[l] * len * scale
    repeat = (1 << ones) - 1
    len *= fold[l]
    mask = 0
    for i in range(veclen // len):
        mask = (mask << rep) + repeat
    return mask


class BackendAVX512(Backend):
    def __init__(self):
        super().__init__()
        self.VECLEN = 16

    def setLayout(self, layout):
        super().setLayout(layout)
        self.VECLEN //= layout.prec

    def prequel(self, toplevel):
        self.layout.prologue(toplevel)

    def declare_buf(self, buf: Buffer, block: CodeBlock):
        space = 1
        for a, b in buf.iteration:
            space *= b - a
        align = self.layout.prec * 4 * space
        align = 64 if align >= 64 else align
        block.append("bElem {}[{}] __attribute__((aligned({})));".format(buf.name, space, align))
        return buf.name

    def declare_gridref(self, grid: Grid, block: CodeBlock):
        name = self.gridref_name(grid)
        block.append("bElem *{} = &{};".format(
            name, self.layout.elem(grid, [0] * len(self.codegen.TILE_DIM))))
        block.append("{} = (bElem *)__builtin_assume_aligned({}, 64);".format(name, name))
        return name

    def genVectorLoop(self, group: CodeBlock):
        group.append("#pragma omp simd")
        group.append("for (long vit = 0; vit < {}; ++vit)".format(self.VECLEN))
        g = CodeBlock()
        group.append(g)
        return g

    def genStoreLoop(self, group: CodeBlock):
        group.append("#pragma omp simd")
        group.append("for (long sti = 0; sti < {}; ++sti)".format(self.codegen.TILE_SIZE))

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
        comp = roff + st.expr.ConstRef('vit')
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
        from st.expr import ConstRef
        self.printer.shift = shift[:]
        self.printer.offset = offset[:]
        self.printer.rel = ConstRef("vit")
        self.printer.dimrels = dimrels
        if rel:
            self.printer.rel += rel
        return self.printer.print_str(comp)

    def declare_reg(self, name, block: CodeBlock):
        align = self.layout.prec * 4 * self.VECLEN
        align = 64 if align >= 64 else align
        block.append("bElem {}[{}] __attribute__((aligned({})));".format(name, self.VECLEN, align))

    def declare_vec(self, name, block: CodeBlock):
        block.append("__m512i {};".format(name))

    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        block.append("_mm512_store_epi32( & {}[0], {});".format(reg_name, vecbuf_name))

    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        block.append("// merge{} {} ,{}, {} -> {}".format(dim, regl, regr, shift, rego))
        l = 1

        for i in range(dim):
            l *= self.codegen.FOLD[i]
        if l * self.codegen.FOLD[dim] == self.VECLEN:
            # this only requires a shift
            sh = shift * l * self.prec
            block.append("{} = _mm512_alignr_epi32({}, {}, {});".format(rego, regr, regl, sh))
        else:
            # this requires masking
            sh = (self.VECLEN - (self.codegen.FOLD[dim] - shift) * l) * self.prec
            block.append("{} = _mm512_alignr_epi32({}, {}, {});".format(
                rego, regr, regr, sh))
            sh = shift * l * self.prec
            mask = genmask(self.codegen.FOLD, dim, shift, self.prec, self.VECLEN)
            block.append("{} = _mm512_mask_alignr_epi32({}, {}, {}, {}, {});".format(
                rego, rego, mask, regl, regl, sh))

    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        block.append("// read {} -> {}".format(str(offset), name))
        if rel is not None:
            rel = [rel]
        block.append("{} = _mm512_load_epi32(& {});".format(
            name, self.layout.elem(grid, offset, rel)))

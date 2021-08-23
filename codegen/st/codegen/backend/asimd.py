from st.codegen.backend.base import PrinterRed, CodeBlock
from st.codegen.backend.avx512 import BackendAVX512
from st.grid import Grid


class BackendASIMD(BackendAVX512):
    def __init__(self):
        super().__init__()
        self.VECLEN = 4

    def checkConfig(self):
        super().checkConfig()
        layout = self.layout
        if tuple(layout.fold) != (self.VECLEN,):
            raise ValueError("Fold and vector type (ASIMD) mismatch")

    def declare_vec(self, name, block: CodeBlock):
        block.append("uint32x4_t {};".format(name))

    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        block.append("vst1q_u32((uint32_t *) & {}[0], {});".format(reg_name, vecbuf_name))

    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        block.append("// read {} -> {}".format(str(offset), name))
        if rel is not None:
            rel = [rel]
        block.append("{} = vld1q_u32((uint32_t *) & {});".format(
            name, self.layout.elem(grid, offset, rel)))

    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        block.append("// merge{} {} ,{}, {} -> {}".format(dim, regl, regr, shift, rego))
        if dim > 0:
            raise RuntimeError("Cannot merge on dimension {} for ASIMD".format(dim))
        block.append("{} = vextq_u32({}, {}, {});".format(rego, regl, regr, shift * self.prec))

class BackendSVE(BackendAVX512):
    def __init__(self, length):
        super().__init__()
        self.VECLEN = length//32

    def checkConfig(self):
        super().checkConfig()
        layout = self.layout
        if tuple(layout.fold) != (self.VECLEN,):
            raise ValueError("Fold and vector type (ASIMD) mismatch")

    def declare_vec(self, name, block: CodeBlock):
        block.append("svuint32_t {};".format(name))

    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        block.append("svst1_u32(svptrue_b32(), (uint32_t *) & {}[0], {});".format(reg_name, vecbuf_name))

    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        block.append("// read {} -> {}".format(str(offset), name))
        if rel is not None:
            rel = [rel]
        block.append("{} = svld1_u32(svptrue_b32(), (uint32_t *) & {});".format(
            name, self.layout.elem(grid, offset, rel)))

    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        block.append("// merge{} {} ,{}, {} -> {}".format(dim, regl, regr, shift, rego))
        if dim > 0:
            raise RuntimeError("Cannot merge on dimension {} for SVE".format(dim))
        block.append("{} = svext_u32({}, {}, {});".format(rego, regl, regr, shift * self.prec))


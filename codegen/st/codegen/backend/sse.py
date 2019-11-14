from st.codegen.backend.base import PrinterRed, CodeBlock
from st.codegen.backend.avx512 import BackendAVX512
from st.grid import Grid


class BackendSSE(BackendAVX512):
    def __init__(self):
        super().__init__()
        self.VECLEN = 4

    def checkConfig(self):
        super().checkConfig()
        layout = self.layout
        if self.codegen.ALIGNED:
            if (tuple(layout.fold) != (4,) and self.prec == 1) or (
                    tuple(layout.fold) != (2,) and self.prec == 2):
                raise ValueError("Fold and vector type (AVX2) mismatch")
        else:
            if tuple(layout.fold) != (self.VECLEN,):
                raise ValueError("Fold and vector type (AVX2) mismatch")

    def declare_vec(self, name, block: CodeBlock):
        block.append("__m128i {};".format(name))

    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        block.append("_mm_store_si128((__m128i *) & {}[0], {});".format(reg_name, vecbuf_name))

    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        block.append("// read {} -> {}".format(str(offset), name))
        if rel is not None:
            rel = [rel]
        block.append("{} = _mm_load_si128((__m128i *) & {});".format(
            name, self.layout.elem(grid, offset, rel)))

    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        block.append("// merge{} {} ,{}, {} -> {}".format(dim, regl, regr, shift, rego))
        if dim > 0:
            raise RuntimeError("Cannot merge on dimension {} for SSE".format(dim))
        block.append("{} = _mm_alignr_epi8({}, {}, {});".format(rego, regr, regl, shift * self.prec * 4))

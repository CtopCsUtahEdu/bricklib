from st.codegen.backend.base import PrinterRed, CodeBlock
from st.codegen.backend.avx512 import BackendAVX512
from st.grid import Grid


class BackendAVX2(BackendAVX512):
    def __init__(self):
        super().__init__()
        self.VECLEN = 8

    def checkConfig(self):
        super().checkConfig()
        layout = self.layout
        if self.codegen.ALIGNED:
            if (tuple(layout.fold) != (4, 2) and self.prec == 1) or (
                    tuple(layout.fold) != (2, 2) and self.prec == 2):
                raise ValueError("Fold and vector type (AVX2) mismatch")
        else:
            if tuple(layout.fold) != (self.VECLEN,):
                raise ValueError("Fold and vector type (AVX2) mismatch")

    def declare_vec(self, name, block: CodeBlock):
        block.append("__m256i {};".format(name))

    def store_vecbuf(self, vecbuf_name, reg_name, block: CodeBlock):
        block.append("_mm256_store_si256((__m256i *) & {}[0], {});".format(reg_name, vecbuf_name))

    def read_aligned(self, grid: Grid, offset, name: str, block: CodeBlock, rel=None):
        block.append("// read {} -> {}".format(str(offset), name))
        if rel is not None:
            rel = [rel]
        block.append("{} = _mm256_load_si256((__m256i *) & {});".format(
            name, self.layout.elem(grid, offset, rel)))

    def merge(self, rego, regl, regr, dim, shift, block: CodeBlock):
        block.append("// merge{} {} ,{}, {} -> {}".format(dim, regl, regr, shift, rego))
        if dim > 1:
            raise RuntimeError("Cannot merge on dimension {} for AVX2".format(dim))
        if dim == 1:
            block.append("{} = _mm256_permute2x128_si256({}, {}, 3);".format(rego, regr, regl))
        elif dim == 0:
            block.append("{} = _mm256_alignr_epi8({}, {}, {});".format(rego, regr, regl, shift * self.prec * 4))

from st.codegen.backend.base import PrinterRed, CodeBlock
from st.codegen.backend.avx512 import BackendAVX512
from st.grid import Grid


class BackendFlex(BackendAVX512):
    def __init__(self):
        super().__init__()
        # Dummy
        self.VECLEN = 16

    def setLayout(self, layout):
        super().setLayout(layout)
        print("Using flex layout for tiled format, setting VECLEN to the size of last dimension")
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

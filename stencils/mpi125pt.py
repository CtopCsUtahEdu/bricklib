from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("in", 3)
output = Grid("out", 3)
# Symmetries of 125pt stencil imply 10 constants, permuted +/-
# 0 0 0 - 1 of these
a0 = ConstRef("MPI_B0")
# 0 0 1 - 6 of these
a1 = ConstRef("MPI_B1")
# 0 0 2 - 6 of these
a2 = ConstRef("MPI_B2")
# 0 1 1 - 12 of these
a3 = ConstRef("MPI_B3")
# 0 1 2 - 24 of these
a4 = ConstRef("MPI_B4")
# 0 2 2 - 12 of these
a5 = ConstRef("MPI_B5")
# 1 1 1 - 8 of these
a6 = ConstRef("MPI_B6")
# 1 1 2 - 24 of these
a7 = ConstRef("MPI_B7")
# 1 2 2 - 24 of these
a8 = ConstRef("MPI_B8")
# 2 2 2 - 8 of these
a9 = ConstRef("MPI_B9")

# Express computation
# output[i, j, k] is assumed
calc = \
       a0 * input(i, j, k) + \
       a1 * input(i + 1, j, k) + \
       a1 * input(i - 1, j, k) + \
       a1 * input(i, j + 1, k) + \
       a1 * input(i, j - 1, k) + \
       a1 * input(i, j, k + 1) + \
       a1 * input(i, j, k - 1) + \
       a2 * input(i + 2, j, k) + \
       a2 * input(i - 2, j, k) + \
       a2 * input(i, j + 2, k) + \
       a2 * input(i, j - 2, k) + \
       a2 * input(i, j, k + 2) + \
       a2 * input(i, j, k - 2) + \
       a3 * input(i + 1, j + 1, k) + \
       a3 * input(i - 1, j + 1, k) + \
       a3 * input(i + 1, j - 1, k) + \
       a3 * input(i - 1, j - 1, k) + \
       a3 * input(i + 1, j, k + 1) + \
       a3 * input(i - 1, j, k + 1) + \
       a3 * input(i + 1, j, k - 1) + \
       a3 * input(i - 1, j, k - 1) + \
       a3 * input(i, j + 1, k + 1) + \
       a3 * input(i, j - 1, k + 1) + \
       a3 * input(i, j + 1, k - 1) + \
       a3 * input(i, j - 1, k - 1) + \
       a4 * input(i + 1, j + 2, k) + \
       a4 * input(i - 1, j + 2, k) + \
       a4 * input(i + 1, j - 2, k) + \
       a4 * input(i - 1, j - 2, k) + \
       a4 * input(i + 1, j, k + 2) + \
       a4 * input(i - 1, j, k + 2) + \
       a4 * input(i + 1, j, k - 2) + \
       a4 * input(i - 1, j, k - 2) + \
       a4 * input(i, j + 1, k + 2) + \
       a4 * input(i, j - 1, k + 2) + \
       a4 * input(i, j + 1, k - 2) + \
       a4 * input(i, j - 1, k - 2) + \
       a4 * input(i + 2, j + 1, k) + \
       a4 * input(i - 2, j + 1, k) + \
       a4 * input(i + 2, j - 1, k) + \
       a4 * input(i - 2, j - 1, k) + \
       a4 * input(i + 2, j, k + 1) + \
       a4 * input(i - 2, j, k + 1) + \
       a4 * input(i + 2, j, k - 1) + \
       a4 * input(i - 2, j, k - 1) + \
       a4 * input(i, j + 2, k + 1) + \
       a4 * input(i, j - 2, k + 1) + \
       a4 * input(i, j + 2, k - 1) + \
       a4 * input(i, j - 2, k - 1) + \
       a5 * input(i + 2, j + 2, k) + \
       a5 * input(i - 2, j + 2, k) + \
       a5 * input(i + 2, j - 2, k) + \
       a5 * input(i - 2, j - 2, k) + \
       a5 * input(i + 2, j, k + 2) + \
       a5 * input(i - 2, j, k + 2) + \
       a5 * input(i + 2, j, k - 2) + \
       a5 * input(i - 2, j, k - 2) + \
       a5 * input(i, j + 2, k + 2) + \
       a5 * input(i, j - 2, k + 2) + \
       a5 * input(i, j + 2, k - 2) + \
       a5 * input(i, j - 2, k - 2) + \
       a6 * input(i + 1, j + 1, k + 1) + \
       a6 * input(i - 1, j + 1, k + 1) + \
       a6 * input(i + 1, j - 1, k + 1) + \
       a6 * input(i - 1, j - 1, k + 1) + \
       a6 * input(i + 1, j + 1, k - 1) + \
       a6 * input(i - 1, j + 1, k - 1) + \
       a6 * input(i + 1, j - 1, k - 1) + \
       a6 * input(i - 1, j - 1, k - 1) + \
       a7 * input(i + 1, j + 1, k + 2) + \
       a7 * input(i - 1, j + 1, k + 2) + \
       a7 * input(i + 1, j - 1, k + 2) + \
       a7 * input(i - 1, j - 1, k + 2) + \
       a7 * input(i + 1, j + 1, k - 2) + \
       a7 * input(i - 1, j + 1, k - 2) + \
       a7 * input(i + 1, j - 1, k - 2) + \
       a7 * input(i - 1, j - 1, k - 2) + \
       a7 * input(i + 1, j + 2, k + 1) + \
       a7 * input(i - 1, j + 2, k + 1) + \
       a7 * input(i + 1, j - 2, k + 1) + \
       a7 * input(i - 1, j - 2, k + 1) + \
       a7 * input(i + 1, j + 2, k - 1) + \
       a7 * input(i - 1, j + 2, k - 1) + \
       a7 * input(i + 1, j - 2, k - 1) + \
       a7 * input(i - 1, j - 2, k - 1) + \
       a7 * input(i + 2, j + 1, k + 1) + \
       a7 * input(i - 2, j + 1, k + 1) + \
       a7 * input(i + 2, j - 1, k + 1) + \
       a7 * input(i - 2, j - 1, k + 1) + \
       a7 * input(i + 2, j + 1, k - 1) + \
       a7 * input(i - 2, j + 1, k - 1) + \
       a7 * input(i + 2, j - 1, k - 1) + \
       a7 * input(i - 2, j - 1, k - 1) + \
       a8 * input(i + 2, j + 2, k + 1) + \
       a8 * input(i - 2, j + 2, k + 1) + \
       a8 * input(i + 2, j - 2, k + 1) + \
       a8 * input(i - 2, j - 2, k + 1) + \
       a8 * input(i + 2, j + 2, k - 1) + \
       a8 * input(i - 2, j + 2, k - 1) + \
       a8 * input(i + 2, j - 2, k - 1) + \
       a8 * input(i - 2, j - 2, k - 1) + \
       a8 * input(i + 2, j + 1, k + 2) + \
       a8 * input(i - 2, j + 1, k + 2) + \
       a8 * input(i + 2, j - 1, k + 2) + \
       a8 * input(i - 2, j - 1, k + 2) + \
       a8 * input(i + 2, j + 1, k - 2) + \
       a8 * input(i - 2, j + 1, k - 2) + \
       a8 * input(i + 2, j - 1, k - 2) + \
       a8 * input(i - 2, j - 1, k - 2) + \
       a8 * input(i + 1, j + 2, k + 2) + \
       a8 * input(i - 1, j + 2, k + 2) + \
       a8 * input(i + 1, j - 2, k + 2) + \
       a8 * input(i - 1, j - 2, k + 2) + \
       a8 * input(i + 1, j + 2, k - 2) + \
       a8 * input(i - 1, j + 2, k - 2) + \
       a8 * input(i + 1, j - 2, k - 2) + \
       a8 * input(i - 1, j - 2, k - 2) + \
       a9 * input(i + 2, j + 2, k + 2) + \
       a9 * input(i - 2, j + 2, k + 2) + \
       a9 * input(i + 2, j - 2, k + 2) + \
       a9 * input(i - 2, j - 2, k + 2) + \
       a9 * input(i + 2, j + 2, k - 2) + \
       a9 * input(i - 2, j + 2, k - 2) + \
       a9 * input(i + 2, j - 2, k - 2) + \
       a9 * input(i - 2, j - 2, k - 2)
output(i, j, k).assign(calc)

STENCIL = [output]

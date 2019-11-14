from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("in", 3)
output = Grid("out", 3)
a0 = ConstRef("MPI_B0")
a1 = ConstRef("MPI_B1")
a2 = ConstRef("MPI_B2")

# Express computation
# output[i, j, k] is assumed
calc = a0 * input(i, j, k) + \
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
       a2 * input(i, j, k - 2)
output(i, j, k).assign(calc)

STENCIL = [output]

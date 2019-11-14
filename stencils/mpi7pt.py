from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("in", 3)
output = Grid("out", 3)
alpha = ConstRef("MPI_ALPHA")
beta = ConstRef("MPI_BETA")

# Express computation
# output[i, j, k] is assumed
calc = alpha * input(i, j, k) + \
       beta * input(i + 1, j, k) + \
       beta * input(i - 1, j, k) + \
       beta * input(i, j + 1, k) + \
       beta * input(i, j - 1, k) + \
       beta * input(i, j, k + 1) + \
       beta * input(i, j, k - 1)
output(i, j, k).assign(calc)

STENCIL = [output]

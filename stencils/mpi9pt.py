from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)
l = Index(3)

# Declare grid
input = Grid("in", 4)
output = Grid("out", 4)
alpha = ConstRef("0.2")
beta = ConstRef("0.1")

# Express computation
# output[i, j, k, l] is assumed
calc = alpha * input(i, j, k, l) + \
       beta * input(i + 1, j, k, l) + \
       beta * input(i - 1, j, k, l) + \
       beta * input(i, j + 1, k, l) + \
       beta * input(i, j - 1, k, l) + \
       beta * input(i, j, k + 1, l) + \
       beta * input(i, j, k - 1, l) + \
       beta * input(i, j, k, l + 1) + \
       beta * input(i, j, k, l - 1)
output(i, j, k, l).assign(calc)

STENCIL = [output]

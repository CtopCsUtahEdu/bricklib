from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)
param = [ConstRef("coeff[0]"), ConstRef("coeff[1]"),
         ConstRef("coeff[2]"), ConstRef("coeff[3]"),
         ConstRef("coeff[4]"), ConstRef("coeff[5]"),
         ConstRef("coeff[6]")]

# Express computation
# output[i, j, k] is assumed
calc = param[0] * input(i, j, k) + \
       param[1] * input(i + 1, j, k) + \
       param[2] * input(i - 1, j, k) + \
       param[3] * input(i, j + 1, k) + \
       param[4] * input(i, j - 1, k) + \
       param[5] * input(i, j, k + 1) + \
       param[6] * input(i, j, k - 1)
output(i, j, k).assign(calc)

STENCIL = [output]

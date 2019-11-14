from st.expr import Index, ConstRef, If
from st.grid import Grid
from st.func import Func

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

maxfunc = Func("max", 2)

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)
param = [ConstRef("coeff[0]"), ConstRef("coeff[1]"),
         ConstRef("coeff[2]"), ConstRef("coeff[3]"),
         ConstRef("coeff[4]"), ConstRef("coeff[5]"),
         ConstRef("coeff[6]")]
zero = ConstRef("0.0")

# Express computation
# output[i, j, k] is assumed
calc = param[0] * maxfunc(input(i, j, k), zero) + \
       param[1] * maxfunc(input(i + 1, j, k), zero) + \
       param[2] * maxfunc(input(i - 1, j, k), zero) + \
       param[3] * maxfunc(input(i, j + 1, k), zero) + \
       param[4] * maxfunc(input(i, j - 1, k), zero) + \
       param[5] * maxfunc(input(i, j, k + 1), zero) + \
       param[6] * maxfunc(input(i, j, k - 1), zero)
calc = If(calc > 0, calc, -calc)

output(i, j, k).assign(calc)

STENCIL = [output]

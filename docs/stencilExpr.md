# A Stencil Expression Language

## Purpose

This language is to capture the capability and test the various aspect of the
"minimal" description for a stencil computation. We plan to support algebraic
operations, transcendental intrinsics, and basic conditionals.

## Design

Computation description composes of:

* Axis index declaration
* Grid (one-/multi-dimensional) declaration
* Computation graph declaration

Code generation targets:

* C++/OpenMP
* CUDA
* Brick (transformed)

## API Details

### Computation description

#### Axis index declaration

Axis index is declared to be contiguous in the 0-th direction then 1-st, 2-nd, etc.

~~~python
index = Index(n:Int)
~~~

Index class can be combined to form index expression

~~~python
idx_expr = f(index)
~~~

#### Grid declaration

Grid is a multi-dimensional "conceptual" array and stored in memory according to
the contiguous-ness of the axis index.

~~~python
grid = Grid(backend_name: str, dimensions: int)
~~~

#### Computation graph declaration

Computations are constructed with class method and operator overloading with
Expression class.

~~~python
expr = g(grid_in(f0(index_i0),...,fn(index_in)))
grid_out.assign(expr)
~~~

#### Code generation

Code generation is carried out by first specifying the backend and transform the code into the appropriate abstraction

~~~python
abstraction = CUDA([grid_out])
abstraction.print_code()
~~~

## Example

Examples are located in the `stencils` directory. A short example of 7-pt
stencil can be expressed as follows:

~~~python
# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("input", 3)
output = Grid("output", 3)

# Express computation
# output[i, j, k] is assumed
calc = param[0] * input(i, j, k) + \
       param[1] * input(i + 1, j, k) + \
       param[2] * input(i - 1, j, k) + \
       param[3] * input(i, j + 1, k) + \
       param[4] * input(i, j - 1, k) + \
       param[5] * input(i, j, k + 1) + \
       param[6] * input(i, j, k - 1)
output.assign(calc)
~~~

## Code generation using the library interface

Stencil code includes 3 different parts:

* The layout
* The backend vectorizer
* The code generation heuristics

The code generation for brick can be performed with, this will write the code to stdout:

~~~python
backend = BackendAVX512()
layout = Brick(dim=[8, 8], fold=[8], prec=2)
g = CodeGen(backend, layout, dimsplit=True)
g.gencode([output])
~~~

## Code generation with C++

Script is ended with stencil expression to generate.

~~~python
STENCIL=[output]
~~~

See [integration](docs/integration.md).

## Enable Code Suggestion in VS Code

Put the following in `.vscode/settings.json`

~~~json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/codegen"
    ],
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}/codegen"
    ]
}
~~~

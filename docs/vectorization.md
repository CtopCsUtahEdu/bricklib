# Vectorization in Bricks

One of main target of the code generation is to handle discontinuity using vectorization. The process largely depends on
which vectorization ISA is used.

## Vectorization Abstraction

brick(...)
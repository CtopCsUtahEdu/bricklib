# Codegen Integration

Compilation have two phase when using the code generator. 

1. The code generator is invoked to inject the generated code to the source.
2. Compile using host compiler as normal.

## Basic Integration Usage

`include/vecscatter.h` defined two different code generation targets: a) for use with tiled code b) more importantly, 
for use with bricked code.

The code generator will run C/C++ preprocessor that can resolve all preprocessor defines.

For examples see Line 35 in [weak/main.cpp](weak/main.cpp). The detailed usage are shown with #brick(...) and #tile(...)
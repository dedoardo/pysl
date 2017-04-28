# PYSL
PYthon Shading Language Compiler

TL = top-level (global scope)
BL = block-level (inside a function or nested functions scope)
### Types(ALL)
Non-opaque types follow the HLSL syntax. 
Supported scalar types:
```
bool
int
uint
float
```
*double* : not supported because of a literal problem, just trying to find a good way to specify it
*half*, *dword*   : not supported
`<type><dim0>[x<dim1>]`
External HLSL Documentation: [Link](https://msdn.microsoft.com/en-us/library/windows/desktop/bb509707(v=vs.85).aspx)

### Casting
Currently casting is done via the `@` operator following the syntax:
`<dest_type>@<name>`
That gets translated to
`(<dest_type>)<name>`

### Qualifiers(TL, BL)
Qualifiers are associated with a type and follow the syntax:
`<qualifier>.<type>` 
Currently two major qualifiers are supported (implementing more is trivial, just need to find the time/use case)
```
const : Block-level assignment or declaration
out   : Parameter
```

### Options (TL)
This is a very simple extension to the language that can simplify the work for backend developers that work with 
ubershaders, techniques and such.
`options(...)` can only be found in the global scope and instructs the metadata exporter to add possible compilation switches. It is usually found at the beginning of the source file, but there is no requirement. When compiling to HLSL it adds a 
```
#if defined(__INTELLISENSE__)
#define <option0>
#define <option1>
...
#endif
```
That helps with intellisense under Visual Studio if [HlslTools](https://github.com/tgjones/HlslTools) is being used. If no output metadata file is specified the helper block **will be** written anyway.

### Struct (TL)
Top level class that has no decorators associated to it. They are not exposed to the application as they **ARE NOT** constant buffers (see ConstantBuffer). They are just syntactic sugar for the shader. 
```python
class TestStruct:
    name0 : type0
    name1 : type1
```

### StageInput and Entry Points
Top-level class that define a specific shader input. StageInputs are strictly related to entry points and to avoid duplicate code. 
Entry points do not take any parameter and do not return anything.

```python
@StageInput(VSin)
class VertexInput:
    position : float3 = POSITION0

@StageInput(VSout, PSin)
class PixelInput:
    position : float4 = SV_POSITION

@StageInput(PSout)
class PixelOutput:
    color : float4
    
@VS
def vs_ep():
    output.position = input.position

@PS
def ps_ep():
    output.color = float4(1.0, 1.0, 1.0, 1.0)
```
**Every entry point has a local variable named `input` and one `output` that refer to that stage's in and out attributes**
While in HLSL you would create a struct on the stack and return it, in GLSL entry points are void and variables are set globally (they have `out` qualifiers). The approach chosen in PYSL is to declare what data you are going to return and what not and automatically to the job for you. Rewriting the parameters as input to the entry points would be error-prone(more checking to be done) and fundamentally redundant.
**Do not write entrypoints the HLSL way**, it will result in duplicate code

### Constant Buffers


### Opaque types (Sampler, Texture)



TODO:
Specify types in documentation
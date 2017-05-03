# PYSL: PY(thon) S(hading) L(anguage)

**Note:**  Project in development, currently only Vertex and Pixel shaders are supported, more stages and features will come in the future.

## Table of Contents:
- [Introduction](#introduction)
  - [What is it](#what-is-it)
  - [Disclaimer](#disclaimer)
- [Usage](#usage)
- [Language](#language)
  - [Comments](#comments)
  - [Native Types](#native-types)
    - [Scalar](#scalar)
    - [Vector Matrix](#vector-matrix)
  - [Casting](#casting)
  - [Vector Access](#vector-access)
  - [Matrix Access](#matrix-access)
  - [Qualifiers](#qualifiers)
  - [Math Operations](#math-operations)
  - [Construction and Initialization](#construction-and-initialization)
  - [Options](#options)
  - [Struct](#struct)
  - [StageInput](#stageinput)
  - [Entry Points](#entry-points)
  - [Constant Buffer](#constant-buffer)
  - [Sampler](#sampler)
  - [Intrinsics](#intrinsics)

## Introduction
## What is it
*PYSL* is a subset of [Python 3.6](https://www.python.org/about/) that is translatable to *HLSL* and *GLSL*. There is no special syntax, the code is **syntactically valid** Python. The specification contains a list of operators, decorators and tokens that are recognized by the compiler (`pyslc`). 
The code has to be valid python because the input file is parsed into an [AST](https://docs.python.org/3/library/ast.html) that is then traversed and written out as *HLSL*, *GLSL* or both. As the output is readable (blank lines are also preserved) later tweaks can and should be done if needed or possible. Current outputs are [**HLSL Shader Model 5**](https://msdn.microsoft.com/en-us/library/windows/desktop/ff471356(v=vs.85).aspx) and [**GLSL 4.5**](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.50.pdf).
`pyslc` in addition to *HLSL* and *GLSL* allows to export metadata in *JSON* containing entry points, compile flags, resources and more custom data.  
C++ headers can also be exported containing structure definitions made to match the layout of the one specified in the shader. 
For more information see [Usage](#usage).

##### Disclaimer
This is not a full-fledged compiler. `pyslc` allows evaluation of the output (using `fxc` and `glsllangValidator`), but a correctly compiled *PYSL* script is **not guaranteed** to be a valid shader. If you try to access the `z` component of a 2D vector, it's easier to let the backend compiler complain. This does not mean that there is no type-checking and such, but it is strictly limited to what concerns *PYSL* specific features, nothing else.  
*PYSL* has actually not that many special tokens and pretty much everything you need to know is contained inside this very README.  
This is meant for people who already have some knowledge of a shading language, the documentation reflects this.  
This is not the ultimate solution, it aims at saving **~50% of the work**. The extra time that might be required to tweak the output is made up by the less verbosity of python compared to C-like code.  
Nonetheless the output is valid shader code and with some attention complex and valid shaders can be written and compiled directly.

## Usage
`pysl.py` is contained inside the root directory, all it really does is call `pysl/pyslc.py`.
```ruby
usage: pysl.py [-h] [-ohlsl5 OHLSL5] [-oglsl45 OGLSL45] [-ojson OJSON]
               [-ohpp OHPP] [-vhlsl5 VHLSL5] [-vglsl45 VGLSL45]
               output

PYthon Shading Language compiler

positional arguments:
  output

optional arguments:
  -h, --help        show this help message and exit
  -ohlsl5 OHLSL5    HLSL destination path
  -oglsl45 OGLSL45  GLSL destination path
  -ojson OJSON      JSON metadata destination path
  -ohpp OHPP        C++ header destination path
  -vhlsl5 VHLSL5    HLSL compiler path for validation(fxc)
  -vglsl45 VGLSL45  GLSL compiler path for validation(glslLangValidator)

```

## Language
*PYSL* is heavily inspired to *HLSL*, thus if you are familiar with it you can pretty much start coding right away. The parts that are borrowed from *GLSL* are to make the compilation less painful.  
Very often the translation of each part of the language is reported, to make it easier to locate and understand errors in compilation.
Top-level = global scope
Block-level = any code that is inside a function at any level.

### Comments
Strings expressions (without any storage target) are translated to comments. They can appear both at Top-level and Block-level.
```python
'''foo routine'''
def foo() -> float:
    'Returns a random number'
    return 0.3
```
Gets translated to
```c
/*foo routine*/
void foo() 
{ 
    /*Returns a random number*/
    return 1.0; 
}
```

### Native Types
Non-opaque native types are 1-1 mapped to *HLSL* types.
##### Scalar
Supported scalar types: `void`, `bool`, `int`, `uint`, `float`
Unsupported scalar types: `double`, `half`, `dword`  
**`double`** is not supported because of a literal problem, waiting to find a clean way to specify it.

##### Vector Matrix
Vector: `<scalar_type><elements>`  
Matrix: `<scalar_type><rows><cols>`  
Non scalar types can have `2`, `3` or `4` components. To make life easier `1` components are not supported, just use the corresponding scalar or vector type.   
This is because *HLSL* allows `float1x4` but *GLSL* doesn't compile `vec1`.  
Additional documentation(HLSL): [Link](https://msdn.microsoft.com/en-us/library/windows/desktop/bb509707(v=vs.85).aspx)

### Casting
Currently casting is done via the `@` operator following the syntax:
`<dest_type>@<name>`
```
HLSL: (<dest_type>)<name>
GLSL: <dest_type>(<name>)
```

### Vector Access
Swizzling is the same as in *HLSL* and *GLSL*, with 2 separate sets that can't be mixed together:
`rgba`
`xyzw`

```c
val : float3 = { 1.0, 2.0, 3.0 }
tmp : float3 = val.xya #INVALID 
tmp : float3 = val.xyw #OK
tmp : float3 = val.rga #OK
```
If the swizzle code is invalid *PYSL* won't complain, no checking is done.

### Matrix Access
Matrices are not swizzable as it is not supported by GLSL. Subscript operators are also discouraged with matrices. The only way to access matrix rows, cols and elements is via the `row[n]` `col[n]` intrinsics.
```
row[<n>](<matrix>, <row>)
col[<n>](<matrix>, <col>)

<n>: is an integer that can be 1, 2, 3, 4 and indicates the number of elements to be retrieved from the beginning of the row
<row>, <col>: 0-th indexed row to access
```

### Qualifiers
Qualifiers are associated with a type and follow the syntax:
`<qualifier>.<type>` 
Currently two major qualifiers are supported (implementing more is trivial, just need to find the time/use cases)
```
const : Block-level assignment or declaration
out   : Parameter
```
Example:
```python
def incr(x : out.int) -> void:
    x += 1
```
Translated to:
```c
void incr(out int x) { x+= 1; }
```

### Math Operations
All mathematical operators are supported and work as usual, meaning that `<vector> * <vector>` does a component-wise multiplication and `<vector> * <scalar>` scales off all components by `<scalar`>. 
Vector transformations are done using the `mul` intrinsic as in *HLSL* and the first element is supposed to be a vector. The order stays the same in *HLSL* but is swapped in *GLSL*
```
PYSL: mul(<vector>, <matrix>)
HLSL: mul(<vector>, <matrix>)
GLSL: <matrix> * <vector>
```

### Construction and Initialization
Building vectors from other vectors works the same in *HLSL* and *GLSL*. You can build a `N` dimensional vector with any other combination of vectors and scalars who add up to `N` components (components are filled in from left to right), examples include:
```python
foo : float3 = float3(0.0, 0.0, 0.0)
foo : float3 = float3(float2(0.0, 0.0), 0.0)
foo : float4 = float4(0.0, float2(0.0, 0.0), 0.0)
```
Vector can also be initialized from a single value. The constructor needs to be repeated, otherwise the compiler won't recognize the construction and will just be like assigning a single float to a vector (which is valid HLSL, but not HLSL)
```python
foo : float3 = float3(1.f) # OK
foo : float3 = 1.0 # INVALID
```
Initializer-lists are also a valid and less-verbose option, they are both supported in HLSL and GLSL the same way, have no special meaning in *PYSL* and are translated as they are.
Some notes:
- They only work with composite types. `float a = { 1.f };` is valid HLSL but invalid *GLSL* [[4.1.11]](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.50.pdf), please don't do it.
- Do not use initializer-lists for matrices, reason being that HLSL and GLSL have different order of initialization and they need to be transposed.
- Do not initialize a matrix with a single value as it will result in a compilation error on HLSL. Can be worked around, but not a priority.

Matrix constructor take elements in row-major order, meaning that the first N elements correspond to the first row
```python
# OK
foo : float4x4 = float4x4(1.0, 0.0, 0.0, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0) 
# INVALID
foo : float4x4 = { 1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0 }

# INVALID
foo : float4x4 = float4x4(0.0)

# INVALID
foo : float4x4 = { 0.0 }
```


### Options
This is a very simple extension to the language that is meant simplify the work for developers that work with uber-shaders, techniques and such.
`options(...)` can only be found at top-level and instructs the metadata exporter to add possible compilation switches. It is usually found at the beginning of the source file, but can be anywhere. When compiling to HLSL it adds a 
```
#if defined(__INTELLISENSE__)
#define <option0>
#define <option1>
...
#endif
```
That helps with intellisense under Visual Studio if [HlslTools](https://github.com/tgjones/HlslTools) is being used. If no output metadata file is specified the helper block **will be** written anyway.

### Struct
Top-level class that has no decorators associated to it. They are not exposed to the application as they **ARE NOT** constant buffers (see [ConstantBuffer](#constantbuffer)). They are just syntactic sugar for the shader. 
Currently there is no way to specify padding, size and such.
```python
class TestStruct:
    <name> : <type>
    <name> : <type>
```

### StageInput
Top-level class that represents a specific shader input as indicated by its decorator.
```python
@StageInput(<stages>)
class Input:
	<name> : <type> = <semantic>
<stages>: Can be one or more of the following: VSin, VSout, PSin, PSout. (<stage>[in|out])
<semantic>: HLSL semantic as described in :https://msdn.microsoft.com/en-us/library/windows/desktop/bb509647(v=vs.85).aspx
```
Semantics are used for 2 things:
- Identify built-in attributes (System-Values)
- Match input/output parameters from different stages
**Use HLSL-style semantics** as they are directly translated into HLSL without any checking. System-Values need to be correct as they need to be mapped to the GLSL ones.
For a list of HLSL-GLSL System-Values mapping see: https://anteru.net/blog/2016/mapping-between-hlsl-and-glsl/.  
`StageInput`'s fields can only be accessed from an [Entry Point](#entrypoint) using the special variables `input` and `output`. The type is matched with the corresponding input based on the stage of the function. **They are not global** and accessing them from anywhere else will trigger compilation errors. This might upset GLSL developers as they are used to have them global, but in HLSL this is not the case. If you need to have values in different functions just pass them around, the whole shader will probably be inlined anyway.
```python
# INVALID
def get_foo() -> float4:
	return float4(input.position.xyz, 1.0)
   
@VS
def vs_ep():
	output.position = get_foo()

# CORRECT
def get_foo(position : float3) -> float4:
	return float4(position.xyz, 1.0)

@VS
def vs_ep():
	output.position = get_foo(input.position)
```

## Entry Points
Top-level `void`-returning function with no arguments (They are automatically filled in) with a `<stage>` decorator
```python
@<stage>
def entry_point() -> void:
    pass

<stage>: One of the following: VS, PS.
```
Do not name a function `main`, otherwise GLSL shaders won't compile. In order to have multiple shaders in a single file a couple of compilation flags have to be added before compiling the script from the application.  
```
#version 450
#define <entry_point> main
#define PYSL_<shader_type>
```
That for example can be done as follows:
```cpp
char *sources[4] = { "#version 450\n", "#define <entry_point> main\n", "#define PYSL_<shader_type>\n" ,glsl_source_code };
glShaderSourceARB(shader, 4, sources, NULL);
```
This is the fastest way to circumvent this problem, the `pyslc` emitter won't write the `#version` tag at beginning of the file as extra `#define`s are needed.

## Constant Buffer
Top-level class containing the `ConstantBuffer(...)` decorator.
```python
@ConstantBuffer(<size>)
def Example:
    <name> : <type> = <offset>

<size>, <offset>: They are not expressed in bytes, but in constant unit, where a constant equals 32-bits. <size> forces the structure to be a certain size, while <offset> indicates where from the beginning of the structure the variable should be located.
```
Example:
```python
@ConstantBuffer(16 * 16)
def PerFrame:
    view_projection : float4x4 = 0
    view : float4x4 = 16
```
GLSL `enhanced_layouts` alignments are not supported. Use `<offset>` to obtain the same behavior.

`ConstantBuffer`s can be exported to a `C++` header and will mantain the same padding as specified in the shader. Exported code should be compatible with `msvc`, `gcc` and `clang`, but due to the presence of `static_assert`s all over the place it requires `C++11`.

### Sampler
Top-level resource declaration:
```
<name> : register(<opaque_type>, <slot>) = (<key0> = <val0>, <key1>=<val1>, ...)
<opaque_type>: One of the following:
 Sampler1D, Sampler2D, Sampler3D, SamplerCube,
 Sampler1DArray, Sampler2DArray, Sampler2DMS,
 Sampler2DMSArray, SamplerCubeArray, Sampler1DShadow,
 Sampler2DShadow, Sampler1DArrayShadow,
 Sampler2DArrayShadow, SamplerCubeShadow,
 SamplerCubeArrayShadow.
 <slot>: Binding point
 <key>, <val>: List of key,value that are exported as metadata. Key has to be a name, while value can be a name, number or string literal. There are **no** keywords, all is custom.
```
Example:
```python
default_sampler : register(Sampler2D, 0) = export(
    MIN_FILTER = POINT,
    MAG_FILTER = LINEAR
  )
```
Gets exported to:
```json
"Samplers": [
    {
        "Name": "default_sampler",
        "MIN_FILTER": "POINT",
        "MAG_FILTER": "LINEAR"
    }
```

GLSL uses a single sampler and looks up using `texture(sampler, uv, ...)` while HLSL does `Texture.Sample(Sampler, uv, ...)`. Using the HLSL approach would cause some issues as you could do `Texture.Sample(Sampler0)` `Texture.Sampler(Sample1)`. That is not straightforward to translate to GLSL, as it would require some contextual knowledge that doesn't play well with the current `pyslc.py` implementation. The approach used to handle the `Sampler / Texture` problem is closer to GLSL here as there are no textures but just samplers, the supported samplers correspond to the PascalCased ones specified in section [4.1] of the [GLSL spec](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.50.pdf).   
Although the use of exclusive samplers an object-oriented approach is kept and thus sampling follows the following syntax:
``` 
`<name>.<sample_function>(<args>)`
```
Opaque types are the only object-oriented part of the API. Intrisics could have been used, but having a distinction between standard operation and operations on special objects is beneficial.
Now a list of the methods supported by the `Sampler` and their translation. Pay careful attention to the parameters and how they are translated as there are inconsistencies between *GLSL* and *HLSL*. For more details on the parameters consult the respective documentations.
### `Sampler.Sample`
```xml
<sampler>.Sample(<uv>, [<bias|cmp> [,<offset>]])
---
<uv>: Texture coordinates in normalized space, the number of dimensions depends on the <sampler>'s type works the same in HLSL/GLSL
<bias|cmp>: If <sampler> is shadow the second parameters indicates the lod bias, otherwise the value to be compared against
<offset>: Integer offset in **texel-space** to be added to the <uv>.
```
```xml
if <sampler>.isShadow():
    HLSL: <texture>.SampleCmp(<sampler>, <uv>, <cmp>) <texture>.SampleCmp(<sampler>, <uv>, <cmp>, <offset>)
    GLSL: texture(<sampler>, <uv, cmp>) textureOffset(<sampler>, <uv,cmp>, <offset>)
else:
    HLSL: <texture>.SampleBias(<sampler>, <uv>, <bias>)
    GLSL: texture(<sampler>, <uv>, <bias>) textureOffset(<sampler>, <uv>, <offset>, <bias>)
```

### `Sampler.Load`
**Unsupported types:**
- **`Shadow`**
- **`Cube, CubeArray`**

Miplevel is ignored if texture is multisampled, just specify 0. 
This is because multisample textures have no miplevels.
Offset cannot be applied to multisampled textures, *GLSL* has no overload for it.
```xml
<sampler>.Load(<texel_coord>, <miplevel> [,<offset|sample>])
---
<texel_coord>: Texture coordinates in **texel-space**.
<miplevel>: Mipmap level to sample from (LOD).
<offset|sample>: If <sampler> is multi-sample then indicates the index of the sample to take otherwise indicates the integer offset in **texel-space** to be added to the <uv>
```
```xml
if <sampler>.isMultiSample()
    HLSL: <texture>.Load(<sampler>, <texel_coord>, <sample>)
else:
    HLSL: <texture>.Load(<sampler>, <texel_coord, miplevel>, 0, <offset>)

GLSL: texelFetch(<sampler>, <texel_coord>, <miplevel>, <sample>) texelFetchOffset(<sampler>, <texel_coord>, <miplevel>, <offset>) 
```

### `Sampler.SampleGrad`
**Unsupported types:**:
- **`Shadow`**
- **`MS`**

```xml
<sampler>.SampleGrad(<uv>, <ddx>, <ddy>)
---
<uv>: Texture coordinates in normalized space, the number of dimensions depends on the <sampler>'s type works the same in HLSL/GLSL
<ddx>: Rate of change of texture coordinate per pixel in the window's X direction
<ddy>: Rate of change of texture coordinate per pixel in the window's Y direction
<ddx> and <ddy> are known as derivative functions, obtained from the 2x2pixels computation and used for instance for trilinear/anisotropic filtering to determine the skewing.
```

```xml
HLSL: <texture>.SampleGrad(<sampler>, <uv>, <ddx>, <ddy>)
GLSL: textureGrad(<sampler>, <uv>, <ddx>, <ddy>)
```

### `Sample.SampleLevel`
```xml
<sampler>.SampleLevel(<uv>, <miplevel>)
---
<uv>: Texture coordinates in normalized space, the number of dimensions depends on the <sampler>'s type works the same in HLSL/GLSL
<miplevel>: Mipmap level to sample from (LOD).
```

```xml
HLSL: <texture>.SampleLevel(<sampler>, <uv>, <miplevel>)
GLSL: textureLod(<sampler>, <uv>, <miplevel>)
```

### `Sample.Gather`
**Unsupported types:**
- **`1D`**
- **`3D`**
- **`MS`**

```xml
<sampler>.Gather(<uv>, <channel|cmp> [,<offset>])
---
<uv>: Texture coordinates in normalized space, the number of dimensions depends on the <sampler>'s type works the same in HLSL/GLSL
<channel|cmp>: If <sampler> is shadow it indicates the value to be compared against, otherwise the component(<channel>) to be fetched. <channel> has to be a number literal, constant integer expressions are not supported.
```

```xml
if <sampler>.isShadow():
    HLSL: <texture>.GatherCmp(<sampler>, <uv>, <cmp>, [<offset>, 0])
    GLSL: textureGather(<sampler>, <uv>, <cmp>) textureGatherOffset(<sampler>, <uv>, <cmp>, <offset>)
else:
    HLSL: <texture>.Gather[Red|Green|Blue|Alpha](<sampler>, <uv>, [<offset>, 0])
    GLSL: textureGather(<sampler>, <uv>, <channel>) textureGatherOffset(<sampler>, <uv>, <offset>, <channel>)
```
`<offset>` cannot be applied to `SamplerCubexxx` as *GLSL* has no overload for it

### `Sample.GetDimensions`
This one requires some wrappers for *HLSL* to be written as *HLSL* returns void and takes `out` parameters while *HLSL* returns a vector containing all the values. The idea is to use the same signature as *GLSL* and just write a couple of free functions that replicate the behavior.

Some additional notes:
- In order to access array add an extra coordinate to the `<uv` or `<texel_coord>`, just as you would in *HLSL* and *GLSL*
- https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/texelFetchOffset.xhtml doesn't take a `<sample>`
- `<offset> as for specification mustbe a constant expression (evaluable at compile-time)
- *GLSL*'s `textureProj` has no mapping


### Intrinsics
Intrinsics have pretty straightforward mappings, `pysl/pysl.py` contains a list of supported and dropped intrinsics and how they translate.

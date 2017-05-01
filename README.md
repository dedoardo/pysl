# PYSL
PYthon Shading Language compiler

This is not a full-fledged compiler. It's purpose is to take a shader written in PYSL and translate it the best way possible. There are some checks around, but rewriting an entire compiler was not the purpose as it would be redundant. The general idea is:
- If some syntax is shared between the backend languages, **no checking** is done as the compiler will complain anyway
- If there are collision in semantics/syntax then indeed checks are done and code is modified accordingly
- Parameter types are not checked. If you try to sample a `sampler2D` with a single integer I'll let the backend compiler complain
- Every time you use an intrinsic or a method check the documentation. Translations are usually pretty straightforward, but there might be some caveats (e.g. Combination of offsets/multisample in `Sampler::Load` when translating to GLSL). Usually the inconsistencies are reported.

This is not the ultimate solution, it just aims are saving you *95%* of the work, further optimization or tweaks might have to be done and *should* be done, this is possible because the compilation output is readable and still high-level. Nonetheless the outputs are valid and with some attention complex and valid shaders can be written without later tweaking.

Outputs:
HLSL SM5.0
GLSL `#version 450`

TL = top-level (global scope)
BL = block-level (inside a function or nested functions scope)

### `pyslc`
```ruby
usage: pyslc.py [-h] [-ohlsl5 OHLSL5] [-oglsl45 OGLSL45] [-ojson OJSON]
                [-ohpp OHPP]
                output

PYthon Shading Language Compiler

positional arguments:
  output

optional arguments:
  -h, --help        show this help message and exit
  -ohlsl5 OHLSL5    HLSL destination path
  -oglsl45 OGLSL45  GLSL destination path
  -ojson OJSON      JSON metadata destination path
  -ohpp OHPP        C++ header destination path
```
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

### Swizzling
Swizzling is valid as in HLSL / GLSL. Two sets:
`rgba`
`xyzw`
They sets can't be mixed together:
```c
val : float3 = { 1.0, 2.0, 3.0 }
tmp : float3 = val.xya #INVALID 
tmp : float3 = val.xyw #OK
tmp : float3 = val.rga #OK
```
As the swizzling is the same in HLSL and GLSL no checking is actually done

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
Entry points do not take any parameter and do not return anything. In order to compile GLSL shaders and yet have a single file the entry point must be defined as follows:
If compiling a vertex shader
```c
char *sources[4] = { "#define <entry_point> main\n", "#define PYSL_<shader_type>\n", "#version 450\n" ,glsl_source_code };
glShaderSourceARB(shader, 2, sources, NULL);
```
Unfortunately it's currently the easiest and fastest way to circumvent this problem. When writing the glsl export `#version` won't be written.
`PYSL_<shader_type>` where shader type is either `VERTEX_SHADER`, `GEOMETRY_SHADER` or `PIXEL_SHADER`  is needed for GLSL aswell.
**Do not name any function main(), you won't be able to compiler your GLSL shader otherwise** 


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

Semantics use the HLSL syntax for easier translation, but any name could be really used. They are needed to pair input/output StageInputs even if their order is changed.
```python
@StageInput(VSout)
class VertexOutput:
    depth : float2 = NORMAL0
    normal : float3 = NORMAL1

@StageInput(PSin)
class PixelInput:
    normal : float3 = NORMAL1
    depth : float2 = NORMAL0
```
When compiling to `GLSL4.5` assigns the `PixelInput::depth` to slot `0` and `PixelInput::normal` to slot `1` as expected. The `out` structure gives the order and the `in` tries to match the layout. This is because the `in` needs the `out` to be correctly declared to proceed.

### Constant Buffers(TL)
Constant buffers are Top-level class definitions. They work pretty much the same way StageInputs except that the semantic(HLSL-style: SV_Position, Position, Texcoord, ...) wouldn't make any sense and is replaced with an offset.
ConstantBuffers are preceded by a single decorator `ConstantBuffer` where you can specify a specific size for the resulting
structure.  
```python
@ConstantBuffer(16 * 16)
def PerFrame:
    view_projection : float4x4 = 0
    view : float4x4 = 16
```
**Note: <offset> and <enforced_size> are not in bytes, but correspond to 32-bit constants**. To obtain the actual byte-size just multiply by 4. 
GLSL `enhanced_layouts` alignments are not supported. Use offset to obtain the same behavior.

### Opaque types (Sampler, Texture) (TL)
Opaque types are a Top-Level resource declaration that is declared as follows:
`<name> : register(<opaque_type>, <slot>) = (<key0>=<val0>, <key1>=<val1>, ...)`
The use of the register keyword is required and `<slot>` indicates the binding point. GLSL uses a single sampler and looks up using `texture(sampler, uv, ...)` while HLSL does `Texture.Sample(Sampler, uv, ...)`. Using the HLSL approach would cause some issues as you could do `Texture.Sample(Sampler0)` `Texture.Sampler(Sample1)`. That is not straightforward to translate to GLSL, as it would require some contextual knowledge that doesn't play well with the current `pyslc.py` implementation. The approach used to handle the Sampler/Texture problem is closer to GLSL here as there are no textures but just samplers, the supported samplers correspond to the ones specified in section [4.1] of the [GLSL spec](https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.50.pdf).
The sampling is thus 
`<name>.<sample_function>(uv, ...)`
Opaque types are the only object-oriented part of the API. Intrisics could have been used, but having a distinction between standard operation and operations on special objects is beneficial.
#### Sample Methods:
This is just how translation works, for more details regarding allowed parameters check the respective HLSL/GLSL documentation.
#### `Sample`

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

##### `Load`
**NO Shadow**
**NO SamplerCube, SamplerCubeArray**
Miplevel is ignored if texture is multisampled, just specify 0. 
This is because multisample textures have no miplevels.
Offset cannot be applied to multisampled textures, GLSL has no overload for it.
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

##### `SampleGrad`
**No Shadow**
**No multisample**
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

##### `SampleLevel`
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

##### `Gather`
** No Sampler1Dxxx, Sampler3Dxxx, No multisampled**
<channel> has to be a number literal. Constant integer expression are not supported
```xml
<sampler>.Gather(<uv>, <channel|cmp> [,<offset>])
---
<uv>: Texture coordinates in normalized space, the number of dimensions depends on the <sampler>'s type works the same in HLSL/GLSL
<channel|cmp>: If <sampler> is shadow it indicates the value to be compared against, otherwise the component(<channel>) to be fetched
```

```xml
if <sampler>.isShadow():
    HLSL: <texture>.GatherCmp(<sampler>, <uv>, <cmp>, [<offset>, 0])
    GLSL: textureGather(<sampler>, <uv>, <cmp>) textureGatherOffset(<sampler>, <uv>, <cmp>, <offset>)
else:
    HLSL: <texture>.Gather[Red|Green|Blue|Alpha](<sampler>, <uv>, [<offset>, 0])
    GLSL: textureGather(<sampler>, <uv>, <channel>) textureGatherOffset(<sampler>, <uv>, <offset>, <channel>)
```
Offset cannot be applied to `SamplerCubexxx` as GLSL has no overload for it

##### `GetDimensions`
This one requires some wrappers for HLSL to be written as HLSL returns void and takes `out` parameters while GLSL returns a vector containing all. The idea is to use the same return semantic as OpenGL and just write a couple of free functions that replicate the behavior.

Notes: 
- In order to access array add an extra coordinate to the `<uv` or `<texel_coord>`, just as you would in HLSL/GLSL
- https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/texelFetchOffset.xhtml doesn't take a <sample>
- <offset> as for specification mustbe a constant expression (evaluable at compile-time)
- GLSL's `textureProj` has no mapping


### Intrinsics

# Backend
List of methods that need to be present. All the type parameters are assumed to be valid PYSL keywords. 
`args` are closures, just call them to automatically evaluate them.
- `init(path : str) -> bool` Initializes the backend. If you need to output some header at the beginning of the file this is the time
- `write(string : str)` Writes directly to the file. This is used by the `pyslc.Translate` in order to write out shared (C-syntax) code.
- `declaration(pysl.Declaration)`
- `options(options : [str])` Called whenever a `options` call is encountered at the TL. ( see `options` docs)
- `struct(struct : pysl.Struct)`
- `stage_input(si : pysl.StageInput)`
- `entry_point_beg(func : pysl.Function, sin : pysl.StageInput, sout : pysl.StageInput)` Function signature, if you need to add instructions at the beginning of the block, now it's time
- `entry_point_end(func : pysl.Function` End of function, just in case something else has to be written out
- `constant_buffer(pysl.ConstantBuffer)`
- `sampler(pysl.Sampler)`
- `method_call(caller : pysl.Object, method : str, args)`
- `constructor(type : str, args)`
- `intrinsic(type : str, args)`
- `special_attribute(attribute : str)` BL `input`/`output`




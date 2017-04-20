from enum import Enum

# The following **all** refer to declarations, not invocations
#-------------------------------------------------------------------------------
# Scalar
class Scalar(Enum):
    BOOL   = 0
    INT    = 1
    UINT   = 2
    FLOAT  = 3
    # DOUBLE = 4 Not supported yet, since i have yet to figure out a way to handle literals


def scalar_to_cpp(scalar : Scalar) -> str:
    if scalar == Scalar.BOOL:
        return 'bool' 
    elif scalar == Scalar.INT:
        return 'int'
    elif scalar == Scalar.UINT:
        return 'unsigned int'
    elif scalar == Scalar.FLOAT:
        return 'float'
    return 'ERROR'

# Vector | Matrix
class Type:
    def __init__(self):
        self.str : str = None
        self.type : str = None
        self.dim0 : int = 1 # Vectors and matrix rows
        self.dim1 : int = 1 # Coloumns

class InputElement:
    def __init__(self):
        self.name : str = None
        self.type : Type = None
        self.semantic : str = None
        self.conditions : [str] = None

class StageInput:
    def __init__(self):
        self.elements : [InputElement] = None
        self.post_conditions : [str] = None

class Constant:
    def __init__(self):
        self.name : str = None
        self.type : Type = None
        self.array_size : int = None
        self.offset : int = None

class Object:
    def __init__(self):
        self.name = None

class ConstantBuffer(Object):
    def __init__(self):
        Object.__init__(self)
        self.constants : [Constant] = None
        self.enforced_size : int = None

class SamplerState(Object):
    def __init__(self):
        Object.__init__(self)
        self.attributes : [(str, str)] = []

class Texture(Object):
    def __init__(self):
        Object.__init__(self)
        self.type : str = None
        self.attributes : [(str, str)] = []

class FunctionArg:
    def __init__(self):
        self.type : Type = None
        self.name : str = None
        self.out : bool = None

class Function:
    def __init__(self):
        self.name = None
        self.return_value : Type = None
        self.args : [FunctionArg] = None

# HLSL convention, if the GLSL version differs it's reported on the side
INTRINSICS = [
    'abs', 
    'acos',
    'all', 
    'any',
    'asin',
    'atan2',
    'ceil',
    'clamp',
    'cos',
    'cosh',
    'cross',
    # Partial derivates are dFd[x|y][|Coarse|Fine] in GLSL
    'ddx', 
    'ddy', 
    'ddx_coarse',
    'ddy_coarse',
    'ddx_fine',
    'ddy_fine',
    'determinant',
    'distance',
    'dot',
    'exp',
    'exp2',
    'faceforward',
    'firstbithigh', # GLSL: findLSB
    'firstbitlow', # GLSL: findMSB
    'floor',
    'fma',
    'frac', # GLSL: fract
    'frexp',
    'fwidth',
    'isnan',
    'ldexp',
    'length',
    'lerp', # GLSL: mix
    'lerp',
    'log',
    'log2',
    'max',
    'min',
    'mul',
    'modf',
    'noise',
    'normalize',
    'pow',
    'radians',
    'reflect',
    'refract',
    'round',
    'sign',
    'sin',
    'sinh',
    'smoothstep',
    'sqrt',
    'step',
    'tan',
    'tanh',
    'transpose',
    'trunc'
]
# Remaining list of dropped intristics (those for which I couldn't **quickly** find a 1-1 mapping)
# <name> HLSL | GLSL
# acosh N|Y
# asinh N|Y
# atan Y|N
# clip Y|N
# countbits Y|N
# degrees N|Y
# dst Y|N
# EvaluateAttribute*** Y|???
# fwidthCoarse fwidthFine N|Y
# HLSL has isfinite GLSL has isinf ^^ add '!' when compiling
# lit N|Y ( is it still used ? )
# log10 Y|N
# mad Y|N
# mod N|Y
# noise1,2,3,4 N|Y
# reversebits Y|N
# roundEven N|Y
# sincos Y|N

# They are almost identical to intrinsics from a parsing perspective
# Note that this is still PYSL no HLSL, just using the same conventions
CONSTRUCTORS = [
    'float', 'float2', 'float3', 'float4',
    'float3x3', 'float4x4'
    'int', 'int2', 'int3', 'int4',
    'uint', 'uint2', 'uint3', 'uint4',
]

class Keywords:
    VertexShaderDecorator = 'VertexShader'
    PixelShaderDecorator = 'PixelShader'
    StageInputDecorator = 'StageInput'
    ConstantBufferDecorator = 'ConstantBuffer'

    SamplerStateConstructor = 'SamplerState'
    TextureConstructor = 'Texture'

    OptionsKey = 'Options'
    ConstantBuffersKey = 'ConstantBuffers'
    SamplerStatesKey = 'SamplerStates'
    TexturesKey = 'Textures'
    VertexShaderKey = 'VertexShader'
    PixelShaderKey = 'PixelShader'
    NameKey = 'Name'
    SizeKey = 'Size'
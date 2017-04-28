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
    """Converts a scalar type to the corresponding C type as a string """
    if scalar == Scalar.BOOL:
        return 'bool' 
    elif scalar == Scalar.INT:
        return 'int'
    elif scalar == Scalar.UINT:
        return 'unsigned int'
    elif scalar == Scalar.FLOAT:
        return 'float'
    return 'ERROR'

class Type:
    """Wraps all the scalar, vector and matrix types"""
    def __init__(self):
        #: String representation, kept here mostly for ease of use
        self.str : str = None

        #: Base type
        self.type : Scalar = None
        
        #: Vector and matrix rows
        self.dim0 : int = 1 
        self.dim1 : int = 1

class Struct:
    def __init__(self):
        self.name : str = None
        self.elements : [(Type, str)] = None

class InputElement:
    def __init__(self):
        self.name : str = None
        self.type : Type = None
        self.semantic : str = None
        self.conditions : [str] = None

class StageInput:
    def __init__(self):
        self.stages : [str] = None
        self.elements : [InputElement] = None
        self.post_conditions : [str] = None

class Constant:
    def __init__(self):
        self.name : str = None
        self.type : Type = None
        self.array_size : int = None
        self.offset : int = None

class Object:
    def __init__(self, name = None):
        self.name = name

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

class FunctionArg(Object):
    def __init__(self, type = None, name = None, out = None):
        Object.__init__(self, name)
        self.type : Type = type
        self.out : bool = out

class Function(Object):
    def __init__(self):
        Object.__init__(self)
        self.return_value : Type = None
        self.args : [FunctionArg] = None
        self.stage : str = None # if entry point

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
TYPES = [
    'bool', 'bool2', 'bool3', 'bool4',
    'int', 'int2', 'int3', 'int4',
    'uint', 'uint2', 'uint3', 'uint4',
    'float', 'float2', 'float3', 'float4',
    'bool1x1', 'bool1x2', 'bool1x3', 'bool1x4',
    'bool2x1', 'bool2x2', 'bool2x3', 'bool2x4',
    'bool3x1', 'bool3x2', 'bool3x3', 'bool3x4',
    'bool4x1', 'bool4x2', 'bool4x3', 'bool4x4',
    'int1x1', 'int1x2', 'int1x3', 'int1x4',
    'int2x1', 'int2x2', 'int2x3', 'int2x4',
    'int3x1', 'int3x2', 'int3x3', 'int3x4',
    'int4x1', 'int4x2', 'int4x3', 'int4x4',
    'uint1x1', 'uint1x2', 'uint1x3', 'uint1x4',
    'uint2x1', 'uint2x2', 'uint2x3', 'uint2x4',
    'uint3x1', 'uint3x2', 'uint3x3', 'uint3x4',
    'uint4x1', 'uint4x2', 'uint4x3', 'uint4x4',
    'float1x1', 'float1x2', 'float1x3', 'float1x4',
    'float2x1', 'float2x2', 'float2x3', 'float2x4',
    'float3x1', 'float3x2', 'float3x3', 'float3x4',
    'float4x1', 'float4x2', 'float4x3', 'float4x4'
]

class Keywords:
    VertexShaderDecorator = 'VS'
    PixelShaderDecorator = 'PS'
    FunctionDecorators = [VertexShaderDecorator, PixelShaderDecorator]
    In = 'in'
    Out = 'out'
    VertexShaderInputDecorator = VertexShaderDecorator + In
    VertexShaderOutputDecorator = VertexShaderDecorator + Out
    PixelShaderInputDecorator = PixelShaderDecorator + In
    PixelShaderOutputDecorator = PixelShaderDecorator + Out
    StageInputDecorators = [VertexShaderInputDecorator, VertexShaderOutputDecorator, PixelShaderInputDecorator, PixelShaderOutputDecorator]
    StageInputDecorator = 'StageInput'
    ConstantBufferDecorator = 'ConstantBuffer'

    SamplerStateConstructor = 'SamplerState'
    TextureConstructor = 'Texture'

    InputValue = 'input'
    OutputValue = 'output'

    OptionsKey = 'Options'
    ConstantBuffersKey = 'ConstantBuffers'
    SamplerStatesKey = 'SamplerStates'
    TexturesKey = 'Textures'
    VertexShaderKey = 'VertexShader'
    PixelShaderKey = 'PixelShader'
    NameKey = 'Name'
    SizeKey = 'Size'
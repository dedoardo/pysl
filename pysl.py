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

class Declaration:
    def __init__(self, type : str = None, name : str = None, qualifiers : [str] = None):
        self.name : str = None
        self.type : str = None
        self.qualifiers : [str] = None

class Object:
    def __init__(self, name = None):
        self.name = name

class Struct(Object):
    def __init__(self):
        Object.__init__(self)
        self.elements : [(Type, str)] = None

class InputElement(Object):
    def __init__(self):
        Object.__init__(self)    
        self.type : Type = None
        self.semantic : str = None
        self.conditions : [str] = None

class StageInput(Object):
    def __init__(self):
        Object.__init__(self)
        self.stages : [str] = None
        self.elements : [InputElement] = None
        self.post_conditions : [str] = None

class Constant(Object):
    def __init__(self):
        Object.__init__(self)
        self.type : Type = None
        self.array_size : int = None
        self.offset : int = None

class ConstantBuffer(Object):
    def __init__(self):
        Object.__init__(self)
        self.constants : [Constant] = None
        self.enforced_size : int = None

class Sampler(Object):
    def __init__(self):
        Object.__init__(self)
        self.type : str = None
        self.texture_name : str = None
        self.slot : int = None
        self.attributes : [(str, str)] = []

class Function(Object):
    def __init__(self):
        Object.__init__(self)
        self.return_value : Type = None
        self.args : [pysl.Declaration] = None
        self.stage : str = None # if entry point

# HLSL convention
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
    'float2x2', 'float2x3', 'float2x4',
    'float3x2', 'float3x3', 'float3x4',
    'float4x2', 'float4x3', 'float4x4'
]

class Keywords:
    # Qualifiers
    OutQualifier = 'out'
    ConstQualifier = 'const'

    # Class decorators
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

    # Semantics
    SVPositionSemantic = 'SV_Position'
    SVVertexIDSemantic = 'SV_VertexID'
    SVInstanceIDSemantic = 'SV_InstanceID'
    SVTargetSemantic = 'SV_Target'
    Semantics = [SVPositionSemantic, SVVertexIDSemantic, SVInstanceIDSemantic, SVTargetSemantic]

    # Opaque types
    SamplerTypes = ['Sampler1D', 'Sampler2D', 'Sampler3D', 'SamplerCube', 'Sampler1DArray', 'Sampler2DArray', 'Sampler2DMS', 'Sampler2DMSArray', 'SamplerCubeArray', 'Sampler1DShadow', 'Sampler2DShadow', 'Sampler1DArrayShadow', 'Sampler2DArrayShadow', 'SamplerCubeShadow', 'SamplerCubeArrayShadow']
    SamplerTypeExts = [t[7:] for t in SamplerTypes]
    Export = 'export'
    
    # Special values
    InputValue = 'input'
    OutputValue = 'output'

    # Metadata keys
    OptionsKey = 'Options'
    ConstantBuffersKey = 'ConstantBuffers'
    SamplerStatesKey = 'SamplerStates'
    TexturesKey = 'Textures'
    VertexShaderKey = 'VertexShader'
    PixelShaderKey = 'PixelShader'
    NameKey = 'Name'
    SizeKey = 'Size'
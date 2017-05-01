import ast

# Language specification that includes the list of cross-module structures
# ------------------------------------------------------------------------------


class Scalar:
    BOOL = 0
    INT = 1
    UINT = 2
    FLOAT = 3
    # DOUBLE = 4 Not supported yet, no f d literals


def scalar_to_cpp(scalar: Scalar) -> str:
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
        # String representation, kept here mostly for ease of use
        self.str: str = None

        # Base type
        self.type: Scalar = None

        # Vector and matrix rows
        self.dim0: int = 1
        self.dim1: int = 1


class Locationable():
    def __init__(self, node: ast.AST = None):
        self.line: int = None
        self.col: int = None
        self.location(node)

    def location(self, node: ast.AST):
        if node:
            self.line = node.lineno
            self.col = node.col_offset


class Declaration(Locationable):
    def __init__(self, type: str = None, name: str = None,
                 qualifiers: [str] = None):
        Locationable.__init__(self)
        self.name: str = None
        self.type: str = None
        self.qualifiers: [str] = None


class Assignment(Declaration):
    def __init__(self):
        Declaration.__init__(self)
        self.value: str = None


class Struct(Locationable):
    def __init__(self):
        Locationable.__init__(self)
        self.name: str = None
        self.elements: [(Type, str)] = None


class InputElement(Locationable):
    def __init__(self):
        Locationable.__init__(self)
        self.name: str = None
        self.type: Type = None
        self.semantic: str = None
        self.conditions: [str] = None


class StageInput(Locationable):
    def __init__(self):
        Locationable.__init__(self)
        self.name: str = None
        self.stages: [str] = None
        self.elements: [InputElement] = None
        self.post_conditions: [str] = None


class Constant(Locationable):
    def __init__(self):
        Locationable.__init__(self)
        self.name: str = None
        self.type: Type = None
        self.array_size: int = None
        self.offset: int = None


class ConstantBuffer(Locationable):
    def __init__(self):
        Locationable.__init__(self)
        self.name: str = None
        self.constants: [Constant] = None
        self.enforced_size: int = None


class Function(Locationable):
    def __init__(self):
        Locationable.__init__(self)
        self.name: str = None
        self.return_value: Type = None
        self.args: [pysl.Declaration] = None
        self.stage: str = None  # if entry point


class Intrinsic(Locationable):
    def __init__(self, name: str = None, num_args: int = None):
        Locationable.__init__(self)
        self.name: str = name
        self.num_args: int = num_args


class Object(Locationable):
    """All types that support method calls should inherit from object"""
    def __init__(self, name: str = None):
        Locationable.__init__(self)
        self.name = name


class Sampler(Object):
    def __init__(self):
        Object.__init__(self)
        self.type: str = None
        self.texture_name: str = None
        self.slot: int = None
        self.attributes: [(str, str)] = []


class Language:
    """
        Namespace containing all the languages string literals.

        For documentation regarding intrinsics see
        https://msdn.microsoft.com/en-us/library/windows/desktop/ff471376(v=vs.85).aspx
        Parameters are the same in GLSL/HLSL if no inline comment is present.
        Intrisics here follow the HLSL naming, GLSL have the same name if
        there is no inline comment.

        Remaining list of dropped intrinsics:
        <name> HLSL | GLSL
        acosh N|Y
        asinh N|Y
        atan Y|N
        clip Y|N
        countbits Y|N
        degrees N|Y
        dst Y|N
        EvaluateAttribute*** Y|???
        fwidthCoarse fwidthFine N|Y
        HLSL has isfinite GLSL has isinf ^^ add '!' when compiling
        lit N|Y ( is it still used ? )
        log10 Y|N
        mad Y|N
        mod N|Y
        noise1,2,3,4 N|Y
        reversebits Y|N
        roundEven N|Y
        sincos Y|N
    """
    intrinsics = [
        Intrinsic('abs', 1),
        Intrinsic('acos', 1),
        Intrinsic('all', 1),
        Intrinsic('any', 1),
        Intrinsic('asin', 1),
        Intrinsic('ceil', 1),
        Intrinsic('clamp', 1),
        Intrinsic('cos', 1),
        Intrinsic('cosh', 1),
        Intrinsic('cross', 2),
        Intrinsic('ddx', 1),
        Intrinsic('ddy', 1),
        Intrinsic('ddx_coarse', 1),
        Intrinsic('ddy_coarse', 1),
        Intrinsic('ddx_fine', 1),
        Intrinsic('ddy_fine', 1),
        Intrinsic('determinant', 1),
        Intrinsic('distance', 2),
        Intrinsic('dot', 2),
        Intrinsic('exp', 1),
        Intrinsic('exp2', 1),
        Intrinsic('faceforward', 3),
        Intrinsic('firstbithigh', 1),
        Intrinsic('firstbitlow', 1),
        Intrinsic('floor', 1),
        Intrinsic('fma', 3),
        Intrinsic('frac', 1),
        Intrinsic('frexp', 2),
        Intrinsic('fwidth', 1),
        Intrinsic('isnan', 1),
        Intrinsic('ldexp', 2),
        Intrinsic('length', 1),
        Intrinsic('lerp', 3),  # GLSL: mix
        Intrinsic('log', 1),
        Intrinsic('log2', 1),
        Intrinsic('max', 2),
        Intrinsic('min', 2),
        Intrinsic('mul', 2),  # GLSL: *
        Intrinsic('modf', 1),
        Intrinsic('noise', 1),
        Intrinsic('normalize', 1),
        Intrinsic('pow', 2),
        Intrinsic('radians', 1),
        Intrinsic('reflect', 2),
        Intrinsic('refract', 3),
        Intrinsic('round', 1),
        Intrinsic('sign', 1),
        Intrinsic('sin', 1),
        Intrinsic('sinh', 1),
        Intrinsic('smoothstep', 3),
        Intrinsic('sqrt', 1),
        Intrinsic('step', 2),
        Intrinsic('tan', 1),
        Intrinsic('tanh', 1),
        Intrinsic('transpose', 1),
        Intrinsic('trunc', 1)
    ]

    def is_intrinsic(func: str) -> bool:
        for intr in Language.intrinsics:
            if intr.name == func:
                return True
        return False

    """
    Unwrapped list of all supported types. (All supproted permutations
    of pysl.Type). see README for details on how to construct native types
    """
    native_types = [
        'bool', 'bool2', 'bool3', 'bool4',
        'int', 'int2', 'int3', 'int4',
        'uint', 'uint2', 'uint3', 'uint4',
        'float', 'float2', 'float3', 'float4',
        'float2x2', 'float2x3', 'float2x4',
        'float3x2', 'float3x3', 'float3x4',
        'float4x2', 'float4x3', 'float4x4'
    ]

    def is_native_type(ctor: str) -> bool:
        return ctor in Language.native_types

    class Qualifier:
        Out = 'out'
        Const = 'const'

    class Decorator:
        VertexShader = 'VS'
        PixelShader = 'PS'
        Stages = [VertexShader, PixelShader]
        In = 'in'
        Out = 'out'
        VertexShaderIn = VertexShader + In
        VertexShaderOut = VertexShader + Out
        PixelShaderIn = PixelShader + In
        PixelShaderOut = PixelShader + Out
        StageInputs = [VertexShaderIn, VertexShaderOut, PixelShaderIn,
                       PixelShaderOut]
        StageInput = 'StageInput'
        ConstantBuffer = 'ConstantBuffer'

    class Semantic:
        """
        Many thanks:https://anteru.net/blog/2016/mapping-between-hlsl-and-glsl/
        """
        ClipDistance = 'SV_ClipDistance'
        CullDistance = 'SV_CullDistance'
        # Coverage
        Depth = 'SV_Depth'
        InstanceID = 'SV_InstanceID'
        IsFrontFace = 'SV_IsFrontFace'
        Position = 'SV_Position'
        PrimitiveID = 'SV_PrimitiveID'
        SampleIndex = 'SV_SampleIndex'
        StencilRef = 'SV_StencilRef'
        Target = 'SV_Target'
        VertexID = 'SV_VertexID'
        All = [ClipDistance, CullDistance, Depth, InstanceID, IsFrontFace,
               Position, PrimitiveID, SampleIndex, StencilRef, Target, VertexID]

    class SpecialAttribute:
        Input = 'input'
        Output = 'output'

    class Export:
        """Metadata export"""
        Keyword = 'export'
        Options = 'Options'
        ConstantBuffers = 'ConstantBuffers'
        Samplers = 'Samplers'
        VertexShader = 'VertexShader'
        PixelShader = 'PixelShader'
        Name = 'Name'
        Size = 'Size'

    class Sampler:
        Types = ['Sampler1D', 'Sampler2D', 'Sampler3D', 'SamplerCube',
                 'Sampler1DArray', 'Sampler2DArray', 'Sampler2DMS',
                 'Sampler2DMSArray', 'SamplerCubeArray', 'Sampler1DShadow',
                 'Sampler2DShadow', 'Sampler1DArrayShadow',
                 'Sampler2DArrayShadow', 'SamplerCubeShadow',
                 'SamplerCubeArrayShadow']

        TypesExts = [t[7:] for t in Types]

        class Method:
            """For documentation and parameters see README"""
            Sample = 'Sample'
            Load = 'Load'
            SampleGrad = 'SampleGrad'
            SampleLevel = 'SampleLevel'
            Gather = 'Gather'
            All = [Sample, Load, SampleGrad, SampleLevel, Gather]

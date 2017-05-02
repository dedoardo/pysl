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
        self.set_location(node)

    def set_location(self, node: ast.AST):
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

    def get_element(self, name: str) -> InputElement:
        for element in self.elements:
            if element.name == name:
                return element
        return None


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
    class NativeType:
        """
        Unwrapped list of all supported types. (All supproted permutations
        of pysl.Type). see README for details on how to construct native types
        """
        _ALL = [
            'bool', 'bool2', 'bool3', 'bool4',
            'int', 'int2', 'int3', 'int4',
            'uint', 'uint2', 'uint3', 'uint4',
            'float', 'float2', 'float3', 'float4',
            'float2x2', 'float2x3', 'float2x4',
            'float3x2', 'float3x3', 'float3x4',
            'float4x2', 'float4x3', 'float4x4'
        ]

        @staticmethod
        def is_in(ctor: str) -> bool:
            return ctor in Language.NativeType._ALL

    class Intrinsic:
        """
            Namespace containing all the languages string literals.

            For documentation regarding intrinsics see
            https://msdn.microsoft.com/en-us/library/windows/desktop/ff471376(v=vs.85).aspx
            Parameters are the same in GLSL/HLSL if no inline comment is present.
            Intrisics here follow the HLSL naming, GLSL have the same name if
            there is no inline comment. Feel free to change the name, but
            **DO NOT CHANGE** the number of parameters, for any reason.

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
        ABS = Intrinsic('abs', 1)
        ACOS = Intrinsic('acos', 1)
        ALL = Intrinsic('all', 1)
        ANY = Intrinsic('any', 1)
        ASIN = Intrinsic('asin', 1)
        CEIL = Intrinsic('ceil', 1)
        CLAMP = Intrinsic('clamp', 1)
        COS = Intrinsic('cos', 1)
        COSH = Intrinsic('cosh', 1)
        CROSS = Intrinsic('cross', 2)
        DDX = Intrinsic('ddx', 1)
        DDY = Intrinsic('ddy', 1)
        DDX_COARSE = Intrinsic('ddx_coarse', 1)
        DDY_COARSE = Intrinsic('ddy_coarse', 1)
        DDX_FINE = Intrinsic('ddx_fine', 1)
        DDY_FINE = Intrinsic('ddy_fine', 1)
        DETERMINANT = Intrinsic('determinant', 1)
        DISTANCE = Intrinsic('distance', 2)
        DOT = Intrinsic('dot', 2)
        EXP = Intrinsic('exp', 1)
        EXP2 = Intrinsic('exp2', 1)
        FACEFORWARD = Intrinsic('faceforward', 3)
        FIRSTBITHIGH = Intrinsic('firstbithigh', 1)
        FIRSTBITLOW = Intrinsic('firstbitlow', 1)
        FLOOR = Intrinsic('floor', 1)
        FMA = Intrinsic('fma', 3)
        FRAC = Intrinsic('frac', 1)
        FREXP = Intrinsic('frexp', 2)
        FWIDTH = Intrinsic('fwidth', 1)
        ISNAN = Intrinsic('isnan', 1)
        LDEXP = Intrinsic('ldexp', 2)
        LENGTH = Intrinsic('length', 1)
        LERP = Intrinsic('lerp', 3)  # GLSL: mix
        LOG = Intrinsic('log', 1)
        LOG2 = Intrinsic('log2', 1)
        MAX = Intrinsic('max', 2)
        MIN = Intrinsic('min', 2)
        MUL = Intrinsic('mul', 2)  # GLSL: *
        MODF = Intrinsic('modf', 1)
        NOISE = Intrinsic('noise', 1)
        NORMALIZE = Intrinsic('normalize', 1)
        POW = Intrinsic('pow', 2)
        RADIANS = Intrinsic('radians', 1)
        REFLECT = Intrinsic('reflect', 2)
        REFRACT = Intrinsic('refract', 3)
        ROUND = Intrinsic('round', 1)
        SIGN = Intrinsic('sign', 1)
        SIN = Intrinsic('sin', 1)
        SINH = Intrinsic('sinh', 1)
        SMOOTHSTEP = Intrinsic('smoothstep', 3)
        SQRT = Intrinsic('sqrt', 1)
        STEP = Intrinsic('step', 2)
        TAN = Intrinsic('tan', 1)
        TANH = Intrinsic('tanh', 1)
        TRANSPOSE = Intrinsic('transpose', 1)
        TRUNC = Intrinsic('trunc', 1)

        ROW = 'row'
        COL = 'col'
        ROW1 = Intrinsic(ROW + '1', 2)
        COL1 = Intrinsic(COL + '1', 2)
        ROW2 = Intrinsic(ROW + '2', 2)
        COL2 = Intrinsic(COL + '2', 2)
        ROW3 = Intrinsic(ROW + '3', 2)
        COL3 = Intrinsic(COL + '3', 2)
        ROW4 = Intrinsic(ROW + '4', 2)
        COL4 = Intrinsic(COL + '4', 2)

        _ALL = [ABS, ACOS, ALL, ANY, ASIN, CEIL, CLAMP, COS, COSH, CROSS, DDX, DDY,
                DDX_COARSE, DDY_COARSE, DDX_FINE, DDY_FINE, DETERMINANT, DISTANCE,
                DOT, EXP, EXP2, FACEFORWARD, FIRSTBITHIGH, FIRSTBITLOW, FLOOR,
                FMA, FRAC, FREXP, FWIDTH, ISNAN, LDEXP, LENGTH, LERP, LOG, LOG2, MAX,
                MIN, MUL, MODF, NOISE, NORMALIZE, POW, RADIANS, REFLECT, REFRACT,
                ROUND, SIGN, SIN, SINH, SMOOTHSTEP, SQRT, STEP, TAN, TANH, TRANSPOSE,
                TRUNC, ROW1, COL1, ROW2, COL2, ROW3, COL3, ROW4, COL4]

        @staticmethod
        def is_in(func: str) -> bool:
            for intr in Language.Intrinsic._ALL:
                if intr.name == func:
                    return True
            return False

    class Qualifier:
        OUT = 'out'
        CONST = 'const'

    class Decorator:
        VERTEX_SHADER = 'VS'
        PIXEL_SHADER = 'PS'
        STAGES = [VERTEX_SHADER, PIXEL_SHADER]
        IN = 'in'
        OUT = 'out'
        VERTEX_SHADER_IN = VERTEX_SHADER + IN
        VERTEX_SHADER_OUT = VERTEX_SHADER + OUT
        PIXEL_SHADER_IN = PIXEL_SHADER + IN
        PIXEL_SHADER_OUT = PIXEL_SHADER + OUT
        STAGE_INPUTS = [VERTEX_SHADER_IN, VERTEX_SHADER_OUT,
                        PIXEL_SHADER_IN, PIXEL_SHADER_OUT]
        STAGE_INPUT = 'StageInput'
        CONSTANT_BUFFER = 'ConstantBuffer'

    class Semantic:
        """
        Many thanks:https://anteru.net/blog/2016/mapping-between-hlsl-and-glsl/
        """
        CLIP_DISTANCE = 'SV_ClipDistance'
        CULL_DISTANCE = 'SV_CullDistance'
        # Coverage
        DEPTH = 'SV_Depth'
        INSTANCE_ID = 'SV_InstanceID'
        IS_FRONT_FACE = 'SV_IsFrontFace'
        POSITION = 'SV_Position'
        PRIMITIVE_ID = 'SV_PrimitiveID'
        SAMPLE_INDEX = 'SV_SampleIndex'
        STENCIL_REF = 'SV_StencilRef'
        TARGET = 'SV_Target'
        VERTEX_ID = 'SV_VertexID'
        _ALL = [CLIP_DISTANCE, CULL_DISTANCE, DEPTH, INSTANCE_ID, IS_FRONT_FACE,
                POSITION, PRIMITIVE_ID, SAMPLE_INDEX, STENCIL_REF, TARGET, VERTEX_ID]

    class SpecialAttribute:
        INPUT = 'input'
        OUTPUT = 'output'

    class Export:
        """Metadata export"""
        KEYWORD = 'export'
        OPTIONS = 'Options'
        CONSTANT_BUFFERS = 'ConstantBuffers'
        SAMPLERS = 'Samplers'
        VERTEX_SHADER = 'VertexShader'
        PIXEL_SHADER = 'PixelShader'
        NAME = 'Name'
        SIZE = 'Size'

    class Sampler:
        TYPES = ['Sampler1D', 'Sampler2D', 'Sampler3D', 'SamplerCube',
                 'Sampler1DArray', 'Sampler2DArray', 'Sampler2DMS',
                 'Sampler2DMSArray', 'SamplerCubeArray', 'Sampler1DShadow',
                 'Sampler2DShadow', 'Sampler1DArrayShadow',
                 'Sampler2DArrayShadow', 'SamplerCubeShadow',
                 'SamplerCubeArrayShadow']

        TYPE_EXTS = [t[7:] for t in TYPES]

        class Method:
            """For documentation and parameters see README"""
            SAMPLE = 'Sample'
            LOAD = 'Load'
            SAMPLE_GRAD = 'SampleGrad'
            SAMPLE_LEVEL = 'SampleLevel'
            GATHER = 'Gather'
            _ALL = [SAMPLE, LOAD, SAMPLE_GRAD, SAMPLE_LEVEL, GATHER]

            @staticmethod
            def is_in(method: str) -> bool:
                return method in Language.Sampler.Method._ALL
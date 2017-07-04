from . import pysl
from . import hlsl5
from . import glsl45
from . import exporter
from .error import error


g_hlsl: bool = False
g_glsl: bool = False
g_symbols: dict = False


def init(hlsl_path: str, glsl_path: str) -> bool:
    global g_hlsl, g_glsl, g_symbols
    """Initializes the backends and the symbol table -> bool"""
    if hlsl_path:
        g_hlsl = hlsl5.init(hlsl_path)
        if not g_hlsl:
            return False
    if glsl_path:
        g_glsl = glsl45.init(glsl_path)
        if not g_glsl:
            return False

    g_symbols = {}
    return True
    

def finalize():
    if g_hlsl:
        hlsl5.finalize()
    if g_glsl:
        glsl45.finalize()


def text(string: str):
    """Writes text directly to the output such as preprocessor strings"""
    if g_hlsl:
        hlsl5.write(string)
    if g_glsl:
        glsl45.write(string)

# Top-level
# ------------------------------------------------------------------------------


def options(strings: [str]):
    """Adds a bunch of compilation options"""
    exporter.options(strings)

    # HlslTools helper: https://github.com/tgjones/HlslTools
    if g_hlsl:
        hlsl5.options(strings)
    if g_glsl:
        glsl45.options(strings)


def decl_struct(struct: pysl.Struct):
    if struct.name in g_symbols:
        error(struct, "Already defined symbol: {0} as {1}".format(
              g_symbols, g_symbols[struct.name]))
        return

    if g_hlsl:
        hlsl5.struct(struct)
    if g_glsl:
        glsl45.struct(struct)

    g_symbols[struct.name] = struct


def decl_stage_input(si: pysl.StageInput):
    """Stage input declaration"""
    if si.name in g_symbols:
        error(si, "Already defined symbol: {0} as {1}".format(
              g_symbols, g_symbols[si.name]))
        return

    prev_sis = [v for k, v in g_symbols.items() if isinstance(v,
                                                              pysl.StageInput)]
    if g_hlsl:
        hlsl5.stage_input(si, prev_sis)
    if g_glsl:
        glsl45.stage_input(si, prev_sis)

    g_symbols[si.name] = si


def constant_buffer(cbuffer: pysl.ConstantBuffer):
    """Constant buffer declaration"""
    if cbuffer.name in g_symbols:
        error(cbuffer, "Already defined symbol: {0} as {1}".format(
              g_symbols, g_symbols[cbuffer.name]))
        return

    exporter.constant_buffer(cbuffer)
    if g_hlsl:
        hlsl5.constant_buffer(cbuffer)
    if g_glsl:
        glsl45.constant_buffer(cbuffer)

    g_symbols[cbuffer.name] = cbuffer


def sampler(sampler: pysl.Sampler):
    """Sampler state declaration"""
    if sampler.name in g_symbols:
        error(sampler, "Already defined symbol: {0} as {1}".format(
              g_symbols, g_symbols[sampler.name]))
        return

    exporter.sampler(sampler)
    if g_hlsl:
        hlsl5.sampler(sampler)
    if g_glsl:
        glsl45.sampler(sampler)

    g_symbols[sampler.name] = sampler


def _parameter(arg: pysl.Declaration):
    """Writes a function parameter """
    if (arg.type not in pysl.TYPES and(arg.type not in g_symbols or(
       not isinstance(g_symbols[arg.type], pysl.StageInput) and
       not isinstance(g_symbols[arg.type], pysl.Struct)))):
        error(arg, "Type not found: {0}".format(arg.type))
        return

    if g_hlsl:
        hlsl5.declaration(arg)
    if g_glsl:
        glsl45.declaration(arg)


def function_beg(func: pysl.Function):
    if func.stage:
        func_in = None
        func_out = None
        # Special case, looking up input
        for name, obj in g_symbols.items():
            if isinstance(obj, pysl.StageInput):
                for stage in obj.stages:
                    if func.stage + pysl.Language.Decorator.IN == stage:
                        if func_in:
                            error(func, "Multiple possible input values found for entry point: {0}".format(func.name))
                        func_in = obj
                    if func.stage + pysl.Language.Decorator.OUT == stage:
                        if func_out:
                            error(func, "Multiple possible output values found for entry point: {0}".format(func.name))
                        func_out = obj

        if func_in is None or func_out is None:
            error(func, "Undeclared input or output for function stage: {0}:{1}".format(
                func.name, func.stage))
            return

        exporter.entry_point(func)
        if g_hlsl:
            hlsl5.entry_point_beg(func, func_in, func_out)
        if g_glsl:
            glsl45.entry_point_beg(func, func_in, func_out)
    else:
        # Standard C-like function declaration
        text('{0} {1}('.format(func.return_value, func.name))
        for arg in func.args:
            _parameter(arg)
        text(')\n{\n')
        g_symbols[func.name] = func


def function_end(func: pysl.Function):
    if func.stage:
        if g_hlsl:
            hlsl5.entry_point_end(func)
        if g_glsl:
            glsl45.entry_point_end(func)
    else:
        text('};\n\n')

def entry_point_ret(func: pysl.Function):
    if func.stage:
        if g_hlsl:
            hlsl5.entry_point_ret(func)
        if g_glsl:
            glsl45.entry_point_ret(func)

# BLOCK-LEVEL
# ------------------------------------------------------------------------------


def declaration(assignment: pysl.Assignment):
    """Plain old declaration, usually a ast.AnnAssign"""

    # Type is either a scalar basic type
    # or a StageInput. No other type is allowed at the block level
    if (not pysl.Language.NativeType.is_in(assignment.type) and(assignment.type not in g_symbols or(
       not isinstance(g_symbols[assignment.type], pysl.StageInput) and
       not isinstance(g_symbols[assignment.type], pysl.Struct)
       ))):
        error(assignment, "Type not found: {0}".format(assignment.type))
        return

    if g_hlsl:
        hlsl5.declaration(assignment)
    if g_glsl:
        glsl45.declaration(assignment)


def method_call(loc: pysl.Locationable, caller: str, name: str, args):
    """Method calls are used simply as a stylistic way to expose intrinsics"""
    global g_hlsl, g_glsl

    obj = None

    # Right now method calls are supported exclusively by textures, thus
    # the caller has to be registered in the symbol table
    if caller not in g_symbols or not isinstance(g_symbols[caller], pysl.Sampler):
        error(loc, "Expected sampler object in method call: {0}".format(caller))
        return
    obj = g_symbols[caller]

    if g_hlsl:
        old, g_glsl = g_glsl, False
        hlsl5.method_call(obj, name, args)
        g_glsl = old
    if g_glsl:
        old, g_hlsl = g_hlsl, False
        glsl45.method_call(obj, name, args)
        g_hlsl = old


def constructor(loc: pysl.Locationable, typename: str, args):
    """Type constructor, assuming that typename is in pysl.TYPES"""
    global g_hlsl, g_glsl

    exp_num_args = pysl.Language.NativeType.num_arguments(typename)
    if len(args) not in exp_num_args:
        error(loc, "{0} constructor expected {1} arguments, but found: {2}".format(typename, exp_num_args, len(args)))
        return

    if g_hlsl:
        old, g_glsl = g_glsl, False
        hlsl5.constructor(typename, args)
        g_glsl = old
    if g_glsl:
        old, g_hlsl = g_hlsl, False
        glsl45.constructor(typename, args)
        g_hlsl = old


def intrinsic(loc: pysl.Locationable, type: str, args):
    """Intrinsic function, assuming that itype is in pysl.INTRINSICS"""
    global g_hlsl, g_glsl

    intrin = pysl.Language.Intrinsic.find(type)
    if intrin:
        if len(args) != pysl.Language.Intrinsic.find(type).num_args:
            error(loc, "Invalid number of arguments for intrinsic: {0}(). expected {1}, but found {2}".format(
                intrin.name, intrin.num_args, len(args)))
            return

        if g_hlsl:
            old, g_glsl = g_glsl, False
            hlsl5.intrinsic(type, args)
            g_glsl = old
        if g_glsl:
            old, g_hlsl = g_hlsl, False
            glsl45.intrinsic(type, args)
            g_hlsl = old


def cast(type: str, val):
    global g_hlsl, g_glsl

    if g_hlsl:
        old, g_glsl = g_glsl, False
        hlsl5.cast(type, val)
        g_glsl = old
    if g_glsl:
        old, g_hlsl = g_hlsl, False
        glsl45.cast(type, val)
        g_hlsl = old


def function_call(loc: pysl.Locationable, function: str, args):
    """Function call encountered, same for both backends"""
    global g_hlsl, g_glsl

    if function not in g_symbols or not isinstance(g_symbols[function], pysl.Function):
        error(loc, "Unrecognized function name: {0}".format(function))
        return

    text('{0}('.format(function))
    for i in range(len(args) - 1):
        args[i]()
        text(', ')
    if args:
        args[-1]()
    text(')')


# Location is specified as function is too generic
def attribute(loc: pysl.Locationable, func: pysl.Function, attribute: str, value: str):
    if pysl.Language.SpecialAttribute.is_in(attribute):
        if func.stage:
            src_stage = func.stage
            # Looking up associated stageinput
            if attribute == pysl.Language.SpecialAttribute.INPUT:
                src_stage += pysl.Language.Decorator.IN
            elif attribute == pysl.Language.SpecialAttribute.OUTPUT:
                src_stage += pysl.Language.Decorator.OUT

            si = None
            for name, symbol in g_symbols.items():
                if isinstance(symbol, pysl.StageInput) and src_stage in symbol.stages:
                    si = symbol

            if si is None:
                error(loc, "Failed to find StageInput that matches attribute: {0} inside: {1}".format(attribute, func.name))
                return

            if si.get_element(value) is None:
                error(loc, "Undeclared attribute: {0}.{1} inside: {2}".format(attribute, value, func.name))
                return

            if g_hlsl:
                hlsl5.special_attribute(src_stage, si, attribute, value)
            if g_glsl:
                glsl45.special_attribute(src_stage, si, attribute, value)
        else:
            for arg in func.args:
                if arg.name == attribute:
                    text('{0}.'.format(attribute))
                    return
                error(loc, "Special attribute: {0} has valid meaning only inside an entry point, see documentation for more details".format(attribute))
    else:
        obj = g_symbols[attribute]
        if obj:
            if g_hlsl:
                hlsl5.member(obj, value)
            if g_glsl:
                glsl45.member(obj, value)
        else:
            text('{0}.'.format(attribute))
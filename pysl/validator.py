import subprocess
import os
import sys
import shutil
from . import pysl
from . import exporter
from .error import error


def validate_hlsl5(src_path: str, compiler_path: str) -> bool:
    quoted_compiler_path = '"{0}"'.format(compiler_path)
    vertex_shader = exporter.query_entry_point(pysl.Language.Decorator.VERTEX_SHADER)
    pixel_shader = exporter.query_entry_point(pysl.Language.Decorator.PIXEL_SHADER)
    out = 'out.tmp'

    if vertex_shader:
        cmd = '{0} /nologo /E {1} /T vs_5_0 /Fo {2} {3}'.format(quoted_compiler_path, vertex_shader, out, src_path)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            error(None, "Failed to compile hlsl5 VertexShader: {0}".format(src_path))
            for err_line in stderr.decode().splitlines():
                if err_line:
                    sys.stderr.write('{0}: {1}\n'.format(os.path.basename(compiler_path), err_line))
            return False

    if vertex_shader:
        cmd = '{0} /nologo /E {1} /T ps_5_0 /Fo {2} {3}'.format(quoted_compiler_path, pixel_shader, out, src_path)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            error(None, "Failed to compile hlsl5 PixelShader: {0}".format(src_path))
            for err_line in stderr.decode().splitlines():
                if err_line:
                    sys.stderr.write('{0}: {1}\n'.format(os.path.basename(compiler_path), err_line))
            return False

    try:
        os.remove(out)
    except FileNotFoundError:
        pass

    return True


def validate_glsl45(src_path: str, compiler_path: str) -> bool:
    quoted_compiler_path = '"{0}"'.format(compiler_path)    
    vertex_shader = exporter.query_entry_point(pysl.Language.Decorator.VERTEX_SHADER)
    pixel_shader = exporter.query_entry_point(pysl.Language.Decorator.PIXEL_SHADER)

    src = ''
    with open(src_path, 'r') as fs:
        src = fs.read()

    if vertex_shader:
        out = src_path + '.vert'
        hdr = '#version 450\n\n#define {0} main\n#define PYSL_VERTEX_SHADER\n'.format(vertex_shader)
        with open(out, 'w') as fs:
            fs.write(hdr)
            fs.write(src)
        cmd = '{0} {1}'.format(quoted_compiler_path, out)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            error(None, "Failed to compile glsl45 VertexShader: {0}".format(src_path))
            for line in stdout.decode().splitlines():
                if line.startswith('ERROR') or line.startswith('Warning'):
                    sys.stderr.write('{0}: {1}\n'.format(os.path.basename(compiler_path), line))
            return False

    if pixel_shader:
        out = src_path + '.frag'
        hdr = '#version 450\n\n#define {0} main\n#define PYSL_PIXEL_SHADER\n'.format(pixel_shader)
        with open(out, 'w') as fs:
            fs.write(hdr)
            fs.write(src)
        cmd = '{0} {1}'.format(quoted_compiler_path, out)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            error(None, "Failed to compile glsl45 PixelShader: {0}".format(src_path))
            for line in stdout.decode().splitlines():
                if line.startswith('ERROR') or line.startswith('Warning'):
                    sys.stderr.write('{0}: {1}\n'.format(os.path.basename(compiler_path), line))
            return False

    return True
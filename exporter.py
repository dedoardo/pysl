import pysl


def init(json_path: str, cpp_path: str):
    return True

def finalize():
    pass


def options(strings: [str]):
    pass


def struct(struct: pysl.Struct):
    pass


def stage_input(si: pysl.StageInput):
    pass


def constant_buffer(si: pysl.ConstantBuffer):
    pass


def sampler(sampler: pysl.Sampler):
    pass
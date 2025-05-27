from amlb.utils import call_script_in_same_dir

def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", *args, **kwargs)

def run(*args, **kwargs):
    from .exec import run
    return run(*args, **kwargs)

def docker_commands(*args, **kwargs):
    return """
RUN {here}/setup.sh
""".format(here="/".join(__file__.split("/")[:-1]))
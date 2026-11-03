# Minimal stand-in for `pyglet`: f110_env.py sets an option and imports gl at
# module load. Rendering is never exercised by the reference generator.
options = {}
from . import gl  # noqa: F401

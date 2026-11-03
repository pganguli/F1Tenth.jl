# Minimal stand-in for the `gym` package: just enough for f110_env.py and
# waypoint_follow.py to import. The reference generator never uses gym itself.
from . import error, spaces, utils  # noqa: F401


class Env:
    pass

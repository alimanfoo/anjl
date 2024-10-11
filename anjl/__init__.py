from ._canonical import canonical_nj  # noqa
from ._rapid import rapid_nj  # noqa
from ._util import to_string, map_internal_to_leaves  # noqa
from ._layout import layout_equal_angle  # noqa
from ._plot import plot, paint_internal  # noqa
from . import data  # noqa
from . import params  # noqa
import importlib.metadata as _metadata


# This will read version from pyproject.toml.
__version__ = _metadata.version(__name__)

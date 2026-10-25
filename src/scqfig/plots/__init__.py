"""Plot families. Every function draws on a matplotlib Axes you pass in and takes its colours from
the active style, so panels compose freely and switch style with scqfig.use()."""
from ._base import annotated_matrix, colorbar, heatmap  # noqa: F401
from .compare import *         # noqa: F401,F403
from .darkmatter import *      # noqa: F401,F403
from .design import *          # noqa: F401,F403
from .noise import *           # noqa: F401,F403
from .processor import *       # noqa: F401,F403
from .readout import *         # noqa: F401,F403
from .spectrum import *        # noqa: F401,F403
from .states import *          # noqa: F401,F403
from .timedomain import *      # noqa: F401,F403

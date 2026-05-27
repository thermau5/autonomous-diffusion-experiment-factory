from .base import Sampler, SamplerOutput, register_sampler, get_sampler, list_samplers

# Each submodule registers its sampler id(s) on import; the noqa comments keep
# linters from removing the side-effect imports.
from . import edm          # noqa: F401  -- edm_euler, edm_heun
from . import schedule     # noqa: F401  -- karras_schedule, uniform_schedule
from . import ddim         # noqa: F401  -- ddim
from . import ddpm         # noqa: F401  -- ddpm_ancestral
from . import dpm_solver   # noqa: F401  -- dpm_solver
from . import dpm_solver_pp  # noqa: F401  -- dpm_solver_pp
from . import unipc        # noqa: F401  -- unipc
from . import deis         # noqa: F401  -- deis
from . import pndm         # noqa: F401  -- pndm
from . import restart      # noqa: F401  -- restart
from . import proposed_control  # noqa: F401  -- registers "proposed_heun" (Heun core; file still named proposed_control for backward-compat)
from . import proposed_dpmpp    # noqa: F401  -- proposed_dpmpp   (DPM-Solver++ 2M core)
from . import proposed_unipc    # noqa: F401  -- proposed_unipc   (UniPC predictor-corrector core)
from . import proposed_deis     # noqa: F401  -- proposed_deis    (DEIS tAB-2 core)
from . import proposed_restart  # noqa: F401  -- proposed_restart (Restart wrapper + Heun)
from . import ays_unipc          # noqa: F401  -- ays_unipc        (AYS learned schedule on UniPC core)

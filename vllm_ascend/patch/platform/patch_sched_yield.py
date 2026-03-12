import sys

import vllm.distributed.utils
import vllm_ascend.envs as envs_ascend

# Try to import platform classes, handle if they don't exist
try:
    from vllm.platforms import CpuArchEnum, Platform
    is_arm = (Platform.get_cpu_architecture() == CpuArchEnum.ARM)
    USE_SCHED_YIELD = (
        ((sys.version_info[:3] >= (3, 11, 1)) or
         (sys.version_info[:2] == (3, 10) and sys.version_info[2] >= 8))
        and not is_arm)
except (ImportError, AttributeError):
    # Platform classes don't exist in this vLLM version
    # Use default value
    USE_SCHED_YIELD = True

vllm.distributed.utils.USE_SCHED_YIELD = USE_SCHED_YIELD

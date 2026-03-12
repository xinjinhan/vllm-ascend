import sys

import vllm.distributed.utils
import vllm_ascend.envs as envs_ascend

# In CPU simulation mode, skip this platform-specific patch
if not envs_ascend.VLLM_ASCEND_ENABLE_CPU_SIMULATION:
    from vllm.platforms import CpuArchEnum, Platform

    is_arm = (Platform.get_cpu_architecture() == CpuArchEnum.ARM)

    USE_SCHED_YIELD = (
        ((sys.version_info[:3] >= (3, 11, 1)) or
         (sys.version_info[:2] == (3, 10) and sys.version_info[2] >= 8))
        and not is_arm)
else:
    # CPU simulation mode: use default value
    USE_SCHED_YIELD = True

vllm.distributed.utils.USE_SCHED_YIELD = USE_SCHED_YIELD

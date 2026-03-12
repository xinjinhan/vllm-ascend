# vllm-ascend CPU 模拟模式安装部署与使用手册

## 一、概述

CPU 模拟模式是 vllm-ascend 的一项功能特性，允许在没有任何 NPU 硬件的环境中运行 vllm-ascend。该模式通过 Mock NPU 操作并返回假值，结合可配置的时间延迟，模拟 NPU 的执行行为。

### 主要特性

- **无需 NPU 硬件**：纯 CPU 环境即可运行
- **快速验证**：用于框架集成测试和开发调试
- **时间模拟**：可配置的 `CONSUMED_TIME` 模拟执行时间
- **版本同步**：与主线版本保持同步演进

---

## 二、环境要求

### 2.1 基础环境

| 项目 | 要求 |
|------|------|
| Python | >= 3.8 |
| PyTorch | >= 2.0 |
| vllm-ascend | v0.13.0 或更高版本 |

### 2.2 操作系统

- Linux (Ubuntu 20.04+ / CentOS 8+)
- Windows (WSL2)

### 2.3 可选：NPU 环境（用于对比测试）

如果需要在同一环境切换真实 NPU 和模拟模式，请参考 [vllm-ascend 官方安装指南](https://github.com/vllm-project/vllm-ascend)。

---

## 三、安装部署

### 3.1 方式一：从源码安装（推荐）

```bash
# 1. 克隆代码仓库
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

# 2. 切换到对应版本分支
git checkout releases/v0.13.0

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 vllm-ascend（CPU 模拟模式不需要 CANN）
pip install -e .
```

### 3.2 方式二：仅使用 CPU 模拟模式

如果只需要 CPU 模拟功能，可以简化安装：

```bash
# 1. 安装基础依赖
pip install torch>=2.0
pip install vllm>=0.6.0

# 2. 克隆并安装 vllm-ascend（不安装 CANN）
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout releases/v0.13.0
pip install -e . --no-deps  # 跳过 CANN 依赖

# 3. 安装必要的 Python 包
pip install transformers accelerate
```

### 3.3 方式三：从 PyPI 安装（未来版本）

```bash
# 即将支持
pip install vllm-ascend
```

---

## 四、快速开始

### 4.1 启用模拟模式

#### 方法一：环境变量（推荐）

```bash
# 在运行前设置环境变量
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=5

# 启动服务
vllm serve Qwen/Qwen2-0.5B-Instruct --host 0.0.0.0 --port 8000
```

#### 方法二：Python 代码中设置

```python
import os
os.environ["VLLM_ASCEND_ENABLE_CPU_SIMULATION"] = "1"
os.environ["VLLM_ASCEND_CPU_SIMULATED_TIME_MS"] = "10"

# 必须在导入 vllm 之前设置
from vllm import LLM

# 创建推理引擎
llm = LLM(
    model="Qwen/Qwen2-0.5B-Instruct",
    trust_remote_code=True,
)

# 执行推理
outputs = llm.generate(["Hello, world!"])
print(outputs[0].outputs[0].text)
```

### 4.2 验证模拟模式

运行以下命令验证模拟模式是否正常工作：

```bash
# 设置环境变量
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1

# 运行验证脚本
python -c "
import os
os.environ['VLLM_ASCEND_ENABLE_CPU_SIMULATION'] = '1'

from vllm_ascend._cpu_simulation.mock_torch_npu import npu
print(f'Device Name: {npu.get_device_name()}')
print(f'SOC Version: {npu.get_soc_version()}')
print(f'Device Properties: {npu.get_device_properties()}')
print('CPU Simulation Mode: OK')
"
```

预期输出：

```
Device Name: Ascend NPU (Mock)
SOC Version: 220
Device Properties: <__main__.MockDeviceProperties object at ...>
CPU Simulation Mode: OK
```

---

## 五、配置参数

### 5.1 环境变量

| 环境变量 | 说明 | 类型 | 默认值 | 取值范围 |
|---------|------|------|--------|----------|
| `VLLM_ASCEND_ENABLE_CPU_SIMULATION` | 启用/禁用模拟模式 | bool | 0 | 0, 1 |
| `VLLM_ASCEND_CPU_SIMULATED_TIME_MS` | 单次操作模拟时间(毫秒) | float | 0 | 0 ~ 10000 |

### 5.2 时间模拟配置示例

```bash
# 快速模式（无延迟，用于功能验证）
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=0

# 轻度模拟（5ms/操作）
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=5

# 中度模拟（50ms/操作）
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=50

# 重度模拟（200ms/操作）
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=200
```

---

## 六、使用场景

### 6.1 开发调试

```bash
# 启动调试模式（无延迟）
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=0

# 运行 vllm
vllm serve <model> --port 8000
```

### 6.2 CI/CD 集成

```yaml
# .github/workflows/test.yml 示例
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run tests with CPU simulation
        env:
          VLLM_ASCEND_ENABLE_CPU_SIMULATION: 1
          VLLM_ASCEND_CPU_SIMULATED_TIME_MS: 0
        run: |
          pytest tests/ -v
```

### 6.3 性能基准测试

```bash
# 模拟 NPU 性能（每个操作延迟 10ms）
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=10

# 运行基准测试
python benchmark.py --model Qwen/Qwen2-0.5B-Instruct
```

### 6.4 教学演示

```bash
# 演示模式（带延迟，更真实）
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=20

# 运行演示
python demo.py
```

---

## 七、API 参考

### 7.1 Python API

```python
from vllm_ascend._cpu_simulation import (
    is_cpu_simulation_enabled,
    get_simulated_time_ms,
    init_cpu_simulation
)

# 检查是否启用模拟模式
if is_cpu_simulation_enabled():
    print("CPU Simulation Mode is enabled")

# 获取配置的模拟时间
delay = get_simulated_time_ms()
print(f"Simulated time per operation: {delay}ms")

# 手动初始化模拟模块
init_cpu_simulation()
```

### 7.2 时间模拟器

```python
from vllm_ascend._cpu_simulation.time_simulator import (
    simulate_time,
    TimeSimulator
)

# 使用装饰器
@simulate_time
def my_npu_operation():
    pass

# 使用上下文管理器
with TimeSimulator("my_operation"):
    # 执行操作
    pass
```

### 7.3 Mock 操作示例

```python
from vllm_ascend._cpu_simulation.mock_atb import atb
import torch

# ATB linear 返回零 tensor
input_tensor = torch.randn(2, 10, 512)
weight = torch.randn(1024, 512)
output = atb.linear(input_tensor, weight)
# output.shape == (2, 10, 1024)
# torch.all(output == 0) == True
```

---

## 八、故障排除

### 8.1 常见问题

#### 问题 1：导入失败

**症状**：
```
ImportError: No module named 'vllm_ascend'
```

**解决方案**：
```bash
# 重新安装 vllm-ascend
pip install -e .
```

#### 问题 2：模拟模式未生效

**症状**：仍然尝试访问真实 NPU

**解决方案**：
```bash
# 确认环境变量设置正确
echo $VLLM_ASCEND_ENABLE_CPU_SIMULATION
# 应该输出 1

# 在 Python 代码中检查
python -c "import os; print(os.environ.get('VLLM_ASCEND_ENABLE_CPU_SIMULATION'))"
```

#### 问题 3：推理结果不正确

**现象**：这是**预期行为**

**说明**：CPU 模拟模式返回的是假值（零 tensor），目的是验证流程正确性，而不是得到正确的推理结果。

### 8.2 调试模式

```bash
# 启用详细日志
export VLLM_LOG_LEVEL=DEBUG
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1

# 运行并查看日志
vllm serve <model> 2>&1 | grep -i simulation
```

---

## 九、版本信息

| 版本 | vllm-ascend 版本 | 说明 |
|------|------------------|------|
| v0.13.0 | v0.13.0 | 初始版本 |
| 后续版本 | 后续版本 | 随主线同步更新 |

---

## 十、附录

### A. 相关文件路径

```
vllm-ascend/
├── vllm_ascend/
│   ├── _cpu_simulation/           # 模拟模式核心代码
│   │   ├── __init__.py
│   │   ├── mock_torch_npu.py
│   │   ├── mock_atb.py
│   │   └── time_simulator.py
│   ├── patch/
│   │   └── platform/
│   │       └── patch_cpu_simulation.py
│   └── envs.py
└── docs/
    ├── cpu_simulation.md          # 开发文档
    └── cpu_simulation_uml.md     # UML 图表
```

### B. 相关链接

- [vllm-ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [vllm-ascend 文档](https://docs.vllm.ai/projects/ascend/)
- [GitHub Issues](https://github.com/vllm-project/vllm-ascend/issues)

### C. 反馈与支持

如果遇到问题，请通过以下方式获取支持：

1. 提交 GitHub Issue
2. 查看现有 Issue 和解决方案
3. 参与社区讨论

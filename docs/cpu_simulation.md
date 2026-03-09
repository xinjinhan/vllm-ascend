# vllm-ascend NPU 模拟模式开发文档

## 一、背景与目标

### 1.1 改造目的

在 vllm-ascend 项目的实际开发和运维过程中，存在以下痛点：

| 痛点 | 说明 |
|------|------|
| **NPU 资源依赖** | 开发、测试、验证都需要真实的 NPU 硬件，资源稀缺且成本高昂 |
| **大规模部署验证成本** | 验证大规模部署场景（如多节点推理）需要大量 NPU 集群资源 |
| **CI/CD 流程受限** | 自动化测试难以在纯 CPU 环境中运行 |
| **开发调试效率低** | 问题复现和调试需要排队等待 NPU 设备 |

本功能旨在实现：

- **减少对 NPU 资源的依赖**：无需真实 NPU 即可进行大部分开发测试工作
- **快速验证框架性能**：快速验证 vllm-ascend 框架的集成正确性和流程完整性
- **降低大规模部署验证成本**：在 CPU 环境中模拟多节点部署场景，降低验证成本

### 1.2 设计目标

1. **功能目标**：在纯 CPU 环境中运行 vllm-ascend，验证推理流程的正确性
2. **性能模拟**：通过可配置的 `CONSUMED_TIME` 模拟 NPU 执行时间
3. **版本演进**：模拟版本与主线版本保持同步演进

### 1.3 核心设计理念

所有 NPU 操作返回假值（Mock 结果），不执行真实计算，通过可配置的 `CONSUMED_TIME` 模拟执行时间。

---

## 二、版本演进设计

### 2.1 版本管理策略

NPU 模拟模式作为 vllm-ascend 的内置功能，与主线版本保持同步演进。

#### 2.1.1 版本号对应

```
vllm-ascend 版本号格式：v{X}.{Y}.{Z}-{state}

模拟模式版本与主线版本完全对应：
- vllm-ascend v0.14.0 -> 模拟模式 v0.14.0
- vllm-ascend v0.15.0 -> 模拟模式 v0.15.0
```

#### 2.1.2 代码组织

```
vllm_ascend/
├── _cpu_simulation/           # 模拟模式核心代码
│   ├── __init__.py           # 模块入口
│   ├── mock_torch_npu.py    # torch_npu Mock
│   ├── mock_torch_npu_ops.py # torch.npu Mock
│   ├── mock_hccl.py         # HCCL Mock
│   ├── mock_atb.py          # ATB Mock
│   └── time_simulator.py    # 时间模拟器
├── patch/
│   └── platform/
│       └── patch_cpu_simulation.py # 模拟模式 Patch
└── ...
```

#### 2.1.3 演进规则

| 场景 | 处理方式 |
|------|----------|
| 新增 NPU 操作 | 在 `_cpu_simulation/` 中同步添加对应的 Mock |
| NPU 操作接口变更 | 在 Mock 中同步更新，保持接口兼容 |
| 主线版本发布 | 模拟模式同步发布，版本号一致 |
| Bug 修复 | 模拟模式代码随主线一起修复 |

### 2.2 Mock 操作覆盖

#### 2.2.1 核心操作清单

模拟模式需要覆盖的核心 NPU 操作：

| 层级 | 操作类型 | Mock 策略 |
|------|----------|----------|
| **设备层** | `torch.npu.current_device()` | 返回 0 |
| | `torch.npu.get_device_name()` | 返回 "Ascend NPU (Mock)" |
| | `torch.npu.get_device_properties()` | 返回 Mock 属性 |
| | `torch.npu.set_device()` | 空操作 |
| | `torch.npu.synchronize()` | 空操作 |
| **内存层** | `torch.npu.empty_cache()` | 空操作 |
| | `torch.npu.max_memory_allocated()` | 返回 0 |
| **Stream/Event** | `torch.npu.Stream()` | 返回 Mock Stream |
| | `torch.npu.Event()` | 返回 Mock Event |
| | `torch.npu.graph()` | 返回 Mock Graph |
| **通信层** | `all_reduce()` | 返回输入 tensor |
| | `all_to_all()` | 返回输入 tensor |
| | `broadcast()` | 返回输入 tensor |
| **计算层** | ATB `linear()` | 返回零 tensor |
| | ATB `layernorm()` | 返回零 tensor |
| | ATB `softmax()` | 返回零 tensor |
| | ATB `matmul()` | 返回零 tensor |

#### 2.2.2 新增操作同步机制

当主线版本新增 NPU 操作时，按以下流程添加 Mock：

```
1. 主线版本新增 NPU 操作（如 new_npu_op）
       ↓
2. 在对应的 Mock 文件中添加 mock_new_npu_op()
       ↓
3. 单元测试验证 Mock 行为
       ↓
4. 随主线版本一起发布
```

---

## 三、实现内容

### 3.1 环境变量配置

**修改文件**: `vllm_ascend/envs.py`

| 环境变量 | 说明 | 默认值 |
|---------|------|-------|
| `VLLM_ASCEND_ENABLE_CPU_SIMULATION` | 是否启用模拟模式 | 0 (禁用) |
| `VLLM_ASCEND_CPU_SIMULATED_TIME_MS` | 每个 NPU 操作的模拟执行时间(毫秒) | 0 |

### 3.2 核心 Mock 模块

**目录**: `vllm_ascend/_cpu_simulation/`

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块入口，提供 `init_cpu_simulation()` 函数 |
| `mock_torch_npu.py` | Mock `torch_npu` 模块，返回假值 |
| `mock_torch_npu_ops.py` | Mock `torch.npu` 命名空间 |
| `mock_hccl.py` | Mock HCCL 通信库（返回输入） |
| `mock_atb.py` | Mock ATB 操作（返回假值） |
| `time_simulator.py` | 时间模拟器，提供可配置的延迟模拟 |

### 3.3 平台层修改

**修改文件**: `vllm_ascend/platform.py`

- 添加 `is_cpu_simulation_enabled()` 类方法
- 修改相关方法在模拟模式下返回 Mock 结果

### 3.4 工具函数修改

**修改文件**: `vllm_ascend/utils.py`

- 修改 `_init_ascend_device_type()`: 模拟模式使用默认设备类型 (A2)
- 修改 `check_ascend_device_type()`: 模拟模式跳过设备类型检查

### 3.5 Worker 层修改

**修改文件**: `vllm_ascend/worker/worker.py`

- 在 `__init__` 方法中添加模拟模式检测
- 模拟模式跳过 NPU 特定的导入和初始化
- 模拟模式跳过 npugraph_ex 静态内核设置

### 3.6 Patch 系统集成

**新增文件**: `vllm_ascend/patch/platform/patch_cpu_simulation.py`

---

## 四、使用方法

### 4.1 启用模拟模式

```bash
# 设置环境变量
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=5

# 运行 vllm serve
python -m vllm serve Qwen/Qwen2-5-7B-Instruct --host 0.0.0.0 --port 8000
```

### 4.2 在 Python 代码中使用

```python
import os
os.environ["VLLM_ASCEND_ENABLE_CPU_SIMULATION"] = "1"
os.environ["VLLM_ASCEND_CPU_SIMULATED_TIME_MS"] = "10"

# 然后正常导入 vllm
from vllm import LLM
```

---

## 五、架构设计

### 5.1 分层 Mock 架构

```
┌─────────────────────────────────────────────┐
│              vLLM Core                       │
├─────────────────────────────────────────────┤
│         NPUPlatform (Mock)                  │
│  - 设备能力返回 Mock 值                       │
│  - 配置检查跳过                              │
├─────────────────────────────────────────────┤
│         NPUWorker (Mock)                    │
│  - 跳过模型加载                             │
│  - 使用 Mock 操作                           │
├─────────────────────────────────────────────┤
│         Mock NPU Operations                 │
│  - torch.npu.* -> 空操作                    │
│  - torch_npu.* -> 返回假值                  │
│  - ATB ops -> 返回零tensor                  │
├─────────────────────────────────────────────┤
│         Time Simulator                      │
│  - 每次操作增加 CONSUMED_TIME 延迟           │
└─────────────────────────────────────────────┘
```

### 5.2 假值返回策略

| 操作类型 | 返回值 |
|---------|--------|
| `linear` | 零tensor (同输入shape) |
| `layernorm` | 零tensor (同输入shape) |
| `softmax` | 零tensor (同输入shape) |
| `matmul` | 零tensor (正确output shape) |
| `all_reduce` | 返回输入tensor |
| `all_to_all` | 返回输入tensor |
| `synchronize` | 空操作 |

### 5.3 时间模拟机制

延迟时间通过环境变量 `VLLM_ASCEND_CPU_SIMULATED_TIME_MS` 配置：

```bash
# 每个 NPU 操作延迟 5 毫秒
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=5
```

模拟器提供三种使用方式：

1. **自动延迟**: 在所有 Mock 操作中自动添加延迟
2. **装饰器**: `@simulate_time` 装饰器
3. **上下文管理器**: `TimeSimulator` 上下文管理器

---

## 六、版本兼容性

### 6.1 与主线版本的兼容性

| vllm-ascend 版本 | 模拟模式版本 | 兼容性说明 |
|-----------------|-------------|-----------|
| v0.14.0 | v0.14.0 | 初始版本 |
| v0.14.1 | v0.14.1 | Bug 修复 |
| v0.15.0 | v0.15.0 | 新功能同步 |

### 6.2 与 vLLM 版本的兼容性

模拟模式兼容的 vLLM 版本：

| vllm-ascend 版本 | vLLM 版本 |
|-----------------|-----------|
| v0.14.0 | v0.6.x |
| v0.15.0 | v0.7.x |

### 6.3 演进规划

```
版本演进路线图：
├── v0.14.x (初始版本)
│   ├── 基础 Mock 功能
│   ├── 时间模拟
│   └── 核心操作覆盖
│
├── v0.15.x (完善版本)
│   ├── 更多 NPU 操作覆盖
│   ├── 性能模拟优化
│   └── 分布式场景支持
│
└── v1.0.x (成熟版本)
    ├── 完整操作覆盖
    ├── 自动化测试集成
    └── CI/CD 最佳实践
```

---

## 七、验证方法

### 7.1 单元测试

```python
# tests/ut/test_cpu_simulation.py

def test_cpu_simulation_env_vars():
    """测试环境变量读取"""
    os.environ["VLLM_ASCEND_ENABLE_CPU_SIMULATION"] = "1"
    os.environ["VLLM_ASCEND_CPU_SIMULATED_TIME_MS"] = "10"

    from vllm_ascend import envs
    assert envs.VLLM_ASCEND_ENABLE_CPU_SIMULATION == True
    assert envs.VLLM_ASCEND_CPU_SIMULATED_TIME_MS == 10.0

def test_mock_torch_npu():
    """测试 torch_npu mock 功能"""
    from vllm_ascend._cpu_simulation.mock_torch_npu import npu
    assert npu.current_device() == 0
    assert "Mock" in npu.get_device_name()

def test_mock_atb_linear():
    """测试 ATB linear 返回假值"""
    from vllm_ascend._cpu_simulation.mock_atb import atb
    import torch
    input_tensor = torch.randn(2, 10, 512)
    weight = torch.randn(1024, 512)
    output = atb.linear(input_tensor, weight)
    # 输出应该是零tensor
    assert output.shape == (2, 10, 1024)
    assert torch.all(output == 0)
```

### 7.2 在开发机上运行测试用例

在开发机（无需 NPU 硬件）上运行测试时，需要设置以下环境变量：

#### 7.2.1 环境准备

```bash
# 设置环境变量
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=0  # 调试时可设为 0 加快测试速度
```

#### 7.2.2 运行单元测试

```bash
# 进入项目目录
cd /path/to/vllm-ascend

# 运行模拟模式相关的单元测试
python -m pytest tests/ut/test_cpu_simulation.py -v

# 或运行所有单元测试（仅验证模拟模式相关功能）
python -m pytest tests/ut/ -v -k "cpu_simulation" --tb=short
```

#### 7.2.3 运行集成测试

```bash
# 启动模拟模式的推理服务
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=5

# 运行简单的推理测试（使用小模型）
python -m vllm serve Qwen/Qwen2-0.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half

# 或者使用 OpenAI API 进行测试
curl http://localhost:8000/v1/models
```

#### 7.2.4 使用 Python 脚本进行快速验证

```python
# test_simulation.py
import os

# 必须在导入 vllm 之前设置环境变量
os.environ["VLLM_ASCEND_ENABLE_CPU_SIMULATION"] = "1"
os.environ["VLLM_ASCEND_CPU_SIMULATED_TIME_MS"] = "1"

# 导入 vllm（会自动启用模拟模式）
from vllm import LLM

# 验证模拟模式是否生效
from vllm_ascend._cpu_simulation.mock_torch_npu import npu
print(f"Device name: {npu.get_device_name()}")
print(f"Device capability: {npu.get_device_properties()}")

# 验证 ATB 操作返回假值
from vllm_ascend._cpu_simulation.mock_atb import atb
import torch
input_tensor = torch.randn(2, 10, 512)
weight = torch.randn(1024, 512)
output = atb.linear(input_tensor, weight)
print(f"Output shape: {output.shape}, all zeros: {torch.all(output == 0)}")
```

运行脚本：
```bash
python test_simulation.py
```

#### 7.2.5 调试技巧

| 场景 | 调试方法 |
|------|----------|
| 加快测试速度 | 设置 `VLLM_ASCEND_CPU_SIMULATED_TIME_MS=0` |
| 查看模拟模式日志 | 设置 `VLLM_LOG_LEVEL=DEBUG` |
| 验证时间模拟 | 设置 `VLLM_ASCEND_CPU_SIMULATED_TIME_MS=100` 并计时 |
| 验证假值输出 | 在推理后检查输出 tensor 是否全为零 |

### 7.3 集成测试

```bash
# 在没有 NPU 的环境中运行
export VLLM_ASCEND_ENABLE_CPU_SIMULATION=1
export VLLM_ASCEND_CPU_SIMULATED_TIME_MS=5
python -m vllm serve Qwen/Qwen2-5-7B-Instruct --host 0.0.0.0 --port 8000
```

---

## 八、文件变更清单

### 8.1 新增文件

- `vllm_ascend/_cpu_simulation/__init__.py`
- `vllm_ascend/_cpu_simulation/mock_torch_npu.py`
- `vllm_ascend/_cpu_simulation/mock_torch_npu_ops.py`
- `vllm_ascend/_cpu_simulation/mock_hccl.py`
- `vllm_ascend/_cpu_simulation/mock_atb.py`
- `vllm_ascend/_cpu_simulation/time_simulator.py`
- `vllm_ascend/patch/platform/patch_cpu_simulation.py`

### 8.2 修改文件

- `vllm_ascend/envs.py`
- `vllm_ascend/platform.py`
- `vllm_ascend/utils.py`
- `vllm_ascend/worker/worker.py`
- `vllm_ascend/patch/platform/__init__.py`

---

## 九、已知限制

1. **推理结果**: 返回假值，推理结果不正确，仅用于流程验证
2. **性能测试**: 模拟模式不用于真实性能测试
3. **特定模型**: 某些特殊模型加载路径可能需要额外适配

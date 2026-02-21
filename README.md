# SCAIL 本地部署探索路径：从 PyCharm 原生调试到 ComfyUI 流程优化

本仓库记录了在 **RTX 5060 (8GB VRAM)** 硬件环境下，强行部署 Wan2.1 视频生成模型的全过程。这不仅仅是一份教程，更是一次从“原生代码硬刚”到“工具流曲线救国”的完整实验记录。

---

## 🛠️ 第一阶段：原生 SCAIL 源码的“外科手术”

在单卡 8G 显存的 Windows 环境下运行原生源码是极大的挑战。我们通过对底层四个核心文件的“降级”与“脱敏”，成功跑通了推理链路。
### 🐍 依赖环境避坑 (Dependencies)
由于我们屏蔽了 Triton 和 DeepSpeed，你的 `requirements.txt` 需要做出调整：
1. **删除**：`triton` 和 `deepspeed`。
2. **安装核心库**：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
3. **关键库**：确保安装了 `omegaconf`, `einops`, `sat`, `sgm`。

### 1. 自动挂载机制与参数脱敏 (针对 `arguments.py`)
**痛点**：原生代码依赖复杂的命令行参数，且默认开启分布式初始化。
**方案**：
* **默认配置注入**：在 `process_config_to_args` 函数中加入强制逻辑。若检测到 `--base` 参数为空，自动挂载 `wan_pose_14Bsc_xc_cli.yaml`。
* **并行度锁死**：将 `model_parallel_size` 默认锁死为 `1`，防止单卡环境尝试触发分布式初始化导致卡死。
```python
# 修改 arguments.py 中的 process_config_to_args 函数
def process_config_to_args(args):
    # 窗口终极引导：如果运行命令里没给 --base 参数，手动补齐
    if args.base is None:
        args.base = ["configs/sampling/wan_pose_14Bsc_xc_cli.yaml"]
        print(f"窗口提醒：检测到 args.base 为空，已自动挂载：{args.base}")
    
    # 强制锁死模型并行度为 1，适配 Windows 单卡环境
    args.model_parallel_size = 1
 ```
### 2. 暴力绕过 Triton 加速算子 (针对 `triton_rotary.py`)
**痛点**：位置编码（Rotary Embedding）原生依赖 Triton，Windows 环境无法直接运行。
**方案**：
* **装饰器屏蔽**：手动注释掉所有 `@triton.jit` 和 `@triton.autotune` 装饰器。
* **Mock 保护**：通过 `try...except` 引导程序在缺失 Triton 时跳过加速路径，改走标准的 PyTorch 路径。
```python
# 修改 triton_rotary.py 头部逻辑
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None
    print("窗口提醒：未检测到 Triton，将尝试跳过加速路径。")

# 核心手术：必须手动注释掉所有 @triton 装饰器，否则定义函数时即报错
# @triton.autotune(...) 
# @triton.jit
def rotary_kernel(...):
    # 此处逻辑已自动回退至 PyTorch 实现
```

### 3. 分布式引擎的“单卡降级” (针对 `diffusion_video.py`)
**痛点**：`SATVideoDiffusionEngine` 原生会自动初始化模型并行，导致单卡环境陷入初始化死循环。
**方案**：
* **并行逻辑重置**：强行将模型并行度设为 `1`，并修改 `mp_merge_model` 逻辑，确保分布式权重能在单卡环境下正确加载。
* **显存激进清理**：在模型加载的关键节点手动插入 `torch.cuda.empty_cache()`，为 8G 显存挤出运行缓冲区。
```python
# 修改 diffusion_video.py 初始化逻辑
# 强制销毁可能存在的并行环境并重置为单卡模式
destroy_model_parallel()
initialize_model_parallel(1)

# 显存清理介入：在加载权重后立即释放 VRAM 碎片
torch.cuda.empty_cache()
```

### 4. 运行时路径修复 (针对 `sample_video.py`)
**痛点**：Windows 环境下常因路径层级问题出现 `ModuleNotFoundError`。
**方案**：
* **动态系统路径注入**：脚本启动即获取绝对路径并插入 `sys.path`。
* **DeepSpeed 彻底剥离**：显式执行 `del args.deepspeed_config`，避免 Windows 下触发 CUDA 环境冲突。
```python
# 1. 动态系统路径注入
import sys, os
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir) 

# 2. 彻底剥离 DeepSpeed
if hasattr(args, 'deepspeed_config'):
    del args.deepspeed_config # 显式删除，防止引擎调用 Linux 独占的加速库
```
---

## 💾 硬件极限榨取策略 (Hardware Optimization)

除了代码层的修改，硬件层面的“外挂”是 8G 显存能跑通模型的物理前提：

### 1. 虚拟内存 (Swap Space) 终极补偿
* **配置方案**：手动将 Windows 虚拟内存设置为 **60GB - 80GB**。
* **存储要求**：必须放置在 **NVMe SSD** 上。这是配合 `cpu_offload` 使用时，防止加载大型权重（如 T5 编码器）导致系统闪退的关键。


### 2. YAML 配置的“节食”方案
参考本仓库 `configs/wan_pose_14Bsc_xc_cli.yaml`：
* **CPU Offload**：强制开启 `cpu_offload: True`，将内存作为显存的溢出缓冲区。
* **Target 类名修正**：将 T5 编码器类名修正为 `FrozenT5Embedder`，解决加载链路报错。
```yaml
model:
  conditioner_config:
    target: sgm.modules.encoders.modules.FrozenT5Embedder # 修正原生类名解析错误
```
---

## 🎨 第二阶段：ComfyUI 流程优化

在摸清底层逻辑后，我们转向 ComfyUI 进行更高效的工作流调试。

* **避坑要点**：
    * **Manager 假死修复**：针对国内网络环境下 Manager 搜索卡死问题进行代理优化。
    * **文件夹套娃**：修复了插件安装时常见的双层文件夹导致的节点加载失败问题。
* **最终策略**：**本地调试工作流 + 云端（5090/4090）最终产出**。

---

## 📁 核心脚本说明 (Scripts Reference)

为了方便 Windows 用户快速复现，我们将修改后的核心脚本统一存放在 `/scripts` 目录下：

| 文件名 | 核心修改点 | 作用 |
| :--- | :--- | :--- |
| `arguments.py` | 自动挂载 `base` 配置 | 解决命令行参数缺失导致的报错 |
| `triton_rotary.py` | 屏蔽 `@triton` 装饰器 | 绕过 Windows 无法运行 Triton 的限制 |
| `diffusion_video.py` | 强行重置并行度为 1 | 适配单卡 8G 显存环境，防止死循环 |
| `sample_video.py` | 动态路径注入 + GC 清理 | 修复模块找不到问题并防止显存溢出 |

**使用建议**：将上述文件替换掉原生 SCAIL 项目中的对应文件即可直接在 Windows 环境启动。
---

## 🔗 参考与鸣谢 (References & Credits)

本项目是在以下优秀开源项目与教程的启发下完成的，特此感谢：

1. **原生 SCAIL 项目**：
   * [[SCAIL GitHub 仓库链接]](https://github.com/zai-org/SCAIL)
   * *理由：提供了核心的推理框架与 Wan2.1 的原生代码实现。*

2. **@啦啦啦的小黄瓜 (Bilibili)**：
   * **视频教程**：【[ComfyUI]SCAIL震撼来袭！骨骼驱动舞蹈，这个东西“不正常地强”！】https://www.bilibili.com/video/BV1vbqnByEdV?vd_source=625a0575bce7e04151a6443509dfe0be
   * **算力支持**：(https://studio.aigate.cc/images/1028675998928474112?release&channel=C4B7J8G5Z)
   * *理由：UP 主的视频是我进入 Wan2.1 探索的起点，其提供的整合包与云端方案极大缓解了本地 8G 显存的生产压力。*

3. **ComfyUI 社区**：
   * 感谢所有开发 WanVideo 相关节点的贡献者，让低显存用户有了生存空间。
---

## 📂 项目结构
* `configs/`: 存放本阶段修改后的 `wan_pose_14Bsc_xc_cli.yaml` 配置文件。
* `scripts/`: 包含针对 Windows 环境修改的 Python 导入屏蔽脚本。

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

## 🎨 第二阶段：ComfyUI 生产力实战 (基于可视化工作流)

在 PyCharm 环境验证成功后，我们将生产力沉淀到了 ComfyUI 中。针对 **RTX 5060 (8G)** 的硬件限制，本仓库提供的 JSON 工作流实现了资源调度的最大化。

### 1. 工作流核心模块拆解 (Workflow Modules)

本仓库 `workflow/` 目录下的 JSON 文件采用了 **分组解耦** 逻辑：

* **姿态提取区 (Pose Extraction - 蓝色组)**：
    * 使用 `DWPose Estimator` 节点进行预处理。
    * **8G 适配建议**：先禁用其他组，单独运行此组生成骨骼序列（Pose Sequence），避免 Pose 提取与视频生成同时抢占显存。
* **模型加载区 (Models Loading - 紫色组)**：
    * 集成了 `WanVideo VAE`、`T5 Text Encoder` 和 `Transformer` 节点。
    * 通过 `SetNode/GetNode` 逻辑大幅减少连线交叉，提升低显存环境下的调试效率。
* **采样生成区 (Sampling - 核心区)**：
    * 关键节点：`WanVideo Sampler`。
    * 该模块已针对分块加载（Offload）进行优化，确保权重在系统内存与显存间平滑切换。



### 2. 8G 显存极限优化参数 (Optimal Settings)

为了在 8G 环境下稳定跑通工作流，请务必在节点中参考以下设置：

| 参数项 | 推荐值 | 目的 |
| :--- | :--- | :--- |
| **VAE Mode** | **Tiled** (分块) | **核心开关**，防止解码高清视频时显存炸裂 |
| **Sampling Steps** | 20 - 30 | 8G 显存下的效率最优解 |
| **Resolution** | 832x480 / 480x832 | 1.3B 模型在该分辨率下表现最稳定 |
| **Scheduler** | UniPC / DPM++ | 缩短采样时间，减少显存占用时长 |

### 3. 环境补丁与物理外挂 (Hardware Hacks)

* **虚拟内存 (Virtual Memory)**：手动将 Windows 虚拟内存设置为 **60GB - 80GB**。这是 8G 显存用户能点击“Queue Prompt”的物理门槛。
* **节点依赖**：导入 JSON 后，若出现红框，请通过 `Manager -> Install Missing Custom Nodes` 补齐以下插件：
    * `ComfyUI-WanVideo`
    * `ControlNet-Aux` (DWPose 核心)
    * `ComfyUI-VideoHelperSuite`
    * `ComfyUI-Logic` (Set/Get 节点支持)

---

## 🚀 如何从第一阶段过渡到第二阶段？

1.  **路径共享**：在 ComfyUI 的 `extra_model_paths.yaml` 中，将路径指向你第一阶段存放模型权重的目录，避免磁盘空间浪费。
2.  **显存回收**：在 ComfyUI 运行间隙，建议观察任务管理器。如果虚拟内存占用过高，点击 ComfyUI 菜单栏的 **"Free Model and Node Cache"**。
3.  **云端协同**：本地 (8G) 用于工作流逻辑测试；最终的高清长视频产出，建议导出此 JSON 并通过 **AI Gate** 等云端算力平台一键渲染。

#### 🎬 成果展示 (Results Showcase)
我们成功将迈克尔·杰克逊 (Michael Jackson) 的经典舞步迁移到了目标人物照片上。

| **目标原图** | **动作参考视频** | **提取的姿态骨架 (DWPose)** | **最终生成结果 (Wan2.1)** |
| :---: | :---: | :---: | :---: |
| ![Target](assets/a_man.png) | ![Motion](assets/mj_dancing_convert.gif) | ![Pose](assets/body_convert.gif) | ![Final](assets/a_man_convert.gif) |

#### ⚠️ 关键技术细节与显存优化 (Technical Notes & VRAM Optimization)
在使用 **RTX 5090 (32GB 显存)** 进行云端部署实验时，我遇到了以下核心瓶颈及解决方案：

* **显存分配陷阱 (OOM)**:
    * **问题**：在云端同时加载 **32.8GB 的 bf16 完整版主模型** 和 **11.4GB 的 T5 文本编码器** 时，显存会瞬间突破 32GB 极限，导致 `Allocation on device` 报错。
    * **对策**：为了在 32GB 卡上跑通复杂的 DWPose 工作流，建议将 T5 编码器换成 **fp8 版本 (6.73GB)**，并将主模型也降级为 **fp8 (14.4GB)** 以腾出计算空间。

* **姿态检测节点的干扰**:
    * **发现**：`PoseDetectionVitPoseToDWPose` 节点在初始化 ONNX 运行时会占用大量显存。
    * **建议**：务必将所有加载器（Loader）的 `load_device` 设置为 **`offload_device`**，确保姿态检测完成后能及时为采样器腾出显存空间。

* **VAE 文件损坏修复**:
    * **警告**：如果遇到 `header too large` 错误，说明 VAE 文件不完整或损坏。
    * **修复**：重新下载 **508MB** 的 `Wan2_1_VAE_fp32.safetensors` 是最稳妥的解决方案。

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

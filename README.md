# ComfyUI-MiniMax-FusedSwiGLU

专为 **MiniMax H3**（海螺音视频单流 DiT 模型）设计的 **超低显存流式 Fused SwiGLU 算子优化节点**。

---

## 🌟 核心特性

1. **端到端微瓦片流式引擎（End-to-End Micro-Tile Stream）**：
   - **`auto_stream (anti_oom)`（推荐默认）**：为 INT8 量化及 BF16/FP16 模型启用端到端微瓦片流水线。将 `Norm2 → AdaLN → FC1 → SwiGLU → FC2 → Gated Residual Add` 整条链路按 token 切块处理，将激活显存峰值**严格锁定在 ~70MB**（tile=1024）/ **~35MB**（tile=512），彻底解决长序列（$S=20000$）下原版 1.5GB+ 中间激活峰值引发的 OOM 问题。
   - **循环外参数预取（Pre-fetching）**：在进入 tile 循环前一次性提取并转换该段的 AdaLN 调制参数，消除高频索引和数据类型转换开销。
   - **`token_wise (lowest_vram)`**：显式指定端到端微瓦片流式处理。
   - **`channel_wise (legacy)`**：通道维切分模式（保留向后兼容，显存需求高于微瓦片流）。

2. **2D SRAM 融合 Triton 算子与真·Bit-Exact 保证**：
   - **2D 程序分块与列快变调度**：采用 `BLOCK_M x BLOCK_N` 二维分块，并在 1D Grid 中以列为最快变化轴，保证相邻 Program 访问连续物理内存。实测在 Ada Lovelace 架构上有效访存带宽达 **440+ GB/s（较 1D 分块提速 1.62x）**。
   - **严格 Bit-exact 双舍入对齐**：在寄存器内部精准模拟 PyTorch Eager 的 `R(R(silu(g)) * u)` 两次 bf16 舍入截断，实测 `maxdiff == 0.000e+00`，彻底消除误差积累。
   - **Zero-Allocation In-place SwiGLU**：在打包权重输出上直接覆写 Gate 区域，利用 cuBLAS 的 `lda > K` 零拷贝特性衔接后续矩阵乘，单瓦片省去 28MB 中间显存分配且完全无读写数据竞争。
   - **融合 AdaLN Modulate 算子（`triton_modulate_`）**：将 `mul_ + add_` 融合成单次核函数发射，在保证 Bit-exact 前提下使调制阶段提速 **1.59x**。
   - **硬件规划缓存（Plan Cache）**：自适应硬件 L2/VRAM 分块规划并全面使用 `lru_cache` 缓存，消除推理时的 Python 重复规划与 CUDA 属性查询。

3. **双向兼容与原生 INT8 加速**：
   - **INT8 残差直通与特性探测**：自动探测底层 `comfy_kitchen` 的 `ck.int8_linear`，支持时自动折叠门控残差累加至 INT8 GEMM Epilogue（省去一次 HBM 读写），不支持时平滑降级。
   - **模块拓扑与环境自检**：严格校验 MiniMax H3 权重通道比例（$\text{fc1} = 2 \times \text{fc2}$），遇到异常结构或非对应模型时安全回退，绝不静默破坏输出。
   - **全平台优雅降级**：无 Triton 环境（如部分 Windows 部署）自动平滑回退为 PyTorch 原生切片实现，无需额外配置。

---

## 🚀 使用方法

在 ComfyUI 工作流中：
1. 搜索并添加节点：`MiniMax H3 Fused Stream SwiGLU (Anti-OOM)`（分类：`model_patches/minimax`）；
2. 将加载的 MiniMax 模型（`MODEL`）连接到此节点的 `model` 输入端；
3. 将输出的 `model` 直接送入下游采样器（如 KSampler / SamplerCustomAdvanced）或其它补丁节点。

### 参数配置说明：

- **`strategy`**（默认 `auto_stream (anti_oom)`）：
  - `auto_stream (anti_oom)`：**推荐首选**。自动为 INT8 量化与浮点模型分配端到端微瓦片流，锁定单瓦片激活峰值，防止爆显存；
  - `token_wise (lowest_vram)`：显式开启微瓦片流水线；
  - `channel_wise (legacy)`：通道切分模式（保留向后兼容）。
- **`tile_size`**（默认 1024）：
  - `0`：自动自适应（$\ge 14\text{GB}$ 显存设为 1024；$10\text{GB}$ 显存设为 512；$\le 8\text{GB}$ 显存设为 256）；
  - `1024`：16G/24G 显卡（RTX 4080/4090/5080 等）推荐；
  - `512` / `256`：12G / 8G 极限低显存用户推荐。
- **`use_triton`**（默认 True）：
  - 开启片上 Triton SRAM 融合算子加速（SwiGLU / Addcmul / AdaLN modulate）；无 Triton 环境自动回退 PyTorch。

### 自动化验证与测试：

在节点目录下可直接运行全套自动化测试套件：
```bash
python test_suite.py
```

---

## 📊 实测性能指标（RTX 4080 16GB, S=20000 真实尺寸）

| 测试项 | 官方 Eager | 原版 v1 | 本次优化版本 | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| **单层峰值显存** | 3091 MB (OOM点) | 1590 MB | **1049 MB** | 彻底消除 1.5GB+ 中间显存峰值 |
| **单层耗时 (e2e)** | 140.8 ms | 149.8 ms | **145.7 ms** | 流式方案中最优，无 GEMM 碎片开销 |
| **SwiGLU 算子带宽** | 267 GB/s | 267 GB/s | **443 GB/s (1.62x)** | 2D 程序分块与连续访存命中 |
| **数值一致性** | 基线 | 存在细微截断差 | **maxdiff = 0.000e+00** | 严格逐位完全一致 (Bit-exact) |

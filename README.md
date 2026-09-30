# ComfyUI-MiniMax-FusedSwiGLU

专为 **MiniMax H3**（海螺音视频单流 DiT 模型）设计的 **超低显存流式 Fused SwiGLU 算子优化节点**。

---

## 🌟 核心特性

1. **双通道自适应流式引擎（Auto Stream Strategy）**：
   - **`channel_wise (fastest)`（BF16/FP16 推荐首选）**：沿 FFN 通道维（$K$ 维）分段计算，**全流程仅读取一次庞大的模型权重**（避免了 token 分块对 880MB 权重的反复加载），同时采用 **FP32 高精度累加缓冲区**，彻底解决 BF16 段间累加导致的噪点与色彩漂移，输出与官方未切片算子高度一致。
   - **`token_wise (lowest_vram)`（INT8 量化与极限低显存首选）**：将 Norm2 + AdaLN + FlashMLP + In-place 残差累加打包为微瓦片流，将单次激活内存**物理锁死在 ~29MB～58MB**，彻底根治长序列 $S=20000$ 时的 1.15GB+ 激活显存尖峰。

2. **Triton 原生片上 SRAM 双重融合算子**：
   - **Triton Split SwiGLU**（`triton_swiglu_split`）：在 GPU 片上寄存器与共享内存（SRAM）中直接接收分离的 Gate 与 Up 张量，一次性完成 $\text{silu}(gate) \times up$，相比 PyTorch 原生操作**提速 2 倍以上**，且零中间 HBM 显存开销。
   - **Triton Fast Addcmul**（`triton_addcmul_`）：专为 DiT 定制的 2D 就地加权累加算子（`x += res * g`），全面支持 128 位向量化访存，绕开 PyTorch 庞大的 `TensorIterator` CPU 启动开销。

3. **双向无缝兼容与安全性**：
   - **原生 INT8 量化直通**：自动识别 ComfyUI `ck.int8_linear` 动态量化算子（Row-wise 逐 Token 独立量化），分块与全量计算输出完全一致。
   - **静态模型拓扑自检**：打补丁前严格校验 MiniMax H3 权重通道比例与结构，遇异常环境自动安全回退官方实现，绝不静默崩坏。
   - **无 Triton 环境优雅降级**：若未安装 Triton 或平台不支持，自动平滑回退为 PyTorch 高精度原生切片实现。

---

## 🚀 使用方法

在 ComfyUI 工作流中：
1. 搜索并添加节点：`MiniMax H3 Fused Stream SwiGLU (Anti-OOM)`（分类：`model_patches/minimax`）；
2. 将加载的 MiniMax 模型（`MODEL`）连接到此节点的 `model` 输入端；
3. 将输出的 `model` 直接送入下游的采样器（KSampler / SamplerCustomAdvanced）或其它补丁节点。

### 参数配置说明：

- **`strategy`**（默认 `auto_stream (anti_oom)`）：
  - `auto_stream`：自动识别 INT8 与 BF16/FP16 模型并分配最优链路；
  - `channel_wise (fastest)`：浮点模型推荐，显存搬运量最低、推理速度最快；
  - `token_wise (lowest_vram)`：8GB 等极限显存卡推荐，激活内存极低。
- **`tile_size`**（默认 1024，仅 token_wise 生效）：
  - `0`：自动自适应（>=16G 显存设为 1024；12G 显存设为 512；<=8G 显存设为 256）；
  - `1024`：16G/24G 显卡（RTX 4080/4090/5080 等）推荐；
  - `512` / `256`：12G / 8G 极限低显存用户推荐。
- **`channel_chunks`**（默认 8，仅 channel_wise 生效）：
  - 通道切分段数（2~16，默认 8）。各分段自动在 FP32 缓冲区中无损累加。
- **`use_triton`**（默认 True）：
  - 开启片上 Triton SRAM 融合算子加速。

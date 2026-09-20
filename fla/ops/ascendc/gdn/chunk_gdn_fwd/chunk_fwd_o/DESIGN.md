# ChunkFwdO 算子设计文档

> 目标平台：**Ascend 950（A5，`__CCE_AICORE__ == 310`，`CATLASS_ARCH 3510`）**
> 核型：`KERNEL_TYPE_MIX_AIC_1_2`（1 AIC : 2 AIV，即每个 AIC 天然对应 2 个 AIV subBlock）
> 适用范围：GDN（Gated Delta Network）前向链路中 `o` 输出计算

> **AIC/AIV 对应原则**：在 Ascend 950 的 `MIX_AIC_1_2` 核型下，**1 个 AIC 与同编号的 2 个 AIV subBlock 构成一个不可分割的执行单元**（记为 AIC_i + AIV_{i,0} + AIV_{i,1}）。二者共享同一份任务上下文（`taskIdx`、`offsets`、`stage`），通过 `subBlockIdx` 协同处理同一 stage 的不同数据行，而不是"1 个 AIC 把任务分给 2 个独立 AIV"。Vec1 与 Vec2 是**每个 AIV subBlock 内部串行执行的两段向量后处理**，不是分别落到两个不同 AIV 上。

---

## 一、算子功能概述

### 1.1 算子定位

`ChunkFwdO` 是 GDN Chunk 前向算子族中负责输出 `o` 计算的子算子，承接上游通过 WY 分解得到的隐藏状态 `h`，完成当前 chunk 注意力输出 `o` 的合成。

入口注册在 [op_kernel/chunk_fwd_o.cpp](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/chunk_fwd_o.cpp#L60)：

```cpp
extern "C" __global__ __aicore__ void chunk_fwd_o(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    ...
    GDN::ChunkFwdODispatch(...);
}
```

`KERNEL_TYPE_MIX_AIC_1_2` 在 Ascend 950 上启动时，**物理核以"1 AIC + 2 AIV subBlock"为基本块（block）下发**，每个 block 内 AIC 编号与 AIV 编号天然绑定：AIC_i 对应 AIV_{i,0}、AIV_{i,1}。`GDNFwdOKernel::Process` 通过 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 在同一份 kernel 代码内分别走 AIC 与 AIV 两个分支，二者并行执行。

### 1.2 应用场景

- 线性注意力 + 门控 delta 规则模型的前向推理/训练前向
- 长序列建模：序列按 `chunk_size`（64 或 128）切分，块内并行 + 块间递推
- GQA 场景：`HV % HK == 0`，K 头与 V 头解耦
- 既支持定长 `[B, HV, T, V]`，也支持变长（`cu_seqlens` + `chunk_offsets` 拼接，仅 `B=1`）

### 1.3 数学原理

对每个 chunk `i`、每个 V 头 `h`，输出 `o`：

```
o_i = scale · ( intra_chunk_attn + inter_chunk_state )
```

- **块内贡献（intra-chunk）**：
  ```
  attn_i = lower_triangular( (q_i @ k_i^T) · exp(g_i[:,None] - g_i[None,:]) )
  ```
- **块间贡献（inter-chunk）**：
  ```
  o_inter_i = (q_i · exp(g_i)) @ h_i
  ```
- **最终输出**：
  ```
  o_i = scale · (o_inter_i + attn_i @ v_i)
  ```

参考实现见 [tests/pta/test_fwd_o.py](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/tests/pta/test_fwd_o.py#L95-L102)。

---

## 二、公式推导与表达

### 2.1 输入输出关系

| 名称 | 形状 | dtype | 含义 |
|---|---|---|---|
| `q` | `[B, HK, T, K]` | bf16/fp16 | Query |
| `k` | `[B, HK, T, K]` | bf16/fp16 | Key |
| `v` | `[B, HV, T, V]` | bf16/fp16 | Value |
| `h` | `[B, HV, numChunks, K, V]` | bf16/fp16 | 隐藏状态 |
| `g` | `[B, HV, T]` | bf16/fp16/fp32 | 门控（cumsum 后） |
| `cu_seqlens` | 1D | int64 | 变长序列累计长度 |
| `chunk_offsets` | 1D | int64 | `[tb_id, chunk_id]` 扁平对 |
| `o` | `[B, HV, T, V]` | bf16/fp16 | 输出 |

约束：`K = 128`，`V ∈ {128, 256}`，`chunkSize ∈ {64, 128}`，`HV % HK == 0`。

### 2.2 完整计算公式

对每个 `(b, hv, chunk_i, v_block)`，设 `t` 为该 chunk 内 token 索引：

**Step 1: 块内注意力分数**
```
S_i ∈ R^{C×C},  S_i[t,j] = q_i[t] · k_i[j]^T
```

**Step 2: 门控衰减掩码**
```
M_i[t,j] = exp(g_i[t] - g_i[j])        // 仅 j ≤ t 有意义
```

**Step 3: 因果+门控注意力**
```
A_i = lower_triangular(S_i ⊙ M_i)
```

**Step 4: 块间状态贡献**
```
O_inter_i[t] = q_i[t] · exp(g_i[t]) · h_i[:, :]
```

**Step 5: 最终输出**
```
O_i[t] = scale · ( O_inter_i[t] + Σ_{j≤t} A_i[t,j] · v_i[j] )
```

### 2.3 关键参数说明

| 参数 | 含义 |
|---|---|
| `scale` | 通常 `1/sqrt(K)`，最终乘到输出 |
| `chunkSize` | 分块大小，与 `h` 维度对齐 |
| `headGroups = HV / HK` | GQA 组数，K 头被 `headGroups` 个 V 头共享 |
| `PING_PONG_STAGES = 2` | 双缓冲流水级数 |
| `isFinalState` | 标记 chunk 是否为最后一块（影响 `blockTokens` 计算） |

---

## 三、整体架构设计

### 3.1 模块划分

```
op_host/        ← Tiling、ShapeCheck、aclnn 接口
op_kernel/      ← Kernel 主体与子模块
├── chunk_fwd_o.cpp           ← 入口，MIX_AIC_1_2 核型
├── chunk_fwd_o_struct.h      ← TilingData 结构
├── gemm/
│   ├── kernel/gdn_fwd_o_kernel.hpp   ← GDNFwdOKernel 主体
│   └── block/block_scheduler_gdn_fwd_o.hpp ← 分核调度器
└── epilogue/
    ├── gdn_fwd_o_epilogue_policies.hpp       ← 策略类（指定 Ascend950）
    └── block/
        ├── block_epilogue_gdn_fwdo_qkmask.hpp  ← Vec1：QK 掩码
        └── block_epilogue_gdn_fwdo_output.hpp   ← Vec2：输出合成
```

### 3.2 各模块功能

1. **Host Tiling 模块**（[chunk_fwd_o_tiling_processor.h](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_host/chunk_fwd_o_tiling_processor.h)）：
   - `PreCheck`/`ShapeCheck`：维度与 head 配比校验
   - `CommonTiling`：写入 `seqlen/headDim/chunkSize` 等基础参数，处理变长 `tokenBatch`
   - `WorkspaceTiling`：在 GM 上分配 5 块 ping-pong workspace（`v/h/attn/aftermask/mask`）

2. **Kernel 主体**（[gdn_fwd_o_kernel.hpp](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp)）：
   - `GDNFwdOKernel::Init`：绑定 GM 缓冲；在 AIC 分支初始化 `cubeBlockScheduler`，在 AIV 分支初始化 `vecBlockScheduler`，二者基于同一份 `tilingData` 派生
   - `GDNFwdOKernel::Process`：在 AIC 与 AIV 两侧分别进入 `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` 分支并行执行；AIC 与其绑定的 2 个 AIV subBlock 共享同一任务上下文

3. **调度器**（[block_scheduler_gdn_fwd_o.hpp](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/block/block_scheduler_gdn_fwd_o.hpp)）：
   - `BlockSchedulerGdnFwdO`：基类，负责 `curTaskIdx → (b, chunk, vHead)` 映射、ping-pong stage 推进
   - `BlockSchedulerGdnFwdOCube`：Cube 侧接口（`GetCube1Offsets` / `GetCube23Offsets`）
   - `BlockSchedulerGdnFwdOVec`：Vec 侧接口（`GetVec1Offsets` / `GetVec2Offsets`）

4. **Epilogue 模块**：每个 AIV subBlock 内串行执行的两段向量后处理（Vec1 → Vec2）
   - `EpilogueGDNFwdOQkmask`（Vec1）：对 `attn` 应用门控衰减掩码 + 因果掩码 → `aftermask`
   - `EpilogueGDNFwdOOutput`（Vec2）：合成 `o_inter + attn @ v` 并乘 `scale`，回写 `o`
   - 2 个 AIV subBlock 通过 `subBlockIdx` 切分上述两段的行（`mActualPerSubBlock = CeilDiv(mActual, subBlockNum)`），共同完成 AIC 输出 tile 的后处理

### 3.3 数据流图

下图按"1 AIC 天然对应 2 AIV subBlock"原则绘制。每个 block 内 AIC_i 与 AIV_{i,0}、AIV_{i,1} 共享 `taskIdx / stage / offsets` 上下文，AIV 侧两个 subBlock 通过 `subBlockIdx` 在 Vec1/Vec2 内部各自处理不同的行。

```
GM Inputs: q, k, v, h, g
     │
     ▼
╔══════════════════════════════════════════════════════════════════╗
║                    Block i (1 AIC + 2 AIV subBlock)              ║
║                                                                   ║
║  ┌────────────────────────── AIC_i ──────────────────────────┐   ║
║  │                                                            │   ║
║  │  Cube1: Q @ K^T ──► attn_workspace[stage] (fp32)         │   ║
║  │                ──set cube1Done[stage]──►                  │   ║
║  │                                                            │   ║
║  │  Cube2: Q·exp(g) @ H ──► h_workspace[stage] (fp32)       │   ║
║  │  Cube3: aftermask @ V ──► v_workspace[stage] (fp32)     │   ║
║  │                ──set cube3Done[stage]──►                  │   ║
║  └────────────────────────────────────────────────────────────┘   ║
║           ▲                              ▲                        ║
║           │ cube1Done                     │ cube3Done              ║
║           ▼                              ▼                        ║
║  ┌────── AIV_{i,0} ──────┐    ┌────── AIV_{i,1} ──────┐         ║
║  │ Vec1: 行 [0..m/2)      │    │ Vec1: 行 [m/2..m)      │         ║
║  │  attn⊙exp(Δg)⊙tril     │    │  attn⊙exp(Δg)⊙tril     │         ║
║  │  ──► aftermask          │    │  ──► aftermask          │         ║
║  │  ──set vec1Done──►      │    │  ──set vec1Done──►      │         ║
║  │                          │    │                          │         ║
║  │ Vec2: 行 [0..m/2)      │    │ Vec2: 行 [m/2..m)      │         ║
║  │  o = scale·(v+h·exp(g)) │    │  o = scale·(v+h·exp(g)) │         ║
║  │  ──► GM o               │    │  ──► GM o               │         ║
║  │  ──set vec2Done──►      │    │  ──set vec2Done──►      │         ║
║  └────────────────────────┘    └────────────────────────┘         ║
╚══════════════════════════════════════════════════════════════════╝
                            │
                            ▼
                       GM Output: o
```

要点：
- AIC_i 与 AIV_{i,0/1} **同生命周期绑定**，二者必须看到相同的 `stage` 与 `offsets`
- Vec1 与 Vec2 是**每个 AIV subBlock 内部串行执行的两段流程**，不是"AIV_0 做 Vec1、AIV_1 做 Vec2"
- 2 个 AIV subBlock 通过 `subBlockIdx` 切分 mmad 输出 tile 的行（见 §6.1 §6.2 的 `mActualPerSubBlock`），共同完成 AIC 输出的后处理
- AIC 与 AIV 之间的同步通过 4 组 `CrossCoreFlag` 完成（见 §7）

---

## 四、分核逻辑设计

### 4.1 任务划分

任务粒度定义为 `(batch, chunk, vHead)`，总任务数：

```cpp
taskNum = shapeBatch * numChunks * vNumHead;
```

每个任务对应一个 `chunkSize × vHeadDim` 的输出 tile，不切分 K/V 维度（直接整块计算），降低调度开销。

### 4.2 Cube/Vec 核分配与 1:2 绑定关系

Ascend 950 上 `KERNEL_TYPE_MIX_AIC_1_2` 表示 **1 AIC 天然对应 2 个 AIV subBlock**。一个 block 由 1 个 AIC 和 2 个 AIV subBlock 组成：

- **AIC 侧**：`AscendC::GetBlockIdx()` 取 block（cube）编号 `i`，承担 Cube1/Cube2/Cube3 三段 mmad
- **AIV 侧**：2 个 AIV subBlock 与同编号的 AIC 绑定，`subBlockIdx ∈ {0, 1}`；`cubeCoreIdx = GetBlockIdx() / GetSubBlockNum()` 还原出所属 block 编号
- **任务上下文共享**：同一 block 内的 AIC 与 2 个 AIV subBlock **使用完全相同的 `taskIdx / offsets / stage` 序列**，仅在数据行维度通过 `subBlockIdx` 切分

```cpp
// Cube 侧：直接用 block 编号作为 cube 核号
// block_scheduler_gdn_fwd_o.hpp:219
BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tilingData,
                            AscendC::GetBlockIdx(),        // cubeCoreIdx = blockIdx
                            AscendC::GetBlockNum());

// Vec 侧：通过 / GetSubBlockNum() 还原所属 block 编号
// block_scheduler_gdn_fwd_o.hpp:273-274
BlockSchedulerGdnFwdO::Init(cu_seqlens, chunk_offsets, tilingData,
                            AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(),  // 还原 cubeCoreIdx
                            AscendC::GetBlockNum());
```

由此得到 **block i 内的执行单元**：

```
Block i:  AIC_i  ⇄  AIV_{i, subBlockIdx=0}  +  AIV_{i, subBlockIdx=1}
                 (共享 taskIdx / offsets / stage)
```

每个 block 处理的 task 范围由 `cubeCoreIdx * PING_PONG_STAGES` 起步，block 内的 2 个 AIV subBlock 通过 `mActualPerSubBlock = CeilDiv(mActual, subBlockNum)` 把 mmad 输出的行二分。

### 4.3 任务分配策略

采用 **stride 分配 + ping-pong 双缓冲**：

```cpp
taskIdx = cubeCoreIdx * PING_PONG_STAGES;       // 起始任务
...
if (processNewTask) {
    taskIdx += PING_PONG_STAGES * cubeCoreNum;  // 跨核步进
}
```

- 每个 cube 核每轮领取 `PING_PONG_STAGES = 2` 个任务，对应双缓冲 stage0/stage1
- 处理完一轮后跨 `cubeCoreNum * 2` 步进到下一轮
- 这种分配天然支持变长（不同 chunk 的 `blockTokens` 不同）时仍保持负载近似均衡：每个核处理的任务数 ≈ `taskNum / cubeCoreNum`

### 4.4 变长模式任务索引

变长模式下 `chunkIdx` 不再等价于 `batchChunkIdx`：

```cpp
tokenBatchIdx = gmChunkOffsets.GetValue(2 * chunkIdx);     // 序列 id
batchChunkIdx = gmChunkOffsets.GetValue(2 * chunkIdx + 1); // 序列内 chunk id
tokenOffset    = gmSeqlen.GetValue(tokenBatchIdx);
batchTokens   = gmSeqlen.GetValue(tokenBatchIdx + 1) - tokenOffset;
blockTokens   = isFinalState ? (batchTokens - batchChunkIdx * chunkSize) : chunkSize;
```

`isFinalState` 判定：

```cpp
chunkIdx == (numChunks - 1) ||
(isVariedLen && gmChunkOffsets.GetValue(2 * chunkIdx + 3) == 0)
```

即每个变长子序列的最后一块都走尾块处理逻辑，`blockTokens < chunkSize`。

---

## 五、AIC 核作业设计

AIC 核负责 **3 次 Cube 矩阵乘**（见 [gdn_fwd_o_kernel.hpp:247-352](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp#L247)）：

### 5.1 Cube1：Q @ K^T → Attn

```cpp
GemmCoord cube1Shape{blockTokens, blockTokens, kHeadDim};
blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
```

- 输入：`Q [C, K]` (RowMajor), `K^T [K, C]` (ColumnMajor)
- 输出：`Attn [C, C]` (fp32, 写入 `gmAttnWorkspace`)
- Tile 形状：`L1=128×128×128, L0=128×128×128`
- 完成后 `CrossCoreSetFlag<0x2, PIPE_FIX>(cube1Done[streamId])`，通知 Vec1 可以做掩码

### 5.2 Cube2：Q · exp(g) @ H → O_inter

```cpp
GemmCoord cube2Shape{blockTokens, vBlockDim, kHeadDim};
// vBlockDim ≤ 128 → BlockMmadQH128
// vBlockDim > 128 → BlockMmadQH256 (L1=128×256×128, L0=128×256×64)
blockMmadQH(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
```

- 等待 Vec1 完成（`vec1Done`）才能消费 `aftermask` workspace
- 等待 Vec2 完成（`vec2Done`）才能消费 `h/v` workspace（前一轮）
- 输出：`O_inter [C, V]` (fp32, 写入 `gmHWorkspace`)

### 5.3 Cube3：Attn_masked @ V → O_intra

```cpp
GemmCoord cube3Shape{blockTokens, vBlockDim, blockTokens};
blockMmadAttenVNEW(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
```

- 输入：`AttnMasked [C, C]`（Vec1 处理后的 `gmAftermaskWorkspace`），`V [C, V]`
- 输出：`O_intra [C, V]` (fp32, 写入 `gmVWorkspace`)
- 完成后 `CrossCoreSetFlag<0x2, PIPE_FIX>(cube3Done[streamId])`，通知 Vec2 可以合成输出

### 5.4 AIC 优化策略

- **Ping-Pong 双缓冲**：`streamId = GetCurStageId() / GetPrevStageId()`，stage0 计算 + stage1 搬运重叠
- **Cube1 独立 slot**：注释 `vec2Done protects the H/V workspace consumed by Cube2/3; Cube1 uses a separate slot`，Cube1 输出的 `attn` 与 Cube2/3 输入解耦，避免无谓等待
- **PipeBarrier**：`PIPE_MTE2` / `PIPE_FIX` 隔离不同流水线，确保 L1/L0B 数据可见性
- **Tile 选择**：根据 `vBlockDim` 自动选择 `BlockMmadQH128` 或 `BlockMmadQH256`，针对 V=128 和 V=256 优化 L0 形状

---

## 六、AIV 核作业设计

每个 AIV subBlock 与其绑定的 AIC 共享 `taskIdx / stage`，在 subBlock 内部**串行执行 Vec1 与 Vec2 两段向量后处理**（见 [gdn_fwd_o_kernel.hpp:354-413](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp#L354)）。同一 block 内的 2 个 AIV subBlock 通过 `subBlockIdx` 在 Vec1/Vec2 内部各自处理 mmad 输出的不同行。

```
每个 AIV subBlock 的时间线：
  ┌─ Vec1 (当前 stage) ──┬── Vec2 (前一 stage) ──┐
  └──────────────────────┴────────────────────────┘
        ↑                       ↑
   wait cube1Done[s]       wait cube3Done[s_prev]
   set  vec1Done[s]        set  vec2Done[s_prev]
```

### 6.1 Vec1：QK 掩码生成 `aftermask`

```cpp
EpilogueGDNFwdOQkmask epilogue(...);
epilogue(gmAftermaskWorkspace, gmG, gmAttnWorkspace, gmMask,
         chunkSize, blockTokens, kHeadDim, vHeadDim, pingpongFlag, ...);
```

对应 [block_epilogue_gdn_fwdo_qkmask.hpp](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp)，核心流程：

每个 AIV subBlock 通过 `mActualPerSubBlock = CeilDiv(mActual, subBlockNum)` 取得自己负责的行段，subBlock 0 处理 `[0, mActualPerSubBlock)`，subBlock 1 处理 `[mActualPerSubBlock, mActual)`：

1. **门控广播**：`g` 从 GM 搬入 UB → fp32 cast → BRC 广播为 `[C, C]` 矩阵
2. **衰减掩码计算**：
   ```
   mask = exp(min(0, g_leftcast - g_upcast)) ⊙ tril_mask
   ```
   - `Sub(gbrcUp, gbrcLeftcast, gbrcUp)`：`g_left - g_up`
   - `Mins(..., 0.0)`：截断到 ≤ 0
   - `Exp(...)`：得到 `exp(g_left - g_up)`（≤ 1）
   - 与 `maskUbTensor`（下三角）按行乘
3. **掩码应用**：`out = attn * mask`，cast 回 fp16/bf16，写回 `gmAftermaskWorkspace`

### 6.2 Vec2：输出合成

```cpp
EpilogueGDNFwdOOutput epilogue(...);
epilogue(gmO, gmG, gmVWorkspace, gmHWorkspace, scale, ...);
```

对应 [block_epilogue_gdn_fwdo_output.hpp](file:///d:/project/github/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/epilogue/block/block_epilogue_gdn_fwdo_output.hpp)，核心流程：

1. **门控广播**：`exp(g)` 复制为 `[C, V]` 列广播
2. **O_inter 加权**：`gbrcUp = h * exp(g_leftcast)`（`h` 来自 `gmHWorkspace`，即 Cube2 输出）
3. **O_intra 相加**：`gbrcUp = aUb + gbrcUp`（`a` 来自 `gmVWorkspace`，即 Cube3 输出）
4. **scale + cast 回写**：
   ```
   out = scale * (aUb + h * exp(g))
   ```
   cast 到 fp16/bf16，DataCopy 回 GM 的 `o` 区域

### 6.3 AIV 与 AIC 协作机制

由于 1 AIC 天然对应 2 个 AIV subBlock，flag 同步要保证：AIC 的某个 stage 输出必须等到其绑定的 2 个 AIV subBlock 全部消费完才能被覆盖；反之亦然。

- **Vec1 等 `cube1Done`**（AIC_i 的 Cube1[stage] 完成）→ 2 个 AIV subBlock 各自处理自己负责的行 → 各自 set `vec1Done[stage]`（broadcast 模式 `0x2`，AIC 看到即可消费）
- **Cube2/Cube3 等 `vec1Done`**（2 个 AIV subBlock 完成 Vec1）→ AIC_i 可消费 `aftermask`
- **Vec2 等 `cube3Done`**（AIC_i 的 Cube3[stage] 完成）→ 2 个 AIV subBlock 各自处理自己负责的行 → 各自 set `vec2Done[stage]`
- **Cube2/Cube3 等 `vec2Done`**（2 个 AIV subBlock 完成 Vec2）→ AIC_i 下一轮可复用 `h/v` workspace

完整同步链路见 §7.2。

### 6.4 AIV 优化策略

- **UB Ping-Pong**：所有 UB tensor 都有 Ping/Pong 两份，与 AIC 的 stage 同步推进
- **块大小自适应**：
  - `mActualThisSubBlock ≤ 32`：单次处理
  - `> 32`：拆分为 2 个 stage，避免 UB 溢出
- **V > 128 走宽路径**：`ProcessWideOutput` 按 16 行 tile 循环，支持 V=256 时 UB 不溢出
- **8 元素对齐**：`gbrcRealStart = gbrcStart & ~7`，因为 BRC 指令按 8 行对齐工作
- **掩码预生成**：AIV 启动时一次性生成 `maskUbTensor`（64×64 下三角），后续 chunk 复用

---

## 七、同步机制设计

### 7.1 跨核同步原语

Ascend 950 上 AIC/AIV 通过 `Arch::CrossCoreFlag` + `CrossCoreSetFlag` / `CrossCoreWaitFlag` 同步：

```cpp
// block_scheduler_gdn_fwd_o.hpp:98-101
Arch::CrossCoreFlag cube1Done[PING_PONG_STAGES] = {0, 1};
Arch::CrossCoreFlag vec1Done[PING_PONG_STAGES] = {2, 3};
Arch::CrossCoreFlag cube3Done[PING_PONG_STAGES] = {4, 5};
Arch::CrossCoreFlag vec2Done[PING_PONG_STAGES] = {6, 7};
```

8 个 flag ID 分别对应 2 个 stage × 4 个同步点，避免不同 stage 间的 flag 复用冲突。

### 7.2 同步点设置

完整流水一轮（stage `s`，在 block i 内）的同步链：

```
AIC_i: Cube1[s] ──set cube1Done[s]──►
                                     ▼
            ┌────────────────────────────────────────────┐
            │ 2 个 AIV subBlock (绑定的 AIV_{i,0/1})     │
            │   各自 Vec1[s]（按 subBlockIdx 切分行）    │
            │   各自 set vec1Done[s] (broadcast 0x2)    │
            └────────────────────────────────────────────┘
                                     ▼
AIC_i: Cube2[s] (wait vec1Done[s], vec2Done[s_prev])
       Cube3[s] (依赖 Cube2 输入)
       └──set cube3Done[s]──►
                                     ▼
            ┌────────────────────────────────────────────┐
            │ 2 个 AIV subBlock                          │
            │   各自 Vec2[s]（按 subBlockIdx 切分行）    │
            │   各自 set vec2Done[s] (broadcast 0x2)    │
            └────────────────────────────────────────────┘
                                     ▼
AIC_i: Cube2[s_next] 可复用 h/v workspace
```

注意：`0x2` broadcast 模式让 AIC 侧 `WaitFlag` 一次即可看到 2 个 AIV subBlock 的 set；而 2 个 AIV subBlock 都会执行 set（因为每个 subBlock 都跑完整 Vec1+Vec2 流程）。

关键设计：

```cpp
// gdn_fwd_o_kernel.hpp:294-296
Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);
// vec2Done protects the H/V workspace consumed by Cube2/3; Cube1 uses a separate slot.
Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
```

- `vec1Done` 保证 `aftermask` 已就绪
- `vec2Done` 保证前一 stage 的 `h/v` workspace 已被 Vec2 消费完，可以被 Cube2/3 复用

### 7.3 启动与收尾

由于 1 AIC 绑定 2 个 AIV subBlock，启动/收尾都按 block 整体考虑：

```cpp
// 启动：每个 AIV subBlock 都预先 set vec2Done[0/1]，让绑定 AIC 首轮 Cube2/3 不必等待
// gdn_fwd_o_kernel.hpp:361-362（位于 if ASCEND_IS_AIV 分支内）
Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

// 收尾：AIC 末尾 wait vec2Done 两个 stage，确保最后两 stage 的 Vec2（2 个 subBlock 共同）完成
// gdn_fwd_o_kernel.hpp:350-351（位于 if ASCEND_IS_AIC 分支内）
Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);
```

### 7.4 通信效率优化

- **Flag ID 复用**：8 个 flag 全程复用，不动态分配，避免 flag 耗尽
- **PIPE_FIX 关联**：Cube 侧 `CrossCoreSetFlag<0x2, PIPE_FIX>`，让 flag 与 FIX 流水线绑定，自动在 mmad 完成时触发
- **PIPE_MTE3 关联**：Vec 侧 `CrossCoreSetFlag<0x2, PIPE_MTE3>`，让 flag 与 GM 写回绑定，写完即可被对端消费
- **Ping-Pong 解耦**：stage0 计算与 stage1 搬运重叠，flag 同步只发生在 stage 边界，每轮开销均摊

---

## 八、关键技术难点与解决方案

### 8.1 难点一：变长序列的尾块处理

**挑战**：变长模式下每个子序列的最后一块 `blockTokens < chunkSize`，但 Cube mmad 和 Vec epilogue 的 tile shape 都是按 `chunkSize` 固定大小实例化的。

**解决方案**：
- 调度器为每个任务计算 `blockTokens` 和 `isFinalState`：
  ```cpp
  offsets[currStage].blockTokens = isFinalState ?
      (batchTokens - batchChunkIdx * chunkSize) : chunkSize;
  ```
- Cube 侧 `GemmCoord` 使用实际 `blockTokens`：`cube1Shape{blockTokens, blockTokens, kHeadDim}`
- Vec 侧通过 `isContiguousFullTile = (chunkSize == fullChunkSize && nActual == alignedNActual)` 切换 `DataCopy` 与 `DataCopyPad`，避免越界访问

### 8.2 难点二：门控掩码的 BRC 对齐

**挑战**：`exp(g_left - g_up)` 需要 `[C, C]` 矩阵，但 Ascend 950 的 `Broadcast` 指令要求 8 行对齐，而 chunk 大小 64 / 128 在 subBlock 切分后可能不对齐。

**解决方案**：
- 引入 `gbrcRealStart = gbrcStart & ~7`（按 8 对齐的起点）
- `gbrcEffStart = gbrcStart - gbrcRealStart`（subBlock 内的有效起始偏移）
- 对 `mActualThisSubBlock > 32` 的情况拆分为 2 个 stage，分别处理上下半部，确保每段 BRC 都能对齐

### 8.3 难点三：双缓冲 workspace 容量管理

**挑战**：AIC 和 AIV 同时活跃，需要为每个 cube 核预留 2 个 stage 的 workspace，V=256 时单核 workspace 高达：
```
2 * (chunkSize * vHeadDim * 4B) * 2 (v+h) +
2 * (chunkSize * chunkSize * 4B) * 2 (attn+aftermask) ≈ 656KB / 核
```

**解决方案**：
- Host 侧 `WorkspaceTiling` 按 `aicCoreNum` 精确分配：
  ```cpp
  workspaceOffset += aicCoreNum * chunkSize * vHeadDim * sizeof(float) * PING_PONG_STAGES;
  ```
- 5 个 workspace 全部 512B 对齐（`AlignWorkspaceSize`），满足 GM 访问对齐要求
- 额外预留 `16MB RSV + 16MB RSV` 两段缓冲区，应对系统 API 抢占

### 8.4 难点四：跨核 flag 时序（1 AIC : 2 AIV subBlock）

**挑战**：1 AIC 天然绑定 2 个 AIV subBlock，AIC 的某个 stage 输出必须等绑定的 2 个 AIV subBlock 都消费完才能被覆盖，反之 AIV 也要等 AIC 写完才能消费；flag 时序错配会导致数据竞争或死锁。

**解决方案**：
- 所有 flag 走 `0x2` broadcast 模式，2 个 AIV subBlock 各自执行 set 后，AIC 侧 `WaitFlag` 一次即可确认 2 个 subBlock 都已完成
- `vec2Done` 在每个 AIV subBlock 启动时预先 set，确保绑定的 AIC 首轮 Cube2/3 不死等
- AIC 末尾 wait `vec2Done[0/1]` 两个 stage，覆盖最后两 stage 的 Vec2（2 个 subBlock 共同）完成

### 8.5 难点五：AIC/AIV 任务对齐（block 内 1:2 共享上下文）

**挑战**：同一 block 内的 1 个 AIC 与 2 个 AIV subBlock 必须看到完全一致的任务序列，否则会因 `offsets` 错位导致数据竞争。AIC 用 `GetBlockIdx()` 取核号，AIV 用 `GetBlockIdx() / GetSubBlockNum()` 还原 cube 编号，二者必须对齐。

**解决方案**：
- 调度器在 Cube/Vec 两边都基于同一 `cubeCoreIdx` 推进 `taskIdx`，保证同 block 内的 AIC 与 2 个 AIV subBlock 共享同一份 `offsets/stage` 上下文
- `BlockSchedulerGdnFwdO::Init` 对 Vec 侧显式做 `/ subBlockNum` 还原，确保 2 个 AIV subBlock 看到的 `taskIdx` 与绑定 AIC 完全一致
- Vec 侧的 `coreIdx < coreNum * subBlockNum` 判断保证 2 个 subBlock 都进入处理分支
- 在数据维度上通过 `subBlockIdx` 切分行（`mActualPerSubBlock = CeilDiv(mActual, subBlockNum)`），2 个 subBlock 不会写同一行，避免写冲突

---

## 九、性能优化策略

### 9.1 双缓冲流水（Ping-Pong）

- `PING_PONG_STAGES = 2`，同一 block 内 AIC 处理 stage `s` 时，绑定的 2 个 AIV subBlock 处理 stage `s-1`，反之亦然
- 配合 `CrossCoreFlag` 数组实现无锁流水，理论流水吞吐 ≈ 单 stage 耗时的 max(AIC, AIV)，而非 AIC + AIV
- 2 个 AIV subBlock 总是处理同一 stage 的不同行（不会出现 subBlock 0 在 stage0、subBlock 1 在 stage1 的情况）

### 9.2 三段 Cube 并行调度

- Cube1（Q@K^T）与 Cube2/3（Q@H、Attn@V）通过 `vec2Done` 解耦：
  - Cube1 只依赖 `vec2Done`（前一 stage h/v workspace 释放）
  - Cube2/3 依赖 `vec1Done`（当前 stage aftermask 就绪）
- 不同 Cube 段在流水上可重叠，提高 Cube 利用率

### 9.3 Tile 形状自适应

```cpp
if (vBlockDim <= 128) {
    blockMmadQH128(...);   // L1=128×128×128, L0=128×128×128
} else {
    blockMmadQH256(...);   // L1=128×256×128, L0=128×256×64
}
```

- V=128 时用 128×128×128 全 tile，最大化 L0 利用率
- V=256 时调整为 128×256×64，避免 L0 容量溢出，同时保持 K 维 128 不切

### 9.4 UB Ping-Pong + 流水线事件

- 所有 UB tensor（`gUbTensorPing/Pong`、`aUbTensorPing/Pong` 等）双备份
- 通过 `HardEvent::V_MTE2` / `MTE2_V` / `V_MTE3` / `MTE3_V` 事件 ID 配合 pingpongFlag 切换
- 实现 GM→UB 搬运、V 计算、UB→GM 写回三段流水

### 9.5 BRC 指令优化门控广播

```cpp
AscendC::Copy(gcompUbTensor, gUbTensor, 64, 2, {1, 1, 8, 8});  // 64 行 g 数据排布
AscendC::Broadcast<float, 2, 0>(gbrcUpUbTensor, gcompUbTensor, dstUpShape_, srcUpShape_, shareUbTensor);
AscendC::Broadcast<float, 2, 1>(gbrcLeftcastUbTensor, gcompUbTensor[...], dstLeftShape_, srcLeftShape_, shareUbTensor);
```

- 一次搬入 `g`，两次广播分别得到行广播和列广播
- 利用 `shareUbTensor` 作为 BRC 的辅助空间，避免临时分配

### 9.6 掩码复用

AIV 启动时一次性生成 64×64 下三角 mask UB：
```cpp
AscendC::Duplicate<float>(maskUbTensor, 0.0, 64*64);
for (uint32_t i = 0; i < 64; ++i)
    AscendC::Duplicate<float>(maskUbTensor[i*64], 1.0, i+1);
```

后续所有 chunk 共用此 mask，避免每个任务重复生成。

### 9.7 多 dtype 路径特化

```cpp
if (tilingData->dataType == 1) {        // bf16
    if (tilingData->gDataType == 2) {    // g: fp32
        ChunkFwdOKernelImpl<bfloat16_t, float, float>(...);
    } else {                              // g: bf16
        ChunkFwdOKernelImpl<bfloat16_t, bfloat16_t, float>(...);
    }
}
```

- 4 种 `(Q dtype, G dtype)` 组合走不同的模板实例
- `Cast` 指令在 `GElementInput != float` 时才调用，避免 fp32 输入下的冗余 cast
- `Cast` 模式区分：fp16 用 `CAST_NONE`，bf16 用 `CAST_RINT`（符合 bf16 round-to-nearest-even 语义）

### 9.8 GM 对齐与预留

```cpp
static constexpr size_t CHUNK_FWD_O_GM_ALIGN = 512;
static constexpr size_t CHUNK_FWD_O_WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
```

- 512B 对齐满足 Ascend 950 GM 访问最佳对齐粒度
- 16MB 预留段应对 ACL 私有格式透传与系统 API 抢占，避免 workspace 越界

---

## 十、Ascend 950 关键架构特性利用总结

| 950 特性 | 算子利用点 | 代码位置 |
|---|---|---|
| `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC : 2 AIV subBlock） | 同 block 内 AIC 跑 3 段 mmad，2 个 AIV subBlock 协同处理同一 stage 的不同行 | `chunk_fwd_o.cpp:65` |
| `CrossCoreFlag` 跨核同步 | 8 flag 实现 4 同步点 × 2 stage | `block_scheduler_gdn_fwd_o.hpp:98-101` |
| `PIPE_FIX` 流水 | Cube mmad 完成自动触发 flag | `gdn_fwd_o_kernel.hpp:287` |
| `PIPE_MTE3` 流水 | Vec GM 写回完成自动触发 flag | `gdn_fwd_o_kernel.hpp:389, 408` |
| `Broadcast<float, 2, 0/1>` | 门控行/列广播 | `block_epilogue_gdn_fwdo_qkmask.hpp:210-211` |
| `DataCopyPad` | 尾块非整 tile 读写 | `block_epilogue_gdn_fwdo_qkmask.hpp:240` |
| `Cast` 多 round mode | bf16 用 RINT，fp16 用 NONE | `block_epilogue_gdn_fwdo_qkmask.hpp:248, 257` |
| `HardEvent` 事件 ID | UB Ping-Pong 切换 | `block_epilogue_gdn_fwdo_output.hpp:218-219` |
| L1/L0 大容量 | 128×256×64 tile（V=256） | `gdn_fwd_o_kernel.hpp:107-108` |

---

## 十一、总结

`ChunkFwdO` 是针对 Ascend 950 AiCore 深度优化的 GDN 前向输出算子，其设计要点：

1. **数学上**：把 delta rule 的块内注意力与块间递推状态合成统一为 `scale·(Q·exp(g)@H + tril(Q@K^T·exp(Δg))@V)`，一次输出 `o`。
2. **架构上**：1 AIC 天然绑定 2 个 AIV subBlock（`MIX_AIC_1_2`），同 block 内 AIC 跑 Cube 三段 mmad、2 个 AIV subBlock 协同跑 Vec 两段 epilogue（通过 `subBlockIdx` 切分行），5 块 GM workspace ping-pong。
3. **同步上**：8 个 `CrossCoreFlag` 编排 4 个同步点的双缓冲流水，启动时预 set、收尾时双 wait 防死锁。
4. **变长上**：通过 `chunk_offsets` 的 `[tb, c]` pair 表达任意子序列的尾块，配合 `isFinalState` 自适应 `blockTokens`。
5. **性能上**：通过 Tile 自适应、BRC 广播、UB Ping-Pong、HardEvent 流水、掩码复用、dtype 特化等多维度优化，在 950 上实现 Cube/Vector 双流接近峰值的吞吐。

算子整体设计符合 Ascend C 开发规范，模块划分清晰（Host Tiling → Kernel 主体 → Scheduler → Epilogue），便于扩展和维护。

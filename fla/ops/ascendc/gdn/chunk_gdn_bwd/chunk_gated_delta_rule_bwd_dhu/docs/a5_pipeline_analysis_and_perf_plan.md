# chunk_gated_delta_rule_bwd_dhu A5 Cube/Vector 执行流程与流水分析及性能优化方案

> 分析日期：2026-09-17
> 分析对象：`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/`（A5/arch35 路径）
> 参考文档：本目录 `design.md`；Ascend C API v950（`D:\project\documents\AscendC-API_v950_20260831-162807`）
> 说明：本文档为代码级分析产出，同步协议部分已逐条对照 950 API 文档核实；文末附基于分析结论的性能优化方案（尚未实施）。

---

## 1. 算子定位与计算流程回顾

`chunk_gated_delta_rule_bwd_dhu` 是 GDN 反向链路中沿时间轴**倒序扫描**的递推算子，输出三路：

- `dh [B,HV,NT,K,V]`：每个 chunk **入口处**（更新前）的状态梯度 `dH_old`；
- `dv2 [B,HV,T,V]`：上游 value 局部梯度 `dv` 叠加状态通路贡献 `K @ dH`；
- `dh0 [N,HV,K,V]`：递推到序列开头的最终 `dH`（仅 `h0` 非空时输出）。

单个 `(seq, hv)` 的倒序递推（`hk = hv / (HV/HK)`，M 为 chunk 实际长度，`g/gK` 互斥二选一，`E_g=exp`，`E_gK=exp2`）：

```text
dH_old = dH
dh[seq, hv, chunk] = dH_old                       # 公开输出

dv_state = K_blk @ dH_old                         # GEMM0, [M,V]
if g:  dv_state[t,:] *= exp(g_last - g_t)         # gK 分支不加 token gate
dv2_blk = dv_state + dv_blk                       # 公开输出

dH_decay = dH_old
if g:  dH_decay *= exp(g_last);  Q_gate = q * exp(g_t)
elif gK: dH_decay[k,:] *= exp2(gK_last[k]);  Q_gate = q
term_q = Q_gate^T @ d_o                           # GEMM1, [K,V]，乘 scale
term_w = w^T @ dv2_blk                            # GEMM2, [K,V]
dH = dH_decay + term_q * scale - term_w           # fp32 carry，供更早 chunk
```

递推状态 `dH` 全程 fp32（workspace state 段 + 专用 `stateFp32` ping/pong UB，不做 q-dtype round-trip）。分核粒度为 `seq + 4-head window`（**不含 chunkIdx**，同一 `(seq,hv)` 的 chunk 必须单 task 内倒序串行），`taskNum = seqNum × ceil(HV/headsPerTask)`。

---

## 2. A5 核型与任务调度

kernel 为 `KERNEL_TYPE_MIX_AIC_1_2`（`op_kernel/chunk_gated_delta_rule_bwd_dhu.cpp:90`）：每个 AI Core = **1 AIC（block）+ 2 AIV（subblock）**。950 API 文档确认这是唯一支持 mode-4（AIC 与单个 AIV 1:1 同步）的核型配置。

两侧对同一 task 空间的映射：

- Cube 侧（`arch35/chunk_gated_delta_rule_bwd_dhu_cube.h:123-126`）：`blockIdx` 即核号，网格步进遍历 taskNum；
- Vector 侧（`arch35/chunk_gated_delta_rule_bwd_dhu_vector.h:325`）：`coreIdx = GetBlockIdx() / GetSubBlockNum()`，2 个 subblock 各自执行相同 task 循环，workspace slot 地址 `coreIdx*8 + windowStartSlot + headOffset` 两侧算出同一物理位置。

**head 级分工**是 A5 Vector 的关键调度特征：`headOffset % subBlockNum == subBlockIdx_` 决定归属（vector.h:346），subblock0 完整负责 h0/h2，subblock1 完整负责 h1/h3——两个 AIV **并行**处理 4-head window 的不同 head。非归属 subblock 在每个同步点只做 flag 配平（vector.h:382-385、513-515），不读写任何数据。

每核保留 8 个 per-head workspace slot（两个 4-head window 按 `0..3 ↔ 4..7` 轮转，`taskRound & 1`）。

---

## 3. 同步协议（已对照 950 API 文档核实）

### 3.1 mode 2 —— 阶段级粗粒度握手（AIC ↔ 全部 AIV）

| flag | flagId | set 方 | pipe | 含义 |
|---|---|---|---|---|
| `vecToCube` | 2 | 两个 AIV 各 set 1 次 | PIPE_MTE3 | MTE3 搬运（dh 写回、qg 直写 L1、dv2 写回）完成后 Cube 才可继续 |
| `cubeToVec` | 4 | AIC set 1 次 | PIPE_FIX | Fixpipe 输出（CV 直传 / GM copyout）完成后 AIV 才可继续 |

配平规则：Cube 的 1 次 Wait 需要**两个** AIV 都 SetFlag；AIC 的 1 次 SetFlag 解除**两个** AIV 各自的 Wait。因此非归属 subblock 必须空转 set，否则计数器不齐导致 Cube 死锁。

### 3.2 mode 4 —— 矩阵 CV per-tile 细粒度握手（AIC ↔ 单个 AIV）

flagId 命名空间映射（API 文档原文确认）：

```text
AIV0 的 flagId 0-15  ←→  AIC 视角 flagId 0-15
AIV1 的 flagId 0-15  ←→  AIC 视角 flagId 16-31   （CV_SUBBLOCK_FLAG_STRIDE = 16）
```

常量定义（`arch35/chunk_gated_delta_rule_bwd_dhu_common.h:30-33`）：

```text
CV_BUFFER_COUNT                    = 2     # ping/pong 两个 CV buffer
CV_SUBBLOCK_FLAG_STRIDE            = 16
MATRIX_CV_AIV_TO_AIC_FLAG_BEGIN    = 0     # AIV → AIC：free 方向（PIPE_V）
MATRIX_CV_AIC_TO_AIV_FLAG_BEGIN    = 6     # AIC → AIV：ready 方向（PIPE_FIX）
```

Cube 侧用 `(headOffset & 1) * 16` 偏移寻址目标 subblock（cube.h:343-344、557-558）：

- `AIV_TO_AIC(flag) + cvListId`：AIV 归还 CV buffer 的 **free**（cast 消费完后 set，PIPE_V）；
- `AIC_TO_AIV(flag) + cvListId`：AIC 推送 CV tile 的 **ready**（copyL0CToUB 入队后 set，PIPE_FIX）；
- `cvListId`（0/1）区分 ping/pong，形成 per-tile 的 ready/free 环。

### 3.3 每 chunk 每 head 的同步计数（守恒验证）

```text
vecToCube：stage0 末 1 对（2 AIV 各 set）→ Cube stage1 GEMM0 前 1 次 wait
           stage1 末 1 对               → Cube stage2 GEMM2 前 1 次 wait
cubeToVec：Cube stage1 末 1 次 set → 2 AIV 在 stage1 入口各 1 次 wait
           Cube stage2 末 1 次 set → 2 AIV 在 stage2 入口各 1 次 wait
```

stage2 的 Vector 侧**没有尾部 vecToCube**（vector.h:578-635 循环结尾无 set）——下一 chunk 的 stage0 set 由 Cube 的下一次 wait 消费，chunk 间依赖靠 fp32 state 的 `MTE3_MTE2` HardEvent 保序（`CopyOutStateRows` 末尾 set、`CopyInStateRows` 开头 wait，vector.h:854-874），不需要跨核 flag。

CV flag 计数守恒：`InitVectorEvents` 在**两个** subblock 上无条件 set flags 0/1（从 AIC 视角即 0/1 和 16/17 各 +1，vector.h:694-696），每个 AIC wait 必有先行 set；每 tile 消费后 AIV set free 但 AIC 不再消费最后一格——恰好每 flag 残留 1 个未消费计数，由 `DrainPipeFlags`（cube.h:821-825）无条件 wait 4 个 flag 收尾。fp16 路径运行时不走 CV，init set + drain wait 同样配平，不会挂死。

---

## 4. Cube 侧执行流程

### 4.1 片上内存预算（静态断言保证）

L1（512KB，cube.h:693-709）：

| 段 | 容量 | 说明 |
|---|---|---|
| K resident | 2 × 32KB | L1A ping/pong，同 hk 组连续 V head 复用 |
| W resident | 2 × 32KB | L1A，按 `workspaceSlot & 1` |
| qg L1A scratch | 4 × 32KB | **与 Vector 共享**（偏移 128KB 起），Vector MTE3 直写落点 |
| L1B scratch | 2 × tile | state/dO/dv2 中转 |

V=128 合计 320KB，V=256 合计 384KB，均在 512KB 内。

L0A/L0B 各 2 槽 ping-pong；`L0_K_TILE = V==256 ? 64 : 128`（V=256 时 K 维拆两片 MMAD 累加，中间片 `unitFlag=0b10`、末片 `0b11` 触发 copyout）；L0C tile `K×V×4B` fp32，双槽与否由 `ArchTag::L0C_SIZE` 静态判定。

### 4.2 stage1（第一 head 循环，cube.h:152-407）

对每个 headOffset：

1. **预取不等 Vector**：K resident 加载（`needLoadKResident` 判定——同 hk 组复用同一 L1 slot，`releaseKAfterUse` 只在组末 head 最后一次 L1→L0A 拷贝后归还 MTE1_MTE2）与 dO 到 L1B scratch，都在 `CrossCoreWaitFlag(vecToCube)` **之前**发起（cube.h:210-229）；
2. 等 vecToCube（保证 dh 写回 + qg 进 L1 完成）；
3. **GEMM0** `dvState = K @ dh[chunk]`：dh 直接读公开输出。bf16 且非（V=256 且 chunkLen>64）走**内联 MMAD + L0C→UB 矩阵 CV 直传**（cube.h:264-369）：L0C 结果按 `vecRow` 行 tile 从 Fixpipe 直写目标 subblock 的 UB，每 tile 先 wait 该 subblock 的 free、再 copy、再 set ready；否则 `RunResidentMmad` + `CopyL0CToGm` 落 GM workspace；
4. **GEMM1** `termQ = qg^T @ dO`：qg 从 L1A scratch 消费（g 分支 slot=headOffset；gK 分支 slot=组首 head 偏移，cube.h:392-396），结果写 GM workspace；
5. `CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVec)`（cube.h:406）——termQ 的 GM 写已在同 pipe FIX 上排在 flag 之前。

### 4.3 stage2（第二 head 循环，cube.h:408-594）

对每个 headOffset：

1. W^T 加载 L1A resident，在 wait 之前发起；
2. 等 vecToCube（dv2 写回完成），dv2（公开输出）到 L1B scratch；
3. **GEMM2** `termW = w^T @ dv2`。bf16 CV 路径的关键差异：**先 `SwitchL0C()` 再 `SetFlag(cubeToVec)`，然后才开始 CV tile 推送**（cube.h:552-554）——即 early-notify：Vector 可提前进入 stage2 做 termQ/state 的 GM 读入，termW 逐 tile 可见性由 mode-4 ready/free 保证。

Cube 全程只做 3 个 GEMM，全部走 `GM→L1→L0A/L0B→TileMmadTla→L0C→Dst` 的 tile 级 Catlass 路径（`RunResidentMmad`，cube.h:828-907），Dst 按分支选 GM 或 AIV UB；禁止 block 级接口。

---

## 5. Vector 侧执行流程

### 5.1 UB 布局（vector.h:245-267）

bf16 时**最先**分配 `matrixCvPing/Pong`（`vecRow×V×DT`，UB 偏移 0 和 `vecRow*V*sizeof(DT)`）——与 Cube 侧 `resource.ubBuf.GetBufferByByte(0)/(cvStrideBytes)` 指向**同一物理 UB**，即 CV 直传落点；`CopyL0CToUB` 的第 4 个参数 `headOffset & 1` 选择写入哪个 subblock 的 UB。

其余 UB：qInput/gInput/output 各自 ping-pong、`stateFp32` ping-pong（fp32 递推 carry 专用）、按 head 驻留的 `gRawAll/gateFactorAll/dvGateFactorAll`（各 4×gateElems fp32）、fp32 计算区 qFp32/outFp32。`vecRow` 由 host tiling 按 UB 预算搜索（下限 8）。

### 5.2 入口动作

- `hasDh0` 时按 tiling 切分用 `AscendC::Fill` 清零整个 dh0 GM + `SyncAll<true>()` 全 Vector 同步（vector.h:297-323）；
- task 开始后、chunk 循环前，归属 subblock 把所属 head 的 fp32 state workspace 按行 tile 清零（vector.h:351-363）——这是"dht 被忽略、入口固定置零"的实现位置（kernel 入口 `(void)dht`，chunk_gated_delta_rule_bwd_dhu.cpp:78-79）。

### 5.3 每 chunk 三个 head 循环

**stage0（vector.h:373-510）**：

1. gate 准备：g 分支搬 `g[M]`→fp32→`Exp`（`useExp2` 时先乘 LN2）驻留 `gateFactor`；gK 分支只搬最后一个有效 token 的 `gK_last[K]`→`ln2·x→Exp` 得 K 行衰减系数；
2. **K 行 tile 循环**（vecRow 步进）：state fp32 搬入 → `CopyOutFp32Rows` 把 **pre-decay** 状态 cast 写公开输出 `dh[chunk]` → decay（g：整 tile 乘标量 `exp(g_last)`，`MulScalarPtrRegbase`；gK：每行乘 `exp2(gK_last[k])`，`MulRowsByFactorsRegbase`）→ fp32 写回 workspace。dh 写 decay 前状态由 V pipe 顺序保证（cast 先入队）；
3. **qg 生成**（token 行 tile 循环）：q→fp32→（g：每行乘 `exp(g_t)`）→cast DT→ **MTE3 `BLOCK_MODE_VECTOR` DataCopy 直写 L1A NZ 分形 slot**（vector.h:449-494，目的偏移 `colBlock*128*16 + row*16`）。g 分支每 head 独占 slot=headOffset；gK 分支仅组首 head 生成 `qg=q`（`produceQG = headOffset==0 || hq != (hv-1)/HRatio_`，vector.h:454-458），组内其余 head 复用共享 slot——与 Cube 消费侧公式一致（cube.h:393-396）；
4. g 分支生成 `dvGateFactor = exp(g_last − g_t)` 驻留（stage1 复用，不重复搬 g）；
5. set vecToCube（PIPE_MTE3，L1 直写与 dh 写回完成后生效）。

**stage1（vector.h:511-576）**：wait cubeToVec → token 行 tile 循环：dvState 获取（bf16 非 V256 长块：wait CV ready → 从 `matrixCvBuf` cast → set CV free；否则 GM workspace 读）+ dv 读入 → `dv2 = dvState·dvGateFactor + dv`（g）/直接相加（gK）→ cast 写公开输出 dv2 → set vecToCube。

**stage2（vector.h:578-635）**：wait cubeToVec → K 行 tile 循环：termQ（GM）+ termW（CV 消费或 GM）+ state 搬入 → `state += termQ·scale − termW`（Muls/Sub/Add，全 fp32）→ 写回 workspace。**无尾部通知**。

**task 出口**（vector.h:638-664）：hasDh0 时最终 state 写 dh0；`stateVFirst` 走 `CopyOutDh0VFirst`（16×16 TransDataTo5HD 转置 + DataCopyPad 尾块处理）。

---

## 6. A5 相对 A2 的三条直传捷径

1. **qg：UB→L1A 直写**（所有 dtype）。Vector MTE3 用 `BLOCK_MODE_VECTOR`（1×16 cube 分形传输单位，half 的 blockLen 单位 32B，API 文档确认）直接把 qg 写进 L1 的 NZ 布局，Cube 从 L1A scratch 消费，省掉 A2 的 UB→GM workspace→L1 两跳。两侧 L1 偏移约定对齐：Vector 的 `qgL1ScratchOffset = 4×32KB` 与 Cube 的 `L1A_SCRATCH_OFFSET`（K/W resident 之后）指向同一物理 L1。
2. **dvState/termW：L0C→UB 矩阵 CV 直传**（仅 bf16；V=256 且 chunkLen>64 时回退 GM——L0C tile 128×256×4B 与 UB CV buffer 容量所限）。Cube Fixpipe 绕过 GM 直接写目标 subblock UB 固定偏移，per-tile mode-4 ready/free 握手。
3. **regbase/MicroAPI 向量化**：`__simd_vf__` 单趟循环完成 gate 生成、state decay、dv2 合成（`MulRowsByFactorsAddRegbase` 一趟完成乘 gate 加 dv），避免逐元素标量路径（vector.h:31-182）。

---

## 7. 流水时序分析

### 7.1 chunk 内时序（4-head window）

```text
AIC:   [h0:预取K/dO][等v2c][GEMM0→CV推dvState][GEMM1 termQ][set c2v]
       [h1:预取][等v2c][GEMM0][GEMM1][set c2v] [h2...] [h3...]
       ────────────────────────────────────────────────────────────▶
       [h0:预取W][等v2c(dv2)][GEMM2][set c2v(早)][CV推termW] [h1...] ...

AIV0:  [h0 st0:dh/decay/qg][set][h2 st0][set]      ← 与 AIV1 并行
       [等c2v][h0 st1:消费dvState(CV)→dv2][set] [h2 st1]...
       [等c2v][h0 st2:state+=termQ·s−termW] [h2 st2]...

AIV1:  [h0: set(配平)][h1 st0][set][h3 st0][set]
       [等c2v][h1 st1][h3 st1]... [等c2v][h1 st2][h3 st2]...
```

### 7.2 实际重叠点

- **head 粒度错位流水**：Cube 处理 h1 的 GEMM0/GEMM1 时，AIV 正在算 h0 的 dv2；Cube 做 h0 的 GEMM2 时，AIV 可能在做 h1 的 stage1。4 个 head 摊开 Cube/Vector 速度不匹配；
- **stage2 early notify**：Cube 在 termW 首个 CV tile 推送前就 set cubeToVec，AIV 的 termQ/state GM 读入与 Cube 的 Fixpipe 逐 tile 并行；
- **预取不阻塞**：两个循环的 K/W/dO 搬入都在 wait flag 之前发起，MTE2 与等待重叠；
- **双 AIV 并行**：subblock0/1 各自完整负责一半 head，stage0 向量计算吞吐 ×2。

### 7.3 串行约束与气泡

- chunk 间因 fp32 state carry 必须严格倒序串行（`dH_in(c) = dH_out(c+1)`），下一 chunk stage0 必须等本 chunk stage2 的 state 写回（MTE3_MTE2 事件）；
- **stage1 消费延后（已确认气泡）**：stage1 的 cubeToVec 在 termQ GEMM **之后**才 set——dvState 的 CV tile 其实早已在 UB，但 AIV 被 coarse gate 挡住不能提前消费；
- qg slot 复用约束：gK 共享 slot 保持到组内最后一个 head 的 termQ 消费完成，下一 chunk 的覆盖靠 4 个 head 的 stage2 cubeToVec 握手整体隔离。

---

## 8. 已确认的文档-代码不一致与已知缺口

1. **K 驻留层级**：design.md §5.2/§8 表述为"A5 BF16 V=128 把 K 保持在 **L0A slot0** 供同组 V heads 复用"，但代码实际是 **L1 级** resident（`cachedKResidentValid_` ping/pong，每 head 仍重新执行 L1→L0A 拷贝，cube.h:191-213）。L0A slot0 跨 head 驻留未落地，属于文档超前描述或未合入的优化。
2. **dht 被忽略**：kernel 入口 `(void)dht`（chunk_gated_delta_rule_bwd_dhu.cpp:78-79），递推入口固定从 0 初始化（vector.h:351-363）。
3. **dh0 dtype 跟随 q** 而非 design.md §2.2 要求的 fp32 `[N,HV,K,V]`；入口需 `Fill` 整体清零再出口覆写。
4. A5 的 qg/dvState/termW workspace 段仅保留布局（运行时有效数据为 dhState/termQ）。

---

## 9. 性能优化方案（规划，尚未实施）

### 9.1 优化 1（P0）：stage1 cubeToVec 提前通知（对齐 stage2 已有模式）

**现状**：Cube stage1 中 GEMM0（dvState）的 CV tile 推送完成后，AIV 本可开始消费，但 `cubeToVec` 在 GEMM1（termQ）完成之后才 set（cube.h:406），AIV 在 GEMM1 期间空等。

**改动**（仅 A5 arch35；A2 暂不动）：

- 把 headOffset 第一循环末尾的 `CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_)` 移到 GEMM0 完成、GEMM1 开始之前（bf16 CV 分支与 GM 回退分支的 if/else 汇合处、`tensorTermQ` 构造之前），两路径统一生效。

**安全性论证**：

- CV 路径 dvState 的 tile 级可见性由 mode-4 ready/free 保证，不依赖 coarse gate；
- GM 路径 dvState 写与 set 同在 PIPE_FIX，同 pipe 保序；
- flag set/wait 次数不变，mode-2 配平不受影响；
- termQ 可见性由 stage2 的 cubeToVec #2（PIPE_FIX 上晚于 termQ 的 copyL0CToGm）保护，不受本次改动影响；
- gK 共享 qg slot 的生命周期由 vecToCube #1（stage0 末）保护，不涉及本次改动。

### 9.2 优化 2（P1，优化 1 验证后再做）：GEMM0 的 K 驻留 L0A slot0（bf16 + V=128 + GVA）

**现状**：同 hk 组的连续 V head 只做了 L1 级 resident，每个 head 仍重新执行 32KB 的 L1→L0A 拷贝。design.md 声称的 "K 保持在 L0A slot0" 未落地。

**改动**（cube.h GEMM0 内联 MMAD 段，cube.h:264-369）：

- bf16、V=128、组内非首 head 且 K 基址未变时，跳过 `copyL1ToL0A_DvState`，直接复用上次 GEMM0 的 L0A 内容；
- K 固定使用 L0A slot0，组末 head 的 GEMM0 完成后才归还 slot0（`releaseKAfterUse` 时点增加 L0A 维度的 M_MTE1 free）；termQ 继续用现有轮转槽位，避开 slot0 冲突；
- 新增状态变量 `cachedKL0AValid_/cachedKL0ASlot_`，在 chunk 边界与 `cachedKResidentValid_` 同处重置（cube.h:150-151）；
- V=256 不启用（L0_K_TILE=64 双分片轮转，驻留无意义）；noGVA（HK==HV）无收益但逻辑保持正确。

### 9.3 文档修正（随优化 2 一起提交）

`docs/design.md` §5.2/§8/§13 中 "K 驻留 L0A slot0" 的表述目前超前于代码——若优化 2 落地则补记实现细节（slot 分配、事件闭环、启用条件）；若暂缓则先把表述改回与 L1 resident 实际一致，避免下轮误导。

### 9.4 实施与验证流程（按 AGENTS.md 03→04→05 路由）

1. **改动前基线**（A5 远程，msprof）：
   - 现有 perf 用例 `tests/atk/chunk_gated_delta_rule_bwd_dhu/atk_chunk_gated_delta_rule_bwd_dhu_perf.json`（B=1,HK=HV=4,T=512,V=128,chunk=64,bf16）采集 Task Duration、cube_wait/mte1_ratio 等；
   - 补充 2-3 条 GVA/大 T 用例（如 B=1,HK=8,HV=32,T=8192,GVA；B=1,HK=2,HV=4,T=8192；V=256 回归用例）观察 head 流水与 K 复用收益；
   - 基线写入 `prof_output/`（沿用 perf-verify 的 BASELINE.md 格式，新建 dhu 小节）；
   - 改动前用 reference-snapshot 保存当前 wheel+OPP 对照版本（防死锁回退）。
2. **代码修改**：优化 1 → 构建（npu-build-install，A5）→ 验证 → 优化 2 → 构建 → 验证，两步独立提交便于归因。
3. **精度验证**（每步优化后）：
   - `torch_custom/fla_npu/test/test_npu_bwd_dhu_gva.py`（18 case 双标杆，覆盖 V=256/GVA/变长/大 T）；
   - `test/test_chunk_gated_delta_rule_bwd_dhu.py`（含 varlen smoke）；
   - ATK 200 case（`-scope=accuracy`，全定长）+ `_mss.json` racecheck（同步时序改动重点防竞态）。
4. **性能验证**：同口径 msprof 采集，与基线逐用例对比 Task Duration；预期优化 1 在 bf16 主路径消 AIV stage1 空等，优化 2 在 GVA 用例降低 mte1 占比。
5. **回归判据**：精度全部 pass；性能不出现用例级回退（V=256/noGVA 用例持平即可）；无死锁（若挂死先用 reference-snapshot 恢复对照版本区分代码/环境问题）。

### 9.5 暂不做的项（登记为后续）

- termQ 的 L0C→UB CV 直传（生命周期跨 stage2，复杂度高）；
- head 间 overlap 流水（需重设计 workspace ring 与 ready/free 计数）；
- A2 侧 early-notify 同步移植；
- dht/dh0 语义补齐（独立功能项，不与性能优化混提）。

---

## 10. 关键文件索引

| 文件 | 作用 |
|---|---|
| `op_kernel/chunk_gated_delta_rule_bwd_dhu.cpp` | kernel 入口，MIX_AIC_1_2 声明，AIC/AIV 分发 |
| `op_kernel/arch35/chunk_gated_delta_rule_bwd_dhu_cube.h` | A5 Cube：3 GEMM、K/W resident、CV 直传 |
| `op_kernel/arch35/chunk_gated_delta_rule_bwd_dhu_vector.h` | A5 Vector：gate/state/dv2、qg UB→L1 直写、regbase |
| `op_kernel/arch35/chunk_gated_delta_rule_bwd_dhu_common.h` | A5 共享常量（flag 编号、HEADS_PER_TASK、varlen helper） |
| `op_kernel/arch35/chunk_gated_delta_rule_bwd_dhu_struct.h` | TilingData 结构 + 模板参数（24 组合） |
| `op_host/op_tiling/chunk_gated_delta_rule_bwd_dhu_tiling_processor.h` | tiling（headsPerTask、vecRow、workspace 布局、dh0 清零切分） |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py:531` | Python 入口 `npu_chunk_gated_delta_rule_bwd_dhu` |
| `torch_custom/fla_npu/test/test_bwd_dhu.py` | CPU 标杆（fp64/npu/fp32 三模式 golden） |
| `torch_custom/fla_npu/test/test_npu_bwd_dhu_gva.py` | 双标杆主测试（18 case） |
| `tests/atk/chunk_gated_delta_rule_bwd_dhu/` | ATK 工程（200 accuracy + perf + mss 用例） |

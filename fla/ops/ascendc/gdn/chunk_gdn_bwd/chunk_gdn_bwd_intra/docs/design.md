# ChunkGdnBwdIntra Ascend C 融合算子设计

> 方案设计规则版本：`V1`
>
> 本文按仓库 `docs/agents/03-solution-design.md` 的 R01--R19 及 R02-A 独立推导。
> Stage 采用“Cube score -> Vector 合并 W/U 与 D -> Cube 合并 W/U 与 DV”的三阶段
> 方案；设计评审通过前不进入 kernel 实现。

## 1. 目标

本算子融合以下两个设备调用，并按固定顺序返回三个结果：

```text
recompute_w_u_fwd(k, v, beta, A, g, cu_seqlens) -> w, u
chunk_bwd_dv_local(q, k, d_o, g, scale, cu_seqlens, chunk_size) -> dv_local
```

Python 接口语义为：

```python
w, u, dv_local = chunk_gdn_bwd_intra(
    q, k, v, g, beta, A, d_o, scale, chunk_size,
    *, cu_seqlens=None, chunk_indices=None, use_exp2=True,
)
```

目标 SoC 为 A5（Ascend 950）。重点性能 shape 为 `B=1`、`T=8192/16384`、
`HK=HV=96`、`K=V=128`、`chunk_size=64`。基线是同一输入上分别调用
`recompute_w_u_fwd` 和 `chunk_bwd_dv_local` 的设备耗时之和。三阶段方案减少阶段边界、
同步和 launch 管理；同一组内按 GVA 子序号分轮，使后一轮 Vector 可以与当前轮 Cube
形成流水。

数学边界如下。令 `BT=chunk_size`，当前 chunk 有效长度为 `M<=BT`，`hk` 为 value
head `hv` 对应的 q/k head。第一版约束 `HV % HK == 0` 且 `G=HV/HK` 属于 `{1,2,3,4}`，
`hk=floor(hv/G)`；目标 shape 为 `G=1`。

```text
Gate(x) = exp2(x), use_exp2=True (default)
          exp(x),  use_exp2=False

Kb_hv[BT,K] = k_hk[BT,K] * (beta_hv[BT,None] * Gate(g_hv[BT,None]))
Vb_hv[BT,V] = v_hv[BT,V] * beta_hv[BT,None]
W_hv[BT,K]  = A_hv[BT,BT] @ Kb_hv[BT,K]
U_hv[BT,V]  = A_hv[BT,BT] @ Vb_hv[BT,V]

S_hk[BT,BT] = k_hk[BT,K] @ q_hk[BT,K]^T
GateDelta_hv[BT,BT] = Gate(g_hv[None,:] - g_hv[:,None])
D_hv[BT,BT] = cast_main(scale * S_hk * GateDelta_hv * CausalValidMask(M)), hk=floor(hv/G)
DV_hv[BT,V] = D_hv[BT,BT] @ d_o_hv[BT,V]
```

上面四个 `@` 均为 Cube 语义；每个操作数的 shape 已写在表达式中，不用逐元素求和式
描述 matmul。`GateDelta`、scale 和 `CausalValidMask` 是 Vector 逐元素操作。
`S` 在 Cube 累加后以 FP32 驻留在 UB，`GateDelta` 也以 FP32 计算；`D` 在写入
workspace 前转换为主 dtype（FP16/BF16）。`Kb/Vb`、`w/u/dv_local` 使用主 dtype。
`g/beta` 可为 BF16 或 FP32，内部统一转 FP32。

## 2. 范围与 shape

### 2.1 支持范围

```text
SoC                 A5 / Ascend 950
layout              仅支持 BNSD [B,H,T,D]
q, k                [B, HK, T, K]，FP16 或 BF16
v, d_o, w, u, dv    [B, HV, T, V]，与 q 同 dtype
g, beta             [B, HV, T]，BF16 或 FP32，不支持 FP16
A                   [B, HV, T, BT]，与 q 同 dtype
K/V                 第一版固定 128/128
BT                  第一版固定 64；chunk_size 属性必须等于 64
```

定长输入使用 `cu_seqlens=None` 且 `chunk_indices=None`，每个 batch 独立按
`ceil(T/BT)` 个 chunk 处理。变长输入必须同时提供 `cu_seqlens` 和 `chunk_indices`：
`cu_seqlens` 为 `[Seq+1]` 的 INT64，`chunk_indices` 为 `[Nchunk,2]` 的 INT64，每行
是 `(sequence_index, local_chunk_index)`。第一版定长支持任意正 `B`；变长 packed 路径
沿用 CPU 标杆约束 `B=1`，由 `cu_seqlens` 描述多个 sequence 的边界。

对变长 pair `(seq, lc)`：

```text
base = cu_seqlens[seq] + lc * BT
M    = min(BT, cu_seqlens[seq+1] - base)
```

物理 token 使用 packed `[0, cu_seqlens[-1])` 偏移。`M=0` 的 sequence 不生成有效展开实例；
尾 chunk 仍分配完整 `[BT,...]` 逻辑 tile，仅前 `M` 行/列有效，GM 写回时使用固定
stride。`chunk_indices` 不能包含超出 sequence 范围的 pair。

### 2.2 符号和任务

```text
Nchunk = B * ceil(T/BT)                         (fixed)
Nchunk = chunk_indices.shape[0]                (varlen)
Nscore = Nchunk * HK                            Stage 0 Score 数量
Ntask  = Nchunk * HV                            Stage 1/2 与 workspace 数量
tau    = chunk_id * HV + hv                     [0, Ntask)
CG       = 4                                     Cube 每轮展开份数
hk_group = floor(hk / CG)                       当前 q/k head 组
group_id = (chunk_id, hk_group)                 组的唯一标识
valid_r  = min(CG, HK - hk_group * CG)          当前组有效份数
r        = hk mod CG                            组内 q/k head 序号
hk_r     = hk_group * CG + r                    当前 Score 对应的 q/k head
j        = 0, 1, ..., G-1                       GVA 子序号
hv_rj    = hk_r * G + j                         当前 Vector/Cube 输出 head
cslot    = r                                    Cube L1 操作槽
owner    = r mod 2                              AIV owner
vslot    = floor(r / 2)                         owner 的 Score/Vector UB 槽
```

Stage 0 的每份操作处理一个 `(chunk,hk_r)`，Stage 1/2 的每份操作处理一个
`(chunk,hv_rj)`。AIC 在 Stage 0 按 `r=0,1,2,3` 依次计算四个 `hk_r` 的 Score；
AIV0 保留 `r=0,2` 的两份 Score，AIV1 保留 `r=1,3` 的两份 Score。随后按 `j`
分轮：每轮 AIV0 计算 `(r=0,j)`、`(r=2,j)`，AIV1 计算 `(r=1,j)`、`(r=3,j)`；
AIC 再按 `r=0,1,2,3` 完成该轮四个 `hv_rj` 的 Stage 2 操作。最后不足四个 `hk`
的尾组，其空余份不发射搬运、计算或 ready。

## 3. 算子接口

### 3.1 Python/OpDef

```text
OpDef: ChunkGdnBwdIntra
Inputs (required): q, k, v, g, beta, A, d_o
Inputs (optional, value-dependent INT64): cu_seqlens, chunk_indices
Attrs: scale(float, required), chunk_size(int, required), use_exp2(bool, default=True)
Outputs: w, u, dv_local
Return order: (w, u, dv_local)
```

`use_exp2` 保留为可选属性且默认 `true`。上游 Triton 对应实现固定使用 exp2、没有同名
参数；本 NPU 接口保留该扩展：`true` 走 exp2，`false` 走自然指数，测试覆盖两种模式。

### 3.2 ACLNN/L0 参数

```cpp
aclnnStatus aclnnChunkGdnBwdIntraGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *dO, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, double scale,
    int64_t chunkSize, bool useExp2, const aclTensor *wOut,
    const aclTensor *uOut, const aclTensor *dvLocalOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkGdnBwdIntra(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

```cpp
namespace l0op {
const std::array<const aclTensor *, 3> ChunkGdnBwdIntra(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *dO, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, double scale,
    int64_t chunkSize, bool useExp2, const aclTensor *wOut,
    const aclTensor *uOut, const aclTensor *dvLocalOut,
    aclOpExecutor *executor);
}
```

Host 校验失败时返回参数错误，不走其它 SoC、其它 `BT/K/V` 或其它 layout fallback。
`scale` 未显式给出时由 wrapper 传入 `1/sqrt(K)`；本算子内部不重新推导。

## 4. Stage 0--2 完整数学语义

本节按一个 `(chunk,hv)` 描述逻辑 tile。`q/k` 通过 `hk=floor(hv/G)` 读取；A、v、g、
beta、d_o 使用当前 `hv`。无效行按 `M` 屏蔽，所有 tile shape 仍保持固定 `BT=64`。

### 4.1 Stage 总览和 DAG

```text
Stage 0 Cube SCORE（原 S1）:
    k,q -> S (score resident in UB[2])

Stage 1 Vector PREP_WU_GATE（合并原 S0/S3）:
    k,v,g,beta,S -> Kb,Vb,D (three independent GM workspace regions)

Stage 2 Cube WU_DV（合并原 S2/S4）:
    A,Kb,Vb,D,d_o -> w,u,dv_local (final outputs)
```

一个完整 `hk` 组的依赖和轮次为：

```text
Stage 0 Cube: S(hk0), S(hk1), S(hk2), S(hk3)

for j in [0,G):
  Stage 1 Vector round j:
    AIV0 -> hv(hk0,j), hv(hk2,j)
    AIV1 -> hv(hk1,j), hv(hk3,j)
  Stage 2 Cube round j:
    hv(hk0,j), hv(hk1,j), hv(hk2,j), hv(hk3,j)
```

Stage 2 不等待 Stage 1 完成全部 `4G` 份操作；第一轮四份 Vector ready 到齐后即可
启动 Stage 2 round 0，AIC 执行当前 Cube round 时，两个 AIV 可以推进 round 1。
Stage 0 和 Stage 2 共用一个 AIC，彼此不能同时执行。每个 AIV 维护两个 UB slot，
其中的 Score 和 k 保留 `G` 轮，轮内其它数据在 MTE3 完成后复用。

`G=4` 时，首个 `hk0..hk3` 组的实际映射为：

```text
round 0: AIV0 -> hv0,hv8    AIV1 -> hv4,hv12    AIC -> hv0,hv4,hv8,hv12
round 1: AIV0 -> hv1,hv9    AIV1 -> hv5,hv13    AIC -> hv1,hv5,hv9,hv13
round 2: AIV0 -> hv2,hv10   AIV1 -> hv6,hv14    AIC -> hv2,hv6,hv10,hv14
round 3: AIV0 -> hv3,hv11   AIV1 -> hv7,hv15    AIC -> hv3,hv7,hv11,hv15
```

### 4.2 Stage 0：Cube，计算局部 score

公式：

```text
S_{hk_r}[BT,BT] = k_{hk_r}[BT,K] @ q_{hk_r}[BT,K]^T
```

AIC 按当前 `hk` 组的 `r=0,1,2,3` 依次调度，每份操作只处理一个 `hk_r`，q/k
均从该 q/k head 读取。Stage 0 不依赖 Stage 1，每份操作完成后将 Score 写入
对应 owner AIV 的 `vslot`，供该 `hk_r` 对应的 `G` 个 value head 共同使用。q/k
在 Cube 路径各只做一次 GM->L1；k 同时被 Stage 1 Vector 使用时，允许另一次独立
GM->UB 读取，并在对应 AIV slot 中保留 `G` 轮。

L1（每个 AIC，KiB；四份 Cube 操作槽）：

```text
r=0 (`hk_0`): L1[0,16)    `k_{hk_0}`[BT,K]；L0A 输入；本份操作末次消费后释放
               L1[16,32)   `q_{hk_0}`[BT,K]；L0B 输入，按列解释；本份操作末次消费后释放
r=1 (`hk_1`): L1[32,48)   `k_{hk_1}`[BT,K]；同上
               L1[48,64)   `q_{hk_1}`[BT,K]；同上
r=2 (`hk_2`): L1[64,80)   `k_{hk_2}`[BT,K]；同上
               L1[80,96)   `q_{hk_2}`[BT,K]；同上
r=3 (`hk_3`): L1[96,112)  `k_{hk_3}`[BT,K]；同上
               L1[112,128) `q_{hk_3}`[BT,K]；同上
```

L0A/L0B 各使用 16 KiB ping/pong；q 以右操作数转置语义送入 L0B，不建立实体 q.T。
L0C 使用 16 KiB FP32 `[BT,BT]` ping/pong。MMAD 只对前 `M x M` 有效，其余元素
清零；Fixpipe 按 `owner` 将 FP32 Score 写入对应 AIV 的 `vslot`：AIV0 的 `hk0/hk2`
分别写入 `UB[64,80)` / `UB[168,184)`，AIV1 的 `hk1/hk3` 使用其本地同名 slot，
随后发布
`score_ready[group_id,r]`。

执行顺序：

1. 对当前组按 `r=0,1,2,3` 依次等待该 Cube 操作槽的 `C0_free[cslot]` 和对应 owner
   AIV `vslot` 的 `score_free[owner,vslot]`。
2. MTE2 将该份操作的 k/q 搬入对应 L1 区间，MTE1 交错装入 L0A/L0B，完成一次
   `k @ q^T` MMAD。
3. Fixpipe 写对应 owner AIV 的 Score UB slot，发布 `score_ready[group_id,r]`；
   在 MTE1/Cube 完成对 q/k 的最后消费后发布 `C0_free[cslot]`。对应 Vector
   路径完成全部 `G` 轮消费后才发布 `score_free[owner,vslot]`，该 Score slot 才能复用。

固定长度、varlen、tail 和 `M=0` 均按第 2 节处理；无效展开实例不发射 MTE、MMAD 或 ready。

### 4.3 Stage 1：Vector，合并 W/U 预处理与 D 门控

对当前 `(hk_r,j)`，令 `hv=hv_rj=hk_r*G+j`，一次 VF 完成以下全部公式：

```text
gate_hv[s]          = Gate(g_hv[s])
bg_hv[s]            = beta_hv[s] * gate_hv[s]
Kb_hv[BT,K]         = k_{hk_r}[BT,K] * bg_hv[:,None]
Vb_hv[BT,V]         = v_hv[BT,V] * beta_hv[:,None]
GateDelta_hv[t,s]   = Gate(g_hv[s] - g_hv[t])          # FP32 [BT,BT]
D_fp32_hv[BT,BT]    = scale * S_{hk_r} * GateDelta_hv * CausalValidMask(M)
D_hv[BT,BT]         = cast_main(D_fp32_hv)              # main dtype [BT,BT]
```

Stage 1 对每个 `r` 分别等待 `score_ready[group_id,r]`，不建立四份 Score 的组级
barrier，并按 `j=0..G-1` 分轮。AIV0 每轮依次处理 `r=0,2`，AIV1 每轮依次处理
`r=1,3`，每个 AIV 使用自己的 `vslot=0,1`。
每个 slot 的 `k_hk_r` 在本路径只从 GM 搬入一次，与 `S_hk_r` 一起保留到最后一轮；
每轮按 `hv_rj` 搬入 v/g/beta。一次 VF 先在 FP32 寄存器中计算 gate/bg，再用同一份
g 计算 `GateDelta`，同时生成 Kb、Vb 和 D，不拆为两个 VF pass。Kb/Vb/D 转换为
主 dtype 后由一次 `V_MTE3` 分别写入 `W_KB/W_VB/W_D`。

UB（每个 AIV，KiB；主 dtype 2 B，score/GateDelta 为 FP32）：

```text
slot 0: UB[0,16)    k_hk[BT,K]；主 dtype 16 KiB；保留 G 轮
        UB[16,32)   v_hv[BT,V]；主 dtype 16 KiB；逐轮覆盖
        UB[32,48)   Kb[BT,K]；主 dtype 16 KiB；写 W_KB 后释放
        UB[48,64)   Vb[BT,V]；主 dtype 16 KiB；写 W_VB 后释放
        UB[64,80)   S_hk[BT,BT]；FP32 16 KiB；Stage 0 写，保留 G 轮
        UB[80,96)   GateDelta[BT,BT]；FP32 16 KiB；VF 中间量
        UB[96,104)  D[BT,BT]；主 dtype 8 KiB；写 W_D 后释放
slot 1: UB[104,120)  k_hk[BT,K]；同上
        UB[120,136)  v_hv[BT,V]；同上
        UB[136,152)  Kb[BT,K]；同上
        UB[152,168)  Vb[BT,V]；同上
        UB[168,184)  S_hk[BT,BT]；FP32 16 KiB；同上
        UB[184,200)  GateDelta[BT,BT]；FP32 16 KiB；同上
        UB[200,208)  D[BT,BT]；主 dtype 8 KiB；同上
FP32: UB[245.5,245.75) g[BT]；slot 0
      UB[245.75,246)    beta[BT]；slot 0
      UB[246,246.25)    g[BT]；slot 1
      UB[246.25,246.5)  beta[BT]；slot 1
```

大型区间按 512 B 对齐，FP32 子槽按 256 B 对齐。双 slot 主 dtype 区占 144 KiB，
score/GateDelta FP32 区占 64 KiB，g/beta 小向量保守占 1 KiB，总峰值 209 KiB；
`UB[208,245.5)` 连续空闲 37.5 KiB，`UB[246.5,248)` 连续空闲 1.5 KiB，
总空闲 39 KiB。

事件闭环：

1. 首轮等待 `score_ready[group_id,r]`；每个 slot 搬入一次 `k_hk_r`。每轮等待
   `V_MTE2[owner,vslot]`，再按 `hv_rj` 搬入 v/g/beta。
2. VF 完成当前 `hv_rj` 的全部 Kb/Vb/D；`V_MTE3[owner,vslot]` 写三个独立
   workspace，发布 `kb_ready[group_id,j,r]`、`vb_ready[group_id,j,r]`、
   `d_ready[group_id,j,r]`。
3. MTE3 读取该 `vslot` 的 Kb/Vb/D 后发布 `MTE3_MTE2[owner,vslot]`，允许下一轮
   覆盖 v/g/beta、Kb/Vb/GateDelta/D；k 和 Score 不覆盖。
4. `j=G-1` 的 VF 完成对 Score 的最后消费后发布 `score_free[owner,vslot]`；最后一轮
   MTE3 完成后释放 k，Stage 0 才能向该 slot 写入下一组 Score。

尾 chunk 的无效行/列在 VF 中置零；`M=0` 不发射 VF、workspace ready 或 free。

### 4.4 Stage 2：Cube，合并 W/U 与 dv_local

对当前 round `j` 的 `hv_rj`，同一个 Cube Stage 内依次完成三个彼此独立的矩阵乘：

```text
w_hv_rj[BT,K]        = A_hv_rj[BT,BT] @ Kb_hv_rj[BT,K]
u_hv_rj[BT,V]        = A_hv_rj[BT,BT] @ Vb_hv_rj[BT,V]
dv_local_hv_rj[BT,V] = D_hv_rj[BT,BT] @ d_o_hv_rj[BT,V]
```

Stage 2 对 round `j` 等待当前组全部 `valid_r` 份
`kb_ready/vb_ready/d_ready[group_id,j,r]`；完整组为四份，尾组只等待有效份。随后
AIC 按有效 `r` 依次处理 `hv_rj=hk_r*G+j`。A、d_o 按 `hv_rj` 读取；Kb 由
`k_hk_r` 生成，Vb/D 也按 `hv_rj` 对应，三者从独立 GM workspace 各搬入一次。
A 只搬入一次并供前两个 GEMM 复用；D/d_o 的 GEMM 与前两个无数据依赖。每份操作内
三次 MMAD 顺序使用独立 L0A/L0B/L0C ping/pong，Fixpipe 分别直接写正式
`wOut/uOut/dvLocalOut`，不复用任何 workspace 地址。round `j` 的 AIC 执行期间，
两个 AIV 可以计算 round `j+1`。

L1（每个 AIC，KiB；四份 Cube 操作槽）：

```text
r=0: L1[128,136)  A_hv_rj[BT,BT]；8 KiB；两次 W/U GEMM 共享
     L1[136,152)  Kb[BT,K]；16 KiB；W_KB GM->L1
     L1[152,168)  Vb[BT,V]；16 KiB；W_VB GM->L1
     L1[168,176)  D[BT,BT]；8 KiB；W_D GM->L1
     L1[176,192)  d_o[BT,V]；16 KiB；输入 GM->L1
r=1: L1[192,200)  A_hv_rj[BT,BT]；8 KiB
     L1[200,216)  Kb[BT,K]；16 KiB
     L1[216,232)  Vb[BT,V]；16 KiB
     L1[232,240)  D[BT,BT]；8 KiB
     L1[240,256)  d_o[BT,V]；16 KiB
r=2: L1[256,264)  A_hv_rj[BT,BT]；8 KiB
     L1[264,280)  Kb[BT,K]；16 KiB
     L1[280,296)  Vb[BT,V]；16 KiB
     L1[296,304)  D[BT,BT]；8 KiB
     L1[304,320)  d_o[BT,V]；16 KiB
r=3: L1[320,328)  A_hv_rj[BT,BT]；8 KiB
     L1[328,344)  Kb[BT,K]；16 KiB
     L1[344,360)  Vb[BT,V]；16 KiB
     L1[360,368)  D[BT,BT]；8 KiB
     L1[368,384)  d_o[BT,V]；16 KiB
```

每个 L0A/L0B 都维护独立 ping/pong。L0C 为 32 KiB FP32 `[BT,128]` accumulator，
按 `w -> u -> dv_local` 的 Fixpipe 完成顺序复用；每次复用前等待上一次 Fixpipe
完成。三个输出均只写前 `M` 行，尾块其余行不参与有效语义。

执行顺序：

1. 对 round `j` 等待 `valid_r` 份 workspace ready，再按有效 `r` 依次等待
   `C2_free[cslot]`；MTE2 按 `tau=chunk_id*HV+hv_rj` 从 `W_KB/W_VB/W_D` 及输入
   GM 搬入该份操作对应的 L1 区间。
2. 以 A 为左操作数依次完成 `A @ Kb`、`A @ Vb`，Fixpipe 写 w/u；A 在两次
    MMAD 间保留，不重复搬运。
3. 以 D 为左操作数完成 `D @ d_o`，Fixpipe 写 dv_local；发布 `wu_done[tau]`、
   `dv_done[tau]` 和 `C2_free[cslot]`。完成四份操作后进入下一 round。

## 5. Stage 资源分配方案

### 5.1 Stage 序列和最小性

```text
Stage 0 Cube   : score = k @ q^T
Stage 1 Vector : Kb/Vb 预处理 + GateDelta + D
Stage 2 Cube   : w = A @ Kb、u = A @ Vb、dv_local = D @ d_o
```

Stage 0/1/2 是 `S_hk -> (Kb_hv,Vb_hv,D_hv) -> (w_hv,u_hv,dv_hv)` 的最短依赖链。
不能把 Stage 0 和 Stage 1 合并，因为它们分别是 Cube 和 Vector，违反 R01；不能把
Stage 1 和 Stage 2 合并，原因相同。Stage 1 已经在一次 VF 中合并 W/U 预处理与 D
门控，Stage 2 已在一个 Cube Stage 中合并三个独立 GEMM，继续拆分只会增加阶段边界。
因而三阶段是满足 R01、R02、R12 的最少 Stage 划分，符合 R19。

这个选择不依赖“算 V 时 UB 不够”。Stage 1 每个 AIV 的双 slot UB 峰值为 209 KiB，低于
248 KiB；Stage 0 四份 Cube 操作槽占 128 KiB，Stage 2 四份 Cube 操作槽占 256 KiB，
两段 L1 区间互不重叠，合计 384 KiB，低于 512 KiB。Stage 0 每组完成四份 `hk`
Score；Stage 1 每个 round 由两个 AIV 各处理两份 `hv_rj`；Stage 2 每个 round 由
AIC 完成四份 `hv_rj`。Stage 2 只等待首轮四份 Vector ready，不等待全部 `G` 轮；
其执行 round `j` 时允许 Stage 1 推进 round `j+1`，避免 Vector 总工作量随 `G`
增加而线性推迟 Stage 2 首次启动。最终流水效果由 A5 profiling 验证。

### 5.2 L1 全空间和生命周期

```text
L1_total = 512 KiB (A5/Ascend 950 kernel 可用上限，待实现环境再次确认)

L1[0,16)       Stage 0 r0 k_hk0
L1[16,32)      Stage 0 r0 q_hk0
L1[32,48)      Stage 0 r1 k_hk1
L1[48,64)      Stage 0 r1 q_hk1
L1[64,80)      Stage 0 r2 k_hk2
L1[80,96)      Stage 0 r2 q_hk2
L1[96,112)     Stage 0 r3 k_hk3
L1[112,128)    Stage 0 r3 q_hk3
L1[128,136)    Stage 2 round j, r0 A_hv_rj
L1[136,152)    Stage 2 round j, r0 Kb_hv_rj
L1[152,168)    Stage 2 round j, r0 Vb_hv_rj
L1[168,176)    Stage 2 round j, r0 D_hv_rj
L1[176,192)    Stage 2 round j, r0 d_o_hv_rj
L1[192,200)    Stage 2 round j, r1 A_hv_rj
L1[200,216)    Stage 2 round j, r1 Kb_hv_rj
L1[216,232)    Stage 2 round j, r1 Vb_hv_rj
L1[232,240)    Stage 2 round j, r1 D_hv_rj
L1[240,256)    Stage 2 round j, r1 d_o_hv_rj
L1[256,264)    Stage 2 round j, r2 A_hv_rj
L1[264,280)    Stage 2 round j, r2 Kb_hv_rj
L1[280,296)    Stage 2 round j, r2 Vb_hv_rj
L1[296,304)    Stage 2 round j, r2 D_hv_rj
L1[304,320)    Stage 2 round j, r2 d_o_hv_rj
L1[320,328)    Stage 2 round j, r3 A_hv_rj
L1[328,344)    Stage 2 round j, r3 Kb_hv_rj
L1[344,360)    Stage 2 round j, r3 Vb_hv_rj
L1[360,368)    Stage 2 round j, r3 D_hv_rj
L1[368,384)    Stage 2 round j, r3 d_o_hv_rj
L1[384,512)    free, 128 KiB contiguous
```

Stage 0 和 Stage 2 的四份操作区完全不重叠；Stage 1 不使用 L1。静态预留峰值为
`128 + 256 = 384 KiB`，总空闲 128 KiB，最大连续空闲 128 KiB。Stage 0 的 q/k
在对应 score Fixpipe 后释放；Stage 2 的 A/Kb/Vb/D/d_o 在对应三次 MMAD 和最终
Fixpipe 后释放。不存在 Cube-to-Cube resident，因此 R07/R09 的跨 Stage 常驻约束
不触发；四份 L1 操作区仍按 R01 为 Cube 的四份展开独立规划。

### 5.3 UB 全空间和生命周期

```text
UB_total = 248 KiB (A5/Ascend 950 kernel 可用上限，待实现环境再次确认)

UB[0,16)       Stage 1 slot0 k_hk, resident G rounds
UB[16,32)      Stage 1 slot0 v_hv, per-round
UB[32,48)      Stage 1 slot0 Kb
UB[48,64)      Stage 1 slot0 Vb
UB[64,80)      Stage 0 -> Stage 1 S_hk slot0, FP32, resident G rounds
UB[80,96)      Stage 1 slot0 GateDelta, FP32
UB[96,104)     Stage 1 slot0 D
UB[104,120)    Stage 1 slot1 k_hk, resident G rounds
UB[120,136)    Stage 1 slot1 v_hv, per-round
UB[136,152)    Stage 1 slot1 Kb
UB[152,168)    Stage 1 slot1 Vb
UB[168,184)    Stage 0 -> Stage 1 S_hk slot1, FP32, resident G rounds
UB[184,200)    Stage 1 slot1 GateDelta, FP32
UB[200,208)    Stage 1 slot1 D
UB[208,245.5)  free, 37.5 KiB contiguous
UB[245.5,245.75) g slot0, FP32
UB[245.75,246)   beta slot0, FP32
UB[246,246.25)   g slot1, FP32
UB[246.25,246.5) beta slot1, FP32
UB[246.5,248)    free, 1.5 KiB contiguous
```

每个 AIV 接收两份 `S_hk`，按 R04 使用两个 UB slot；`S_hk` 和 `k_hk` 在原地址
保留 `G` 轮。Score 和 GateDelta 按 FP32 保存以对齐 CPU 标杆的 gate/mask 计算；
v/g/beta、Kb/Vb/GateDelta/D 在每轮 MTE3 完成后按真实生命周期覆盖。Kb/Vb/D 是
Vector 结果，按 R06 写入独立 GM，不占用跨 Stage UB resident。每个 AIV 的两份
slot 及当前轮全部输入、输出和中间量计入峰值；总峰值 209 KiB，剩余总空闲
39 KiB，最大连续空闲 37.5 KiB。

### 5.4 独立 GM workspace 与搬运

Workspace 只承载三个独立中间结果，不与最终输出复用。令 `E=sizeof(main)=2` bytes，
`stride_kb=BT*K*E`、`stride_vb=BT*V*E`、`stride_d=BT*BT*E`，均按 512 B 对齐：

```text
W_KB_BASE  = 0
W_KB_BYTES = ALIGN512(Ntask * stride_kb)
W_VB_BASE  = W_KB_BASE + W_KB_BYTES
W_VB_BYTES = ALIGN512(Ntask * stride_vb)
W_D_BASE   = W_VB_BASE + W_VB_BYTES
W_D_BYTES  = ALIGN512(Ntask * stride_d)
workspaceSize = W_D_BASE + W_D_BYTES
```

Stage 0 只生成 `Nscore=Nchunk*HK` 份 Score，并全部驻留在 owner AIV 的 UB，不为 Score
分配 GM workspace。三段 workspace 仍按 `Ntask=Nchunk*HV` 分配，因为 Kb/Vb/D
均按 value head 生成。

每个 `hv_rj` 使用 `tau=chunk_id*HV+hv_rj`，偏移为 `base + tau*stride`：

| 区域 | producer | consumer | GM 搬运 | 生命周期 |
|---|---|---|---|---|
| `W_KB` | Stage 1 MTE3 | Stage 2 GM->L1 | 写 1 次、读 1 次 | Stage 1 完成至 Stage 2 末次读；不复用 |
| `W_VB` | Stage 1 MTE3 | Stage 2 GM->L1 | 写 1 次、读 1 次 | Stage 1 完成至 Stage 2 末次读；不复用 |
| `W_D` | Stage 1 MTE3 | Stage 2 GM->L1 | 写 1 次、读 1 次 | Stage 1 完成至 Stage 2 末次读；不复用 |

三段独立 GM 是 R06 的正常 Vector-to-Cube 边界，不是 R14 容量回退；本设计不将
Kb/Vb/D 写入 w/u/dv_local 输出地址，也不声明零额外 workspace。控制同步使用硬件
CrossCore/EventID，不在 tensor workspace 中隐式复用字节。

目标 shape 的精确核算（BF16/FP16 均为 2 B）：

```text
per-task Kb = 64*128*2 = 16 KiB
per-task Vb = 64*128*2 = 16 KiB
per-task D  = 64*64*2  =  8 KiB

B=1,T=8192,HV=96: Nchunk=128, Ntask=12288
  W_KB = 192 MiB, W_VB = 192 MiB, W_D = 96 MiB, total = 480 MiB

B=1,T=16384,HV=96: Nchunk=256, Ntask=24576
  W_KB = 384 MiB, W_VB = 384 MiB, W_D = 192 MiB, total = 960 MiB
```

变长按实际 `chunk_indices.shape[0]` 计算 `Nchunk`，workspace query 不按最大 T 静态
多分配。三段地址互不重叠，满足独立 workspace 约束。

### 5.5 Ping/pong、ready/free 和任务边界

```text
Stage 0 Score（当前 hk 组按 hk0 -> hk1 -> hk2 -> hk3）:
  C_MTE2[cslot] -> L0A/L0B MMAD -> Fixpipe -> score_ready[group_id,r]
  q/k last-consume -> C0_free[cslot]
  Stage 1 final-round VF[owner,vslot] -> score_free[owner,vslot]

Stage 1 Vector round j:
  score_ready[group_id,r] + V_MTE2[owner,vslot]
  -> one VF(hv_rj) -> V_MTE3[owner,vslot]
  -> kb_ready/vb_ready/d_ready[group_id,j,r] -> MTE3_MTE2[owner,vslot]

Stage 2 Cube round j（按 r0 -> r1 -> r2 -> r3）:
  ready[group_id,j,r] -> C_MTE2[cslot] -> (A@Kb, A@Vb, D@d_o)
  -> Fixpipe final outputs -> C2_free[cslot]
```

Stage 0 覆盖 Score slot 前等待对应 `score_free[owner,vslot]`；Stage 1 在 `G` 轮内
不覆盖 k/Score，只在 `MTE3_MTE2[owner,vslot]` 后覆盖当前轮的 v/g/beta、Kb/Vb/D；
Stage 2 覆盖对应 L1 操作槽前等待 `C2_free[cslot]`。完整组的 round 0 四份 Vector
ready 到齐即可启动 Stage 2；尾组只等待 `valid_r` 份。round `j` 的 Stage 2 与
round `j+1` 的 Stage 1 可并行。
ready 只对有效 `hk_r/hv_rj` 发布，`M=0` 不补发空 ready/free。每个 `hv_rj` 的
workspace 地址在本 invocation 内不复用为其它语义。

### 5.6 R01--R19 / R02-A 检查表

| 规则 | 结论 | 证据 |
|---|---|---|
| R01 | 满足 | Stage 0 四份为 `hk0..hk3`；Stage 2 每轮四份为四个 `hv_rj`；Stage 1 每轮由 AIV0 处理 `r0,r2`、AIV1 处理 `r1,r3` |
| R02 | 满足 | Cube 输入来自输入或前序 ready，见 4.2--4.4 |
| R02-A | 满足 | 全部矩阵乘使用 `@` 并注明 shape、转置、FP32 累加和 Fixpipe |
| R03 | 满足 | A5 L1=512 KiB、UB=248 KiB，见 5.2/5.3 |
| R04 | 满足 | 每个 AIV 固定保留两份 `S_hk` UB slot，并分别供 G 轮 Vector 消费 |
| R05 | 不触发 | 没有需要跨多个 Vector Stage 保留的 Vector 中间结果 |
| R06 | 满足 | Kb/Vb/D 均经独立 GM workspace 后由 Stage 2 读取 |
| R07 | 不触发 | 没有 Cube 中间结果被后续 Cube 作为 resident 读取 |
| R08 | 满足 | Stage 1 两个 AIV 各自的两份 slot 及全部输入、输出、中间量计入 209 KiB 峰值 |
| R09 | 满足 | 当前无跨 Stage Cube resident；四份 Cube 操作区独立规划，若后续产生常驻数据则按 R09 预留四份 |
| R10 | 满足 | q/k 的 Cube 路径各搬一次；k 的 Vector 路径按 hk 搬一次并复用 G 轮；v/g/beta 按 hv 各搬一次 |
| R11 | 满足 | score_free、MTE3_MTE2、C0_free/C2_free 闭环后才覆盖地址 |
| R12 | 满足 | 每个 `hv_rj` 一次搬齐所需的 hv 输入，一次 VF 完成 Kb/Vb/D；共享 k 按 hk 首轮搬入并保留 |
| R13 | 满足 | Stage 0/2 按 hk 组、GVA round 和四份操作区调度时 L1 不重叠，Stage 1 只用 UB |
| R14 | 不触发 | 三段 workspace 是 R06 正常边界，不是容量回退 |
| R15 | 满足 | 原 W/U 与 D 已合并为一个 Vector Stage，三次 Cube GEMM 已合并为一个 Cube Stage |
| R16 | 不触发 | 没有跨 Vector Stage 的 Vector 结果；g/beta 只在 Stage 1 使用 |
| R17 | 满足 | L1/UB 绝对区间、对齐、生命周期、总空闲和最大连续空闲已列出 |
| R18 | 满足 | Stage 0 按 `hk_r` 生产一次 Score；Stage 1/2 统一使用 `hv_rj=hk_r*G+j`，一个 Score 恰好服务 G 个 value head |
| R19 | 满足 | 三阶段已是满足类型和一次 VF 约束的最少划分 |

### 5.7 验证与性能计划

设计通过后按 `docs/agents/04-operator-development.md` 和 `05-operator-testing.md`
进入实现。验证顺序固定为：

1. 以用户 CPU 标杆复现 `Gate`、Kb/Vb、W/U、S/D/dv 语义，覆盖默认 exp2、自然指数、
   固定/变长、`M=1/32/33/64`、G=1/2/4、空 sequence 与尾 chunk。
2. 验证 ACLNN/ATB 接口、workspace size、BNSD shape/dtype、`chunk_indices` 和非法参数。
3. 增加固定长度 `G=4` 调度验证用例：`B=1,T=256,HK=24,HV=96,K=V=128,BT=64`、
   BF16、`use_exp2=True`。该用例检查每个 chunk 的 Stage 0 只计算 `HK=24` 份 Score，
   每份 Score 恰好服务 4 个 value head；Stage 1/2 各产生 `HV=96` 份结果，并逐项与
   CPU 标杆比较 w/u/dv_local。同步检查 `j=0..3` 的 ready 次序、workspace offset、
   Score 末次消费后释放，以及 round 0 后 Stage 2 已启动而无需等待全部四轮 Vector。
   此时 `Nchunk=4`、`Nscore=96`、`Ntask=384`，三段 workspace 分别为 6 MiB、6 MiB、
   3 MiB，总计 15 MiB。
4. 在 A5 对 `T=8192/16384, HK=HV=96, K=V=128` 记录三阶段融合耗时，以及两个基线
   小算子设备耗时之和；同时观察 Stage 0/1/2 的跨 round overlap。
5. 重点 profiling Stage 1 单次 VF 的寄存器压力、GateDelta 计算和三段 workspace 带宽，
   以及 Stage 2 三次 MMAD 的 L0C 复用等待。若精度或性能要求需要改变分组，必须回写
   本文并重新完成整体方案评审。

"""
ChunkFwdO msprof op simulator 调用脚本。

用例: B=1, HK=1, HV=1, T=128, K=128, V=128, chunk_size=128, dtype=bf16
入口: fla_npu.ops.ascendc.chunk_fwd_o (解耦 ctypes 直调 aclnn, 不依赖 torch.ops.npu)

仿真模式下不需要真实 NPU 设备，msopprof 会启动 simulator 后端执行 kernel。
脚本内不做 .npu() 迁移（simulator 会接管 aclnn 调用），仅构造 CPU tensor
作为入参，实际算子在 simulator 中运行。
"""
import math
import sys
import torch
import torch_npu  # noqa: F401  simulator 需要 torch_npu runtime 初始化
import fla_npu
from fla_npu.ops.ascendc import chunk_fwd_o


def main():
    # 用例参数
    B, HK, HV, T, K, V = 1, 1, 1, 128, 128, 128
    chunk_size = 128
    scale = 1.0 / math.sqrt(K)
    dtype = torch.bfloat16
    g_dtype = torch.bfloat16

    num_chunks = (T + chunk_size - 1) // chunk_size  # 1

    print(f"[case] B={B} HK={HK} HV={HV} T={T} K={K} V={V} "
          f"chunk_size={chunk_size} num_chunks={num_chunks} scale={scale:.6f} "
          f"dtype={dtype} g_dtype={g_dtype}", flush=True)

    # 设置 NPU device (simulator 模式下由 msopprof 注入 simulator backend)
    torch.npu.set_device(0)
    device = "npu:0"

    # 构造输入并迁移到 NPU (simulator 接管)
    q = torch.randn(B, HK, T, K, dtype=dtype).npu()
    k = torch.randn(B, HK, T, K, dtype=dtype).npu()
    v = torch.randn(B, HV, T, V, dtype=dtype).npu()
    h = torch.randn(B, HV, num_chunks, K, V, dtype=dtype).npu()
    g = torch.randn(B, HV, T, dtype=g_dtype).npu()

    print("[step] inputs prepared", flush=True)
    print(f"  q: {q.shape} {q.dtype} {q.device}", flush=True)
    print(f"  k: {k.shape} {k.dtype}", flush=True)
    print(f"  v: {v.shape} {v.dtype}", flush=True)
    print(f"  h: {h.shape} {h.dtype}", flush=True)
    print(f"  g: {g.shape} {g.dtype}", flush=True)

    # 调用算子 (定长模式: cu_seqlens=None, chunk_indices=None)
    print("[step] before chunk_fwd_o", flush=True)
    o = chunk_fwd_o(
        q, k, v, h, scale,
        g=g,
        g_gamma=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    print("[step] after chunk_fwd_o", flush=True)
    print(f"  o: {o.shape} {o.dtype}", flush=True)
    assert o.shape == v.shape, f"output shape mismatch: {o.shape} vs {v.shape}"
    print("[step] done, output shape ok", flush=True)


if __name__ == "__main__":
    main()

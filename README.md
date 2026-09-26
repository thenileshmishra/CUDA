## Roadmap (Interview-Focused Study Order)

> Trimmed to what actually matters for a CUDA/GPU role. Follow top to bottom.

### Phase 1 — Setup
`01_Setup/`
- Install CUDA Toolkit, confirm `nvidia-smi` and `nvcc --version` agree on version.
- **Goal:** working, correctly-versioned environment. Nothing else works without this.

### Phase 2 — C/C++ Refresher
`02_C_and_C++_Review/`
- Focus: `01 Pointers`, `02 Custom Types`, `03 Type Casting`.
- Skim: `04 Makefiles` (just enough to compile multi-file projects).
- **Goal:** pointer arithmetic and struct/cast reasoning fluent, not just readable. Most beginner "CUDA bugs" are pointer bugs in disguise.

### Phase 3 — Why GPUs
`03_Gentle_Intro_to_GPUs/`
- CPU vs GPU vs TPU: latency-optimized vs throughput-optimized hardware.
- Host/device terminology, why massively parallel hardware fits deep learning.
- **Goal:** explain the CPU-vs-GPU tradeoff out loud, unscripted. This is the standard interview opener.

### Phase 4 — First Kernels (core phase #1)
`04_Writing_your_First_Kernels/`
1. **CUDA Basics** — thread/block/grid indexing math (1D/2D/3D). Derive the global-index formula from memory.
2. **Kernels** — vector addition, naive matrix multiplication. Write both cold, verify against a CPU reference.
3. **Profiling** — `nsys` for system-level bottlenecks, `ncu` for per-kernel occupancy/memory throughput. Profile everything from here on.
4. **Atomics** — `atomicAdd`-based histogram or reduction; understand race conditions and contention cost.
5. **Streams** — pinned memory, `cudaMemcpyAsync`, events, overlapping compute with data transfer.
- **Goal:** comfortable writing, launching, profiling, and debugging a kernel end to end.

### Phase 5 — Matmul Optimization (core phase #2, highest priority)
`05_Faster_Matmul/`
- Progression: naive → coalesced access → shared-memory tiling → 1D/2D block-tiling → vectorized (128-bit) loads → autotuning.
- Implement every stage yourself, keep a GFLOPS table, know *why* each step helped.
- **Goal:** this is the closest thing to a standard NVIDIA-style take-home. Budget the most time here.

### Phase 6 — CUDA Libraries
`06_CUDA_APIs/`
- **cuBLAS** — call `cublasSgemm`, benchmark against your own Phase 5 kernel.
- **cuDNN** — run the Tanh/Conv2d examples once, understand the opaque-handle API pattern.
- **Goal:** know when to reach for a library vs. hand-roll a kernel.

### Phase 7 — Final Project
`07_Final_Project/`
- Build the MLP MNIST trainer in raw CUDA: forward pass → verify vs NumPy/PyTorch → backward pass → training loop.
- **Goal:** one complete, correct, benchmarked, framework-free neural net — your proof-of-work for interviews.

---

### Time Allocation

| Phase | Weight |
|---|---|
| Matmul Optimization (05) | 40% |
| First Kernels (04) | 35% |
| Setup + C/C++ + GPU Intro (01–03) | 10% |
| CUDA Libraries (06) | 10% |
| Final Project (07) | 5% |

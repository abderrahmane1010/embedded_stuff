# MLIR — Multi-Level Intermediate Representation
## A Comprehensive Technical Overview

---

## Table of Contents

1. [What is MLIR?](#what-is-mlir)
2. [Motivation & Context](#motivation--context)
3. [High-Level Architecture](#high-level-architecture)
4. [Core Abstractions](#core-abstractions)
   - [Dialects](#dialects)
   - [Operations (Ops)](#operations-ops)
   - [Types](#types)
   - [Attributes](#attributes)
   - [Regions & Blocks](#regions--blocks)
5. [IR Structure In Depth](#ir-structure-in-depth)
6. [The Dialect Ecosystem](#the-dialect-ecosystem)
7. [Transformations & Passes](#transformations--passes)
8. [Lowering & Conversion Pipeline](#lowering--conversion-pipeline)
9. [MLIR in the ML/DL Accelerator Landscape](#mlir-in-the-mldl-accelerator-landscape)
10. [Key Design Principles](#key-design-principles)
11. [Comparison with Other IRs](#comparison-with-other-irs)
12. [Toolchain Integration](#toolchain-integration)
13. [Limitations & Open Challenges](#limitations--open-challenges)

---

## What is MLIR?

**MLIR** (Multi-Level Intermediate Representation) is a **compiler infrastructure framework** originally developed at Google, now part of the **LLVM project**. It was publicly introduced in 2019 and open-sourced under the LLVM umbrella.

Its core proposition is deceptively simple:

> _Instead of building yet another monolithic IR, give compiler authors a framework to define their own IRs — and compose them together._

MLIR is **not** a compiler. It is a **meta-IR** and an **infrastructure** for building compilers, transpilers, and program analysis tools — particularly well-suited to the heterogeneous hardware reality of modern ML/DL workloads.

```
Framework Lineage:
  LLVM IR  ──►  designed for CPUs, one level of abstraction
  MLIR     ──►  designed for anything, infinite levels of abstraction
```

---

## Motivation & Context

Before MLIR, every ML framework invented its own IR:

| Framework    | IR / Graph Format        | Problem                                      |
|--------------|--------------------------|----------------------------------------------|
| TensorFlow   | TF Graph, XLA HLO        | Fragmented, hard to extend                   |
| PyTorch      | TorchScript, FX Graph    | Python-centric, limited static analysis      |
| ONNX         | ONNX protobuf            | Interchange only, no transformation infra    |
| Halide       | Halide IR                | Domain-specific, scheduling-focused          |
| TVM          | Relay, TIR               | Two-level, TVM-specific                      |
| LLVM         | LLVM IR                  | CPU/GPU only, too low-level for tensors      |

Each ecosystem built its own passes, pattern matching engines, type systems, and code generators — **reinventing the wheel** at each layer.

The result: a combinatorial explosion of `N frameworks × M backends × P optimization levels` that is impossible to maintain.

**MLIR's answer:** define a common infrastructure where each "level" of abstraction is a first-class **dialect**, and provide reusable machinery for transformations, pattern rewriting, and progressive lowering.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Frameworks                          │
│         PyTorch / TensorFlow / JAX / ONNX / custom DSL         │
└──────────────────────┬──────────────────────────────────────────┘
                       │  import / export
┌──────────────────────▼──────────────────────────────────────────┐
│                     High-Level Dialects                         │
│      torch-mlir │ stablehlo │ tosa │ linalg (named ops)        │
└──────────────────────┬──────────────────────────────────────────┘
                       │  dialect conversion passes
┌──────────────────────▼──────────────────────────────────────────┐
│                    Mid-Level Dialects                           │
│        linalg (generic) │ vector │ tensor │ bufferization       │
└──────────────────────┬──────────────────────────────────────────┘
                       │  lowering passes
┌──────────────────────▼──────────────────────────────────────────┐
│                    Low-Level Dialects                           │
│         memref │ scf (loops) │ cf (control flow) │ arith       │
└──────────────────────┬──────────────────────────────────────────┘
                       │  target-specific lowering
┌──────────────────────▼──────────────────────────────────────────┐
│                    Hardware Backends                            │
│    LLVM IR → CPU │ NVPTX → CUDA │ SPIRV → Vulkan │ custom NPU │
└─────────────────────────────────────────────────────────────────┘
```

This is the **progressive lowering** model: a computation is expressed at a high level of abstraction and stepwise refined toward concrete machine instructions, with **each step being a well-defined dialect-to-dialect conversion**.

---

## Core Abstractions

### Dialects

A **dialect** is MLIR's fundamental extensibility mechanism. It is a namespace that groups together:

- A set of **Operations** (Ops)
- A set of **Types**
- A set of **Attributes**
- Optional **interfaces** and **traits**

```
mlir::Dialect
 ├── name (e.g., "linalg", "arith", "tosa", "gpu")
 ├── Operations  (the verbs)
 ├── Types       (the nouns)
 └── Attributes  (the metadata)
```

Dialects can **co-exist in the same IR file**. An MLIR program can contain ops from `linalg`, `arith`, and `memref` at the same time. This is radically different from traditional compilers where IR is monolithic.

**Built-in dialects (LLVM project):**

| Dialect     | Purpose                                              |
|-------------|------------------------------------------------------|
| `builtin`   | Module, function, basic types (i32, f32, ...)        |
| `arith`     | Arithmetic ops (addi, mulf, cmpi, ...)               |
| `math`      | Math ops (exp, sqrt, sin, ...)                       |
| `memref`    | Memory reference abstractions (buffer semantics)     |
| `tensor`    | Immutable tensor abstractions (value semantics)      |
| `linalg`    | Named + generic linear algebra ops                   |
| `vector`    | SIMD/vector operations                               |
| `scf`       | Structured control flow (for, if, while)             |
| `cf`        | Unstructured control flow (br, cond_br)              |
| `func`      | Function definitions and calls                       |
| `llvm`      | LLVM IR interop                                      |
| `gpu`       | GPU abstractions (launch, threads, barriers)         |
| `spirv`     | SPIR-V code generation (Vulkan/OpenCL)               |
| `nvgpu`     | NVIDIA-specific GPU ops                              |

---

### Operations (Ops)

An **Operation** is the universal unit of computation in MLIR. Every instruction, function, module — everything — is an `Operation`.

An Op consists of:

```
%result : !type = "dialect.op_name"(%operand1, %operand2) 
                  { attr_key = attr_value }
                  (successor_blocks) 
                  : (operand_types) -> result_types
                  {  regions  }
```

Key properties:

- **SSA form** — all values are in Static Single Assignment form
- **Multiple results** — an op can return 0, 1, or N values
- **Nested regions** — ops can contain regions (enabling functions, loops, conditionals as ops)
- **Verifier** — each op defines a `verify()` method; MLIR enforces invariants at all times
- **Custom syntax** — each dialect can define its own human-readable textual format

**Example — a matrix multiply in linalg dialect:**
```mlir
%C = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x4xf32>)
                   outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
```

**Example — same after lowering to linalg generic:**
```mlir
linalg.generic {
    indexing_maps = [affine_map<(m,n,k) -> (m,k)>,
                     affine_map<(m,n,k) -> (k,n)>,
                     affine_map<(m,n,k) -> (m,n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  }
  ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>)
  outs(%C : memref<4x4xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %add = arith.addf %c, %mul : f32
      linalg.yield %add : f32
  }
```

The `linalg.generic` form encodes the **iteration space** (via affine maps) and **computation** (via a body region) separately — making it amenable to tiling, vectorization, and parallelization.

---

### Types

MLIR has a rich, extensible type system. Types are **first-class** and can be user-defined per dialect.

**Built-in types:**

```
Integer types:    i1, i8, i16, i32, i64, iN (arbitrary width)
Float types:      f16, bf16, f32, f64, f80, f128, tf32
Index type:       index (platform-native integer for memory indexing)
Complex:          complex<f32>
Tuple:            tuple<i32, f32>
None:             none

Tensor:           tensor<4x8xf32>        (static, value semantics)
                  tensor<?x8xf32>        (dynamic dimension)
                  tensor<*xf32>          (fully dynamic rank)

MemRef:           memref<4x8xf32>        (buffer, pointer semantics)
                  memref<4x8xf32, 3>     (with memory space = 3)
                  memref<?x8xf32, strided<[8, 1], offset: ?>>

Vector:           vector<4xf32>          (fixed-size SIMD)
                  vector<4x8xf32>        (multi-dimensional)
```

**Custom dialect types** (examples):

```
!quant.uniform<i8:f32, 0.1:-128>   (quantization dialect)
!llvm.ptr<i32>                      (LLVM pointer type)
!gpu.async.token                    (GPU async dialect)
```

---

### Attributes

**Attributes** are compile-time constant metadata attached to ops or stored as op operands. They are **immutable** and **uniqued** (interned).

```mlir
// Integer attribute
arith.constant 42 : i32

// Dense tensor attribute (constant tensor)
arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>

// Affine map attribute
affine_map<(d0, d1) -> (d1, d0)>

// String attribute
{dialect.key = "some string value"}

// Array attribute
{padding = [1, 1, 2, 2] : array<i64>}
```

Attributes are crucial for encoding **configuration** (padding, strides, data layout, quantization parameters) without requiring those to be SSA values.

---

### Regions & Blocks

**Regions** and **Blocks** enable ops to contain nested code:

```
Operation
└── Region (0..N regions per op)
    └── Block (1..N blocks per region, in SSA form)
        ├── Block Arguments (the "phi nodes" of MLIR)
        └── Operations (can themselves have regions)
```

- A **Region** is a control-flow graph (CFG) or a simple sequence of ops
- A **Block** starts with optional block arguments (replacing φ-nodes)
- Nested regions allow expressing **functions**, **loops**, **conditionals**, **parallel sections** — all as ops

```mlir
func.func @relu(%x: tensor<4xf32>) -> tensor<4xf32> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %result = arith.maximumf %x, %zero : tensor<4xf32>
  return %result : tensor<4xf32>
}
```

Here `func.func` is an op containing one region, itself containing one block.

---

## IR Structure In Depth

Every MLIR program is rooted in a **ModuleOp** (a builtin op):

```
ModuleOp
├── FuncOp ("main")
│   ├── Block (entry block, args = func params)
│   │   ├── Op A (e.g., linalg.matmul)
│   │   ├── Op B (e.g., arith.addf)
│   │   └── func.return
│   └── Block (optional successor blocks for CFG)
├── FuncOp ("helper")
│   └── ...
└── GlobalOp (optional globals/constants)
```

**SSA invariant**: every `%value` is defined exactly once and dominates all its uses. MLIR enforces this via a built-in verifier.

---

## The Dialect Ecosystem

Beyond the LLVM built-in dialects, the ecosystem has grown into a rich collection:

### ML / DL Frontend Dialects

| Dialect        | Origin         | Purpose                                               |
|----------------|----------------|-------------------------------------------------------|
| `stablehlo`    | Google/OpenXLA | Stable version of XLA HLO for TF/JAX portability     |
| `chlo`         | Google         | Client HLO — broadcasts, complex math ops             |
| `tosa`         | ARM/MLCommons  | Tensor Operator Set Architecture — hardware-neutral   |
| `torch`        | torch-mlir     | PyTorch FX graph import                               |
| `onnx`         | ONNX-MLIR      | ONNX model import                                     |
| `mhlo`         | Google         | Meta HLO (predecessor to stablehlo)                  |

### Optimization / Mid-Level Dialects

| Dialect        | Purpose                                               |
|----------------|-------------------------------------------------------|
| `linalg`       | Named/generic structured linear algebra (tiling, fusion) |
| `tensor`       | High-level tensor transformations (reshape, extract)  |
| `vector`       | SIMD vectorization abstractions                       |
| `affine`       | Polyhedral loop transformations (affine maps, bounds) |
| `bufferization`| Tensor-to-memref conversion (one-shot bufferizer)     |

### Hardware-Targeting Dialects

| Dialect        | Target                                                |
|----------------|-------------------------------------------------------|
| `llvm`         | LLVM IR → CPU (x86, ARM, RISC-V, ...)                |
| `nvgpu`        | NVIDIA GPU specifics (warpgroup MMA, ...)             |
| `rocdl`        | AMD ROCm / HIP                                        |
| `spirv`        | Vulkan / OpenCL SPIR-V                                |
| `amdgpu`       | AMD GPU dialects                                      |
| `arm_neon`     | ARM NEON intrinsics                                   |
| `arm_sme`      | ARM Scalable Matrix Extension                         |
| `x86vector`    | x86 AVX intrinsics                                    |
| custom NPU     | Vendor-defined (e.g., IREE for embedded, MLIR-AIE for AMD Versal) |

---

## Transformations & Passes

MLIR provides a **Pass Manager** that orchestrates transformations.

### Pass Types

```
Pass
├── OperationPass<OpT>     — scoped to a specific op type (e.g., FuncOp)
├── InterfacePass<IfaceT>  — scoped to ops implementing a given interface
└── ModulePass             — whole-module scope (use sparingly)
```

### Pass Manager Structure

```
PassManager (on ModuleOp)
├── Pass A (on ModuleOp)
├── OpPassManager (on FuncOp)
│   ├── Pass B (on FuncOp)
│   ├── Pass C (on FuncOp)
│   └── Pass D (on FuncOp)
└── Pass E (on ModuleOp)
```

### Pattern Rewriting

The workhorse of MLIR transformations is **declarative pattern rewriting** via `RewritePatterns`:

```cpp
struct MyPattern : OpRewritePattern<linalg::MatmulOp> {
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // Check preconditions
    if (!isSmallMatrix(op)) return failure();
    // Perform rewrite
    rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(op, ...);
    return success();
  }
};
```

Patterns declare a **benefit** (priority), and are applied by a **GreedyPatternRewriteDriver** or a **Dialect Conversion** framework.

### Common Transformation Passes

| Pass Category       | Examples                                              |
|---------------------|-------------------------------------------------------|
| Canonicalization    | Fold constants, remove dead ops, normalize IR         |
| Tiling              | Split loops into tiles for cache/parallelism          |
| Vectorization       | Map linalg ops to vector ops                          |
| Fusion              | Fuse producer-consumer ops to reduce memory traffic   |
| Bufferization       | Convert tensor (value) ops to memref (buffer) ops     |
| Lowering            | Convert high-level dialect ops to lower-level ones    |
| Inlining            | Inline function calls                                 |
| Loop optimization   | Unrolling, jamming, loop-invariant code motion        |

---

## Lowering & Conversion Pipeline

Lowering in MLIR is **progressive**: you go step by step from abstract to concrete.

### Example: MatMul from PyTorch to LLVM IR

```
torch.aten.mm (torch dialect)
        │
        ▼  [torch-to-linalg]
linalg.matmul (linalg dialect)
        │
        ▼  [linalg-to-loops + tile]
linalg.generic → scf.for + arith ops
        │
        ▼  [one-shot-bufferize]
memref ops + scf.for + arith
        │
        ▼  [convert-to-llvm]
llvm dialect ops
        │
        ▼  [LLVM backend]
x86 / ARM / RISC-V machine code
```

Each arrow is a **dialect conversion pass** — a structured transformation that rewrites ops from one dialect to another, guided by **conversion patterns** and **type converters**.

### Partial vs Full Conversion

- **Partial conversion**: some ops remain unconverted (useful for incremental lowering)
- **Full conversion**: all ops must be converted to target dialect(s); verification fails otherwise
- **Legalization**: the conversion framework checks each op against a **LegalityCallback** to decide what needs conversion

---

## MLIR in the ML/DL Accelerator Landscape

MLIR is the critical **software bridge** between ML frameworks and custom silicon. Here's how it fits:

```
┌─────────────┐    ┌──────────────┐    ┌────────────────────────────┐
│  PyTorch /  │    │  torch-mlir  │    │  MLIR optimization passes  │
│  JAX / TF   │───►│  stablehlo   │───►│  (tiling, fusion, quant)   │
└─────────────┘    └──────────────┘    └────────────┬───────────────┘
                                                     │
                          ┌──────────────────────────┼──────────────────┐
                          │                          │                  │
                    ┌─────▼──────┐           ┌───────▼───────┐  ┌──────▼──────┐
                    │  LLVM IR   │           │  SPIR-V/Vulkan│  │  NPU custom │
                    │  (CPU/GPU) │           │    (Khronos)  │  │   dialect   │
                    └─────┬──────┘           └───────────────┘  └──────┬──────┘
                          │                                            │
                    ┌─────▼──────┐                              ┌──────▼──────┐
                    │ x86/ARM    │                              │ NPU binary  │
                    │  binary    │                              │ (vendor SDK)│
                    └────────────┘                              └─────────────┘
```

### Key projects using MLIR for hardware targeting:

| Project         | Target                        | Description                                        |
|-----------------|-------------------------------|-----------------------------------------------------|
| **IREE**        | CPU, GPU, embedded NPUs       | Google's end-to-end ML execution runtime via MLIR  |
| **MLIR-AIE**    | AMD Versal / AI Engine        | Spatial dataflow programming for AMD NPUs          |
| **Triton**      | NVIDIA/AMD GPU                | High-performance GPU kernels via MLIR backend      |
| **CIRCT**       | FPGAs / RTL                   | Circuit IR Compilers and Tools (HW design via MLIR)|
| **TOSA**        | Any NPU                       | ARM-led standard operator set, MLIR-first          |
| **torch-mlir**  | All backends                  | PyTorch → MLIR bridge                              |
| **OpenXLA**     | TPU / GPU                     | Google's XLA reimplementation on MLIR              |
| **Buddy MLIR**  | RISC-V                        | RISC-V vector & matrix acceleration via MLIR       |

The key value for NPU vendors: they can **define their own MLIR dialect** matching their HW ISA, then plug into the existing frontend/optimization pipeline. They only need to write the final lowering pass, not an entire compiler stack.

---

## Key Design Principles

### 1. Generality via Extensibility
No fixed set of operations or types. Every dialect is a first-class extension. The MLIR core is a **framework**, not a language.

### 2. Progressive Lowering
No single big-bang translation. IRs are refined step by step, with each step being verifiable and inspectable.

### 3. Reusability of Infrastructure
Pattern rewriting, pass management, type inference, diagnostics — all defined once in the core and shared across all dialects.

### 4. Verifiability at All Levels
Every dialect defines **verifiers**. The IR is structurally sound at every point in the compilation pipeline — not just at the beginning and end.

### 5. Location Tracking
Every op carries **location information** (source file, line, column). Errors and diagnostics trace back through multiple lowering steps — critical for debugging complex compilation pipelines.

### 6. Declarative Op Definition (ODS / TableGen)
Operations can be defined declaratively in **TableGen** (ODS — Op Definition Spec), which auto-generates C++ boilerplate (builders, accessors, verifiers, printers, parsers):

```tablegen
def Linalg_MatmulOp : LinalgNamedStructuredOp<"matmul", ...> {
  let summary = "Matrix multiplication";
  let arguments = (ins AnyRankedTensor:$A, AnyRankedTensor:$B,
                       AnyRankedTensor:$C);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}
```

### 7. Polyhedral-Friendly (Affine Dialect)
The `affine` dialect encodes loop bounds and access patterns as **affine expressions**, enabling classical polyhedral transformations (tiling, interchange, skewing) as first-class passes.

---

## Comparison with Other IRs

| Feature                  | LLVM IR         | XLA HLO         | TVM Relay/TIR    | MLIR                   |
|--------------------------|-----------------|-----------------|------------------|------------------------|
| **Abstraction levels**   | 1 (low)         | 1 (tensor)      | 2 (graph + loop) | N (extensible)         |
| **Extensibility**        | Limited         | None            | Limited          | Full (dialects)        |
| **Mixed-level IR**       | ✗               | ✗               | Partial          | ✅                      |
| **Reusable passes**      | Partial         | ✗               | Partial          | ✅                      |
| **NPU targeting**        | Indirect        | ✗               | Partial          | ✅ (custom dialect)     |
| **Polyhedral support**   | (Polly, extern) | Partial         | (manual)         | ✅ (affine dialect)     |
| **Framework agnostic**   | ✅               | ✗ (XLA only)    | ✗ (TVM only)     | ✅                      |
| **Open governance**      | LLVM            | Google          | Apache / OctoML  | LLVM                   |

---

## Toolchain Integration

### Building with MLIR

```bash
# MLIR is part of the LLVM monorepo
git clone https://github.com/llvm/llvm-project
cmake -S llvm -B build \
      -DLLVM_ENABLE_PROJECTS="mlir" \
      -DLLVM_TARGETS_TO_BUILD="X86;AArch64;RISCV;NVPTX;AMDGPU"
cmake --build build --target mlir-opt mlir-translate mlir-cpu-runner
```

### Key CLI Tools

| Tool              | Purpose                                                |
|-------------------|--------------------------------------------------------|
| `mlir-opt`        | Run passes on `.mlir` files — primary development tool |
| `mlir-translate`  | Translate MLIR → LLVM IR, SPIR-V, etc.                |
| `mlir-cpu-runner` | JIT-execute MLIR on CPU (for testing)                 |
| `mlir-lsp-server` | Language Server Protocol — IDE support for `.mlir`    |
| `mlir-pdll`       | PDLL — Pattern Description Language for rewrite rules |
| `tblgen`          | TableGen code generation (ODS → C++)                  |

### Python Bindings

MLIR exposes Python bindings via `mlir.core`, enabling IR construction and manipulation from Python:

```python
from mlir.ir import Context, Module
from mlir.dialects import arith, func

with Context():
    module = Module.parse("""
        func.func @add(%a: i32, %b: i32) -> i32 {
            %c = arith.addi %a, %b : i32
            return %c : i32
        }
    """)
```

---

## Limitations & Open Challenges

### 1. Steep Learning Curve
MLIR's generality comes at the cost of conceptual complexity. Understanding dialects, conversion frameworks, and the pass pipeline requires significant investment. The C++ API is verbose.

### 2. Fragmented Ecosystem
Despite being a unifying infrastructure, different projects (IREE, torch-mlir, Triton, OpenXLA) still make diverging choices about which dialects to use and how to connect them. There is no single canonical pipeline.

### 3. Compilation Speed
The pass pipeline infrastructure adds overhead. For large models with complex lowering pipelines, compile times can be significant — though this is improving.

### 4. Dynamic Shape Handling
Static shapes are well-supported. Dynamic shapes (unknown at compile time) require `?` dimensions and dynamic shape propagation, which adds complexity throughout the pipeline. Recent efforts (dynamic shapes in stablehlo, shape dialect) are ongoing.

### 5. Debugging Complex Pipelines
When a lowering pipeline has 20+ passes, tracking down a miscompilation can be difficult despite good location tracking. Tools like `mlir-reduce` and `--mlir-print-ir-after-all` help but are not a silver bullet.

### 6. No Universal NPU Target (yet)
While MLIR is the right infrastructure, there is **no standard NPU dialect** that hardware vendors have converged on. Each vendor (AMD, Qualcomm, ...) still defines its own proprietary dialect. TOSA is the closest thing to a common layer, but it is high-level and hardware-neutral — not a machine-level target.

---

## Summary

```
MLIR in one sentence:
  A compiler infrastructure for building N-level, extensible, reusable,
  and composable IRs — the missing layer between ML frameworks and hardware.

Why it matters for ML/DL accelerators:
  ┌─────────────────────────────────────────────────────────┐
  │  MLIR decouples optimization from hardware targeting.   │
  │  Reuse passes across backends. Write one tiler,         │
  │  vectorizer, or quantizer — run on any target.          │
  │  NPU vendors write one dialect + one lowering pass      │
  │  and get the full ML framework ecosystem for free.      │
  └─────────────────────────────────────────────────────────┘
```

---

*Document based on MLIR upstream as of LLVM 19.x / early 2025. MLIR evolves rapidly; check https://mlir.llvm.org for the latest.*
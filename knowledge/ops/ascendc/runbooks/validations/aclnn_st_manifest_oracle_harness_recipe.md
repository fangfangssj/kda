---
type: "Runbook"
title: manifest 驱动 aclnn ST 框架（独立 oracle）
description: 自研 aclnn ST 三件套配方：用例 manifest + 独立 golden 生成器 + C++ 执行器；两端同源位流、确定性合成、红基线可验证。复现增量：绿基线（kernel 实现后）两轮全量 PASS + 逐位一致闭环补齐；NaN 逐点 isnan + rtol/atol 逐元素比对器；空张量 0 元素仍按最小 1 字节分配、描述符 shape 直通；fp16↔fp32 经 acl_base_rt.h 的 aclFloat16ToFloat/aclFloatToFloat16 链可达。
tags: [task.testing, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# manifest 驱动 aclnn ST 框架（独立 oracle）

## 现象

在 ops-registry-invoke 算子仓里为自定义算子开发 aclnn ST（系统测试，测试先行）时，发现没有现成测试框架可用：

- skill 文档（aclnn-st-framework）声称活参考是 `operators/add_custom_package/tests/aclnn/`，并引用核心头 `aclnn_st_core.h`——但 glob `**/aclnn_st_core.h` 全仓无果，`find /workspace/skills-dev-dev-1` 只命中 `skills/aclnn-st-framework/test_st_template.cpp` 模板。参考工程 `add_custom` 的 test_aclnn.cpp 也不用该框架。
- 已有的 whitebox 用例清单（TTK CSV，33 例）是结构性扫描用例：`input_data_ranges` 在转换产物中为空、值级信息大量为 null，且**单元素 NaN/+Inf 精确注入在 CSV range 语义里不可表达**——精确张量注入本来就是 aclnn ST 的职责。

而任务要求是：用例逐条对应 Validation.md 端到端用例矩阵行、走 `aclnnThnnFusedGruCellGradGetWorkspaceSize → aclnnThnnFusedGruCellGrad` 两段式调用链、期望值由**独立参考 oracle**（golden.py，纯 torch/numpy，禁止被测 kernel 自证）生成、kernel 接口桩阶段红基线（数值用例期望通过数=0）。于是需要自研一套"manifest 驱动 + 独立 oracle + C++ 执行器"的 ST 框架，且数据合成必须确定性、期望值必须可审计。

## 适用场景

- 工程形态：aclnn ST（ops-registry-invoke 算子仓 `tests/aclnn/`），测试先行/红基线或绿基线阶段；亦适用任何"python golden + C++ 设备执行"的双语言测试形态。
- API 路径：aclnn 两段式调用链（GetWorkspaceSize → aclnn 下发 → Sync）；python 侧 `importlib` 动态加载 golden TestSpec；TTK stat_rel_err 比对口径。
- 验证过的平台与 CANN 版本：Ascend950PR（npu-smi 25.7.rc1.6）+ CANN 9.0.0-beta.2 + torch 2.7.1+cpu（与 frontmatter `platforms` 严格一致）。
- 适用粒度：`operator_family`——框架结构与算子无关（换算子改 manifest 的 op/api 名与 golden 路径即可）。
- 路线优先级：`recommended_fastpath`（现成框架缺失时的标准自研路线；若上游补齐 aclnn_st_core.h 可评估替换）。
- 明确不适用/未验证场景：TTK CSV 能直接表达的常规范围用例（走 whitebox/TTK 通道更省）；绿基线（kernel 实现）下的运行本开发会话 未覆盖（红基线 trace）；INT 量化 dtype 用例（执行器 dtype 枚举未含）；多输入流并发的性能型 ST。

## 解决配方

三件套 + 编排，全部落 `tests/aclnn/`：

1. **用例 manifest（`st_cases_<branch>.json`，数据文件）**：每例声明 `case_id` / `source`（逐条引用 Validation.md 矩阵行，可审计）/ `B`/`H`/`dtype`/`has_bias` / `inputs`（`grad_hy` 与 `storage` 各自的 pattern：`uniform(lo,hi)`、`zeros`、`const(value)`、`forward_reachable`（五平面 r,z∈U(0,1)、n∈U(−1,1)、hx,hn∈U(−2,2)）+ `inject`（`set_nan`/`set_pos_inf`，支持 `flat_index` 或 `plane/row/col` 寻址））/ `assert_mode`（`stat_rel_err` 或 `exact_zero`）/ `threshold`。C++ 侧**禁止内联硬编码用例**——shape/dtype 全部来自 manifest。
2. **oracle 生成器（`gen_st_oracle.py`，python）**：读 manifest → 每例确定性合成输入（`torch.Generator`，`seed = crc32(case_id) & 0x7FFFFFFF`，storage 用 `seed ^ 0x5EED`）→ fp32 域合成、注入、**单次 cast** 到目标 dtype → 调独立 golden（`importlib.util.spec_from_file_location` 加载 golden.py；TestSpec.golden 首参不是 self，可直接 `ThnnFusedGruCellGradTestSpec.golden(t1, t2, has_bias)` 类上调用）→ 五路期望 + 两路输入按 dtype 位流落盘（bf16 走 `view(torch.uint16)` 位桥）→ 附 `meta.json`（seed、golden sha256 前 16 位、shape 契约）。golden 输出 shape 与 spec shape_rule 逐路核对。
3. **C++ 执行器（`test_aclnn.cpp`）**：内嵌最小 JSON 解析器（stjson）读 manifest → 读位流 → 上载输入 → **poison 输出** → 两段式 aclnn ×2 轮 → 读回 → stat_rel_err/exact_zero 比对 + 位级确定性检查。空 (0,) 输出按 32 字节占位分配、描述符 shape 仍为 {0}（autogen aclnn 层只查空指针、shape 直通；比对 n==0 短路）。
4. **编排（`run.sh`）**：构建算子包（build.sh）→ `rm -rf build && cmake && make` 构建测试 → `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 gen_st_oracle.py --manifest ... --out build/st_data --golden ...`（**每次全量重生成，禁用"已存在则跳过"缓存**）→ `./test_aclnn <manifest> <data_dir>`。生成数据落 `build/st_data`（随 `rm -rf build` 天然清理，无须改 .gitignore）。

**两端同源**纪律是配方核心：golden 的输入就是落盘的 .bin 本身（fp32 域合成 → 单次 cast → 落盘，device 上载同一份位流），杜绝 fp16/bf16 舍入边界处 device 与 golden 拿到不同数据导致的假失败。

## 设计决策

- **为何 python 生成、C++ 消费**（而不是 C++ 生成、python 算 golden）：单一事实源——python 一趟完成输入合成 + golden 计算，C++ 只读位流；避免两种语言各自合成导致的分布不一致。
- **为何每次重生成数据**：期望值必须与当前 golden 一致，缓存会让"golden 修了 bug、ST 还在用旧期望"的陈旧错配静默发生。
- **为何数据进 `build/st_data`**：`run.sh` 每轮 `rm -rf build`，生成数据随之清理再重建——不引入 .gitignore 变更，也不留陈旧产物。
- **为何 golden 用 importlib 动态加载**：golden.py 位于 docs/ 开发目录而非包内，且须与被测 kernel 源码级隔离（禁止自证）；`spec_from_file_location` 不依赖 sys.path 布局。TestSpec.golden 首参非 self（golden(grad_hy, storage, has_bias, *pure_outputs)），类上直调即可，无须实例化。
- **为何 C++ 内嵌 JSON 解析器**：测试二进制不引第三方依赖；manifest 是自产数据文件，受限子集（对象/数组/字符串（含 \u 转义与代理对）/数字/字面量）足够。
- **空输出 (0,) 的 32B 占位**：autogen aclnn 层（Nnopbase 框架码）的检查宏只查空指针、shape 直通；0 字节 `aclrtMalloc` 行为不明，占位分配 + 描述符 dims{0} 是稳妥路线（本开发会话 E04 实测通过）。比对侧 0 元素短路 PASS（TTK `_check_empty` 同语义）。
- **确定性双跑**：同输入连跑 2 次位级比对，作为每例的必检项——红基线阶段它在 stub 下平凡通过（两轮同为 poison），绿基线阶段才有区分力。
- **TORCH_DEVICE_BACKEND_AUTOLOAD=0**（python 生成步骤）：env.sh 记载该变量用于"关 torch_npu autoload，避免 GE-init 抢占 vendor tiling 注册导致 561002"——本开发会话 未复现 561002，属防御性沿用（来源：项目 env.sh，非本开发会话 实测）。
- **golden 独立性红线**：golden.py 头注释明确竞品 oracle（torch.ops.aten...）只能在隔离子进程调用——aclnn 进程已初始化 aclrt，同进程调用会 SIGSEGV；生成器是独立进程，安全。
- 被放弃的方案：直接复用 whitebox CSV（结构性用例、值级注入不可表达）；把用例硬编码进 C++（违反 manifest 驱动约定、不可审计）。
- 当前方案属 `recommended_fastpath` 路线。

## 系统信息

- 目录布局（算子仓内）：`operators/ThnnFusedGruCellGrad_package/tests/aclnn/{st_cases_branch0.json, gen_st_oracle.py, test_aclnn.cpp, run.sh, CMakeLists.txt}`；生成数据 `tests/aclnn/build/st_data/<case_id>/*.bin + meta.json`。
- 独立 oracle：`docs/ThnnFusedGruCellGrad/develop/golden.py`（纯 torch/numpy TestSpec；本开发会话 实测 sha256 前 16 位 b674160b54ea8f86）。
- 运行环境：Ascend950PR + CANN 9.0.0-beta.2；torch 2.7.1+cpu；宿主 x86_64。
- 构建链：`operators/build.sh <pkg>` 只构建/安装算子包（vendor .run → `$ASCEND_OPP_PATH/vendors/...`）；**tests 不在包构建内**，由 `tests/aclnn/CMakeLists.txt` 独立 cmake/make，链接 `ascendcl`/`nnopbase`/`cust_opapi`，include 用安装后的 vendor op_api 头（`aclnn_thnn_fused_gru_cell_grad.h`）——测试验证的是安装产物而非源码树。
- 运行期：`LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/<Vendor>/op_api/lib:$LD_LIBRARY_PATH`。
- 用例矩阵：20 例 = Validation.md branch-0 全部 15 行（E01–E06/S1–S6/N1/P5/P7）+ 分支设计 §9 补充 5 行（B0-a/c/d1/d2/e，B0-b==P5 已含）。

## 完整代码

### 1) oracle 生成器 `gen_st_oracle.py`（终版全文，verbatim）

```python
#!/usr/bin/env python3
"""Branch-0 aclnn ST 用例数据生成器（Task 28.1.3，测试先行）。

读取 ``st_cases_branch0.json``（分支 0 aclnn ST 用例清单，逐条对应
``docs/ThnnFusedGruCellGrad/design/Validation.md``「端到端测试用例矩阵」本分支行），
为每条用例**确定性**合成输入（grad_hy / storage），并调用**独立参考 oracle**
``docs/ThnnFusedGruCellGrad/develop/golden.py``（Validation.md「参考实现 (Golden)」，
纯 torch/numpy，不 import / 复用被测 kernel 任何产物——禁止被测 kernel 自证）
现算五路期望输出，全部以目标 dtype 原始位流落盘供 C++ ST 测试消费：

    <out>/<case_id>/grad_hy.bin                 输入 grad_hy  (B, H)   dtype 位流
    <out>/<case_id>/storage.bin                 输入 storage   (B, 5H)  dtype 位流
    <out>/<case_id>/exp_grad_input_gates.bin    期望 (B, 3H)
    <out>/<case_id>/exp_grad_hidden_gates.bin   期望 (B, 3H)
    <out>/<case_id>/exp_grad_hx.bin             期望 (B, H)
    <out>/<case_id>/exp_grad_input_bias.bin     期望 (3H,) / has_bias=false → (0,) 0 字节
    <out>/<case_id>/exp_grad_hidden_bias.bin    期望 (3H,) / has_bias=false → (0,) 0 字节
    <out>/<case_id>/meta.json                   溯源元数据（seed / golden sha256 / shape 契约）

「两端同源」纪律：device 侧（C++ aclnn 测试）与 golden 消费**同一份** dtype 化位流
——golden 的输入就是落盘的 .bin 本身（fp32 域合成 → 单次 cast 到目标 dtype → 落盘），
杜绝 fp16/bf16 舍入边界处 device 与 golden 拿到不同数据导致的假失败。

输入合成口径（Validation.md「参考实现 (Golden)」节）：
  - ``forward_reachable``：storage 五平面按前向可达分布采样——
    r, z ~ U(0,1)（sigmoid 输出域）、n ~ U(-1,1)（tanh 输出域）、hx, hn ~ U(-2,2)（任意有限值）；
  - ``uniform``：grad_hy ~ U(lo, hi)（上游梯度，任意有限值）；
  - ``zeros`` / ``const`` / ``inject``（set_nan / set_pos_inf）：spec extreme_inputs 值级物化。

确定性：每条用例独立 torch.Generator，seed = crc32(case_id)——同 manifest 多次生成位级一致。

用法：
    python3 gen_st_oracle.py --manifest st_cases_branch0.json --out <dir> \
        --golden <repo>/docs/ThnnFusedGruCellGrad/develop/golden.py
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import zlib
from pathlib import Path

import torch

PLANES = ("r", "z", "n", "hx", "hn")
OUTPUT_NAMES = (
    "grad_input_gates",
    "grad_hidden_gates",
    "grad_hx",
    "grad_input_bias",
    "grad_hidden_bias",
)
TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_golden(golden_path: Path):
    """import 独立 oracle golden.py（纯 torch/numpy，与被测 kernel 源码级隔离）。"""
    spec = importlib.util.spec_from_file_location("thnn_fused_gru_cell_grad_golden", golden_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ThnnFusedGruCellGradTestSpec.golden


def case_seed(case_id: str) -> int:
    return zlib.crc32(case_id.encode("utf-8")) & 0x7FFFFFFF


def synth_fp32(spec: dict, rows: int, cols: int, seed: int) -> torch.Tensor:
    """按 manifest 输入描述在 fp32 域合成 (rows, cols) 张量（确定性）。"""
    g = torch.Generator().manual_seed(seed)
    pattern = spec.get("pattern")
    if pattern == "zeros":
        t = torch.zeros(rows, cols, dtype=torch.float32)
    elif pattern == "const":
        t = torch.full((rows, cols), float(spec["value"]), dtype=torch.float32)
    elif pattern == "uniform":
        t = torch.empty(rows, cols, dtype=torch.float32).uniform_(
            float(spec["lo"]), float(spec["hi"]), generator=g
        )
    elif pattern == "forward_reachable":
        # storage 专用：五平面 [r|z|n|hx|hn] 前向可达分布（Validation.md 输入合成口径）
        planes = {
            "r": torch.empty(rows, cols // 5).uniform_(0.0, 1.0, generator=g),
            "z": torch.empty(rows, cols // 5).uniform_(0.0, 1.0, generator=g),
            "n": torch.empty(rows, cols // 5).uniform_(-1.0, 1.0, generator=g),
            "hx": torch.empty(rows, cols // 5).uniform_(-2.0, 2.0, generator=g),
            "hn": torch.empty(rows, cols // 5).uniform_(-2.0, 2.0, generator=g),
        }
        t = torch.cat([planes[p] for p in PLANES], dim=1)
    else:
        raise ValueError(f"unknown input pattern: {pattern!r}")
    return t


def inject(spec: dict, t: torch.Tensor, H: int) -> torch.Tensor:
    """值级注入（spec extreme_inputs）：fp32 域、cast 之前执行。"""
    for inj in spec.get("inject", []):
        op = inj["op"]
        if "flat_index" in inj:
            idx = int(inj["flat_index"])
        else:
            plane = inj["plane"]
            if plane not in PLANES:
                raise ValueError(f"unknown storage plane: {plane!r}")
            idx = int(inj["row"]) * t.shape[1] + PLANES.index(plane) * H + int(inj["col"])
        if not 0 <= idx < t.numel():
            raise ValueError(f"inject index {idx} out of range (numel={t.numel()})")
        if op == "set_nan":
            t.view(-1)[idx] = float("nan")
        elif op == "set_pos_inf":
            t.view(-1)[idx] = float("inf")
        else:
            raise ValueError(f"unknown inject op: {op!r}")
    return t


def save_bits(t: torch.Tensor, path: Path) -> None:
    """目标 dtype 原始位流落盘（bf16 经 uint16 位桥保真）。"""
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        path.write_bytes(t.view(torch.uint16).numpy().tobytes())
    else:
        path.write_bytes(t.numpy().tobytes())


def check_shape_contract(case: dict, outs, B: int, H: int) -> None:
    """golden 输出 shape 与 spec shape_rule 逐路核对（Validation.md「精度验收标准」）。"""
    has_bias = bool(case["has_bias"])
    expected = [
        (B, 3 * H), (B, 3 * H), (B, H),
        (3 * H,) if has_bias else (0,),
        (3 * H,) if has_bias else (0,),
    ]
    for name, got, want in zip(OUTPUT_NAMES, outs, expected):
        if tuple(got.shape) != want:
            raise ValueError(
                f"{case['case_id']}: golden 输出 {name} shape {tuple(got.shape)} != shape_rule {want}"
            )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="st_cases_branch0.json 路径")
    ap.add_argument("--out", required=True, help="输出数据目录（如 build/st_data）")
    ap.add_argument(
        "--golden",
        required=True,
        help="独立 oracle golden.py 路径（docs/ThnnFusedGruCellGrad/develop/golden.py）",
    )
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    golden_path = Path(args.golden)
    if not golden_path.is_file():
        print(f"FATAL: golden 不存在: {golden_path}", file=sys.stderr)
        return 2
    golden = load_golden(golden_path)
    golden_sha = hashlib.sha256(golden_path.read_bytes()).hexdigest()[:16]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = manifest.get("cases", [])
    if not cases:
        print("FATAL: manifest cases 为空", file=sys.stderr)
        return 2

    for case in cases:
        cid = case["case_id"]
        B, H, dtype_str = int(case["B"]), int(case["H"]), case["dtype"]
        if dtype_str not in TORCH_DTYPE:
            print(f"FATAL: {cid}: 不支持的 dtype {dtype_str}", file=sys.stderr)
            return 2
        dtype = TORCH_DTYPE[dtype_str]

        # 输入合成（fp32 域 → 注入 → 单次 cast 到目标 dtype；seed 按用例确定）
        grad_hy32 = inject(case["inputs"]["grad_hy"], synth_fp32(case["inputs"]["grad_hy"], B, H, case_seed(cid)), H)
        storage32 = inject(
            case["inputs"]["storage"], synth_fp32(case["inputs"]["storage"], B, 5 * H, case_seed(cid) ^ 0x5EED), H
        )
        grad_hy = grad_hy32.to(dtype)
        storage = storage32.to(dtype)

        # 独立 oracle 现算期望（golden 输入即落盘位流张量——两端同源）
        outs = golden(grad_hy, storage, bool(case["has_bias"]))
        check_shape_contract(case, outs, B, H)

        case_dir = out_root / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        save_bits(grad_hy, case_dir / "grad_hy.bin")
        save_bits(storage, case_dir / "storage.bin")
        for name, t in zip(OUTPUT_NAMES, outs):
            save_bits(t, case_dir / f"exp_{name}.bin")

        meta = {
            "case_id": cid,
            "source": case["source"],
            "B": B,
            "H": H,
            "dtype": dtype_str,
            "has_bias": bool(case["has_bias"]),
            "seed_grad_hy": case_seed(cid),
            "seed_storage": case_seed(cid) ^ 0x5EED,
            "golden": str(golden_path),
            "golden_sha256_16": golden_sha,
            "shapes": {
                "grad_hy": [B, H],
                "storage": [B, 5 * H],
                "outputs": [list(o.shape) for o in outs],
            },
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(
            f"[gen] {cid}: dtype={dtype_str} B={B} H={H} bias={int(bool(case['has_bias']))} "
            f"grad_hy={B * H} storage={B * 5 * H} outs={[list(o.shape) for o in outs]}"
        )

    print(f"[gen] done: {len(cases)} case(s) -> {out_root} (golden={golden_path}, sha256:{golden_sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 2) 用例 manifest（`st_cases_branch0.json` 结构 + 代表性条目，verbatim）

```json
{
  "schema_version": 1,
  "op_name": "ThnnFusedGruCellGrad",
  "api_name": "aclnnThnnFusedGruCellGrad",
  "task": "STATE.md Task 28.1.3 · Branch-0 Kernel ST（测试先行，红基线）",
  "branch": {
    "tiling_key": 0,
    "entry_condition": "非空（B>0 且 H>0）且 Group 触发条件不成立（aLoopCntTotal > coreNum/2 或 rLoopCntTotal <= 1），见 docs/ThnnFusedGruCellGrad/design/branches/DESIGN-BRANCH-0.md「分支定位」节"
  },
  "golden": {
    "path": "docs/ThnnFusedGruCellGrad/develop/golden.py",
    "spec": "docs/ThnnFusedGruCellGrad/design/Validation.md「参考实现 (Golden)」",
    "independence": "期望值由独立参考 oracle golden.py（纯 torch/numpy，TTK TestSpec ThnnFusedGruCellGradTestSpec）生成；不 import、不复用被测 kernel 的任何产物，禁止被测 kernel 自证"
  },
  "tolerance_by_dtype": {
    "float32": { "standard": "stat_rel_err", "threshold": 1e-5 },
    "float16": { "standard": "stat_rel_err", "threshold": 1e-3 },
    "bfloat16": { "standard": "stat_rel_err", "threshold": 4e-3 }
  },
  "compare_semantics": "stat_rel_err（同 TTK）：rel_err=|a-g|/(|g|+1e-7)；NaN 需两侧同 NaN、Inf 需两侧同号，否则 nan_inf_mismatch 判 FAIL；有限元素 mere<th 且 mare<10*th 判 PASS；五路输出分别比对均须过阈。exact_zero：五路输出逐元素 == 0.0（无容差，Validation.md S1 equals 0.0）。确定性：同输入连跑 2 次五路输出位级一致。",
  "cases": [
    {
      "case_id": "E01",
      "source": "Validation.md「端到端测试用例矩阵」正向 #E01（标准形状；DynamicShape 组①：(4,8)/(4,24)/(4,8)/(24,)）",
      "B": 4, "H": 8, "dtype": "float32", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-5
    },
    {
      "case_id": "S1",
      "source": "Validation.md「端到端测试用例矩阵」特殊值与极值 #S1（grad_hy 全零 → 五路输出 equals 0.0 精确断言，无容差）",
      "B": 4, "H": 8, "dtype": "float32", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "zeros" },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "exact_zero", "threshold": 1e-5
    },
    {
      "case_id": "S2",
      "source": "Validation.md「端到端测试用例矩阵」特殊值与极值 #S2（grad_hy 单元素 NaN → 对应 hidden 列五路输出全 NaN，位置逐位对齐）",
      "B": 4, "H": 8, "dtype": "float32", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0,
                     "inject": [ { "op": "set_nan", "flat_index": 13 } ] },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-5
    },
    {
      "case_id": "S3",
      "source": "Validation.md「端到端测试用例矩阵」特殊值与极值 #S3（storage 单元素 NaN（r 平面 row1 col3）→ 沿触及链路 IEEE 传播）",
      "B": 4, "H": 8, "dtype": "float32", "has_bias": true,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
        "storage": { "pattern": "forward_reachable",
                     "inject": [ { "op": "set_nan", "plane": "r", "row": 1, "col": 3 } ] }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-5
    },
    {
      "case_id": "E04",
      "source": "Validation.md「端到端测试用例矩阵」正向 #E04（has_bias=false：门梯度/grad_hx 正常、两路 bias 输出 (0,)）",
      "B": 4, "H": 8, "dtype": "float32", "has_bias": false,
      "inputs": {
        "grad_hy": { "pattern": "uniform", "lo": -2.0, "hi": 2.0 },
        "storage": { "pattern": "forward_reachable" }
      },
      "assert_mode": "stat_rel_err", "threshold": 1e-5
    }
  ]
}
```

（其余 15 例同构：E02/E03 换 dtype 与 threshold、S4 注入 `set_pos_inf`、S5 storage `zeros`、S6 grad_hy `{"pattern": "const", "value": 60000.0}` fp16、N1/P5/P7/B0-* 换 B/H，`source` 逐条引用矩阵行或分支设计 §9 行。）

### 3) run.sh 编排（Step 2–4，终版 verbatim）

```bash
# --- Step 2: Build the ACLNN test ---
##   Clean build directory, create it, run cmake and make.
cd "${SCRIPT_DIR}"
rm -rf build
mkdir -p build
cd build
cmake ..
make

## --- Step 3: Generate Branch-0 ST case data (inputs + expected from independent oracle) ---
##   st_cases_branch0.json → build/st_data/<case_id>/*.bin：输入按 manifest 确定性合成，
##   期望值由独立 oracle golden.py（纯 torch/numpy，不 import 被测 kernel）现算落盘。
##   每次全量重生成（禁用「已存在则跳过」缓存——保证期望值与当前 golden 一致）。
##   TORCH_DEVICE_BACKEND_AUTOLOAD=0：关 torch_npu autoload（同 develop/env.sh 口径）。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 "${SCRIPT_DIR}/gen_st_oracle.py" \
    --manifest "${MANIFEST}" --out "${SCRIPT_DIR}/build/st_data" --golden "${GOLDEN}"

## --- Step 4: Run the test against the freshly installed operator ---
##   The installed vendor library directory must be in LD_LIBRARY_PATH so the
##   runtime linker can find libcust_opapi.so.
##   ASCEND_OPP_PATH is set by the CANN environment; it typically resolves to
##   something like /home/<user>/Ascend/cann-9.1.0/opp.
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/ThnnFusedGruCellGrad/op_api/lib/:${LD_LIBRARY_PATH}
./test_aclnn "${MANIFEST}" "${SCRIPT_DIR}/build/st_data"
## 退出码：smoke + Branch-0 ST 全过 = 0；kernel 数值链未实现的接口桩阶段为红基线
## （数值用例 0 passed、仅最小通路用例通过），退出非 0 属预期（Task 28.1.3 口径）。
```

（头部变量：`REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"`、`MANIFEST="${SCRIPT_DIR}/st_cases_branch0.json"`、`GOLDEN="${REPO_ROOT}/docs/ThnnFusedGruCellGrad/develop/golden.py"`。）

### 4) C++ 执行器骨架（manifest 驱动部分，verbatim）

内嵌 JSON 解析器（stjson）终版构造——**Parser 按值持有文本**（初版把 `const char*` 指进临时 `std::string` 导致悬垂解析失败，修复后 s_ 指向自有 `text_`，成员声明顺序保证析构安全；由写入事件 + 修复 edit 组装）：

```cpp
class Parser {
public:
    explicit Parser(std::string text) : text_(std::move(text)), s_(text_.data()), n_(text_.size()) {}

    bool Parse(Value& out)
    {
        SkipWs();
        if (!ParseValue(out)) {
            return false;
        }
        SkipWs();
        return i_ == n_;
    }

private:
    // text_ 持有解析文本（s_ 指向其内部，须在 s_ 之前声明以保证析构顺序）
    std::string text_;
    const char* s_;
    size_t n_;
    size_t i_ = 0;
    // SkipWs/Consume/ParseValue/ParseLit/ParseNumber/ParseString（含 \uXXXX 与代理对）/
    // ParseArray/ParseObject：标准递归下降，方法体见测试源文件
};
```

套件驱动（读 manifest → 逐例执行 → 汇总）：

```cpp
static bool RunStSuite(const std::string& manifestPath, const std::string& dataDir,
                       int& total, int& passed, int& failed, int& errors)
{
    std::vector<StCase> cases;
    std::string err;
    if (!LoadManifest(manifestPath, cases, err)) {
        fprintf(stderr, "[ST] manifest 装载失败: %s\n", err.c_str());
        return false;
    }
    total = static_cast<int>(cases.size());

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        fprintf(stderr, "[ST] aclrtCreateStream 失败\n");
        return false;
    }

    printf("[ST] Branch-0 suite: manifest=%s (%d case(s)), data=%s\n", manifestPath.c_str(), total,
           dataDir.c_str());
    for (const auto& c : cases) {
        StCaseResult res;
        if (!RunStCase(c, dataDir, stream, res)) {
            ++errors;
            printf("[ST] %s ERROR: %s\n", c.caseId.c_str(), res.message.c_str());
            continue;
        }
        if (res.precisionOk && res.determinismOk) {
            ++passed;
        } else {
            ++failed;
        }
    }
    aclrtDestroyStream(stream);
    return true;
}
```

单用例执行 `RunStCase` 的八段流程（读位流 → shape 契约校验 → 设备分配（空输出 32B 占位）→ 输入上载 → poison + 两段式 ×2 轮 → 读回 → 比对 → 确定性），其 poison 调用点与比对器 `CompareTensor`、位级转换 `Fp16BitsToFloat`/`Bf16BitsToFloat` 的完整 verbatim 实现见配套各卡（[ST 红基线输出 poison 防 stub 巧合命中](st_red_baseline_poison_prefill_exact_zero.md)、[aclnn ST 自研 TTK stat_rel_err 比较器](aclnn_st_cxx_stat_rel_err_comparator_port.md)、[fp16/bf16 位流转 float 零依赖转换](cxx_fp16_bf16_bitstream_to_float_conversion.md)）——组合后即为完整测试文件。

## 预防

- 动手写 ST 前先确认框架现状：skill 文档引用的核心头是否真的存在（glob 一下），不存在就直接进自研路线，不要在"找框架"上空转。
- manifest 里每条用例必须带 `source` 溯源（引用矩阵行/设计文档行）——这是"逐条对应"验收与后续审计的唯一凭据。
- 期望值永远由独立 oracle 现算落盘，禁止被测 kernel 自证、禁止缓存旧期望。
- 生成器先在 /tmp 跑 sanity（独立 numpy 复算一两例 golden 位级一致）再接 C++——fail early。
- python 生成步骤带 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`（防 torch_npu autoload 干扰，见设计决策）。

## 验证信号

来源说明：以下信号来自 GRU Cell 反向算子 Branch-0 aclnn ST 的完整开发与红基线运行（kernel 接口桩、期望通过数=0 属任务预期口径）。

- 生成器 sanity（独立于被测 kernel 的验证）：20 例全部生成；E01 golden 五路期望经纯 numpy 从 .bin 复算 bitwise-equal；S1 期望全零；S2 单 NaN 传播计数 3/3/1/3/3（两路 gates 各 3、grad_hx 1、两路 bias 各 3）；S5 gin==go 直通且 grg/gig 全零；S6 fp16 60000.0 精确表示；E04 两路 bias 期望 0 字节（来源平台：950；证据等级：tool_observed）。
- 构建：任务指定命令 `( set -o pipefail; cd operators && ./build.sh ThnnFusedGruCellGrad_package )` 退出码 0（首跑与终态复核两次）；tests/aclnn cmake+make 编译链接通过（来源平台：950；证据等级：tool_observed）。
- 端到端 `run.sh` 全流程（构建包 → 构建测试 → 生成数据 → 运行）：`[gen] done: 20 case(s)`、smoke 1/1 PASSED、`Branch-0 ST summary: PASSED=0 FAILED=20 ERROR=0 TOTAL=20`、退出码 255（红基线预期非 0）（来源平台：950；证据等级：tool_observed）。
- 覆盖核对：20 例覆盖 Validation.md branch-0 全部 15 行 + DESIGN-BRANCH-0 §9 补充 5 行（B0-b==P5 已含）；`[REF_SAMPLE]` 注释保留 ×4；smoke PASSED 计数==1（来源平台：950；证据等级：tool_observed）。
- 确定性：全部 20 例同输入连跑 2 次五路输出位级一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- golden 接口耦合：`TestSpec.golden` 签名/类名变化需同步生成器；golden 文件移动需改 `--golden` 路径（meta.json 记录 golden sha256 前 16 位可供核对）。
- stjson 是受限 JSON 子集（无注释、无尾逗号；支持 \u 转义与代理对）——manifest 须由工具生成或按 schema 手写。
- 空输入张量（全 (0,) 输入的分支）未在框架中实测——本卡场景为空**输出**（占位 32B + dims{0}）；空输入分支的 ST 属另一分支任务，未验证。
- 绿基线（kernel 实现后）下整套框架的过阈/门禁行为未在本开发会话 验证（红基线 trace）。
- 确定性双跑在 stub 阶段平凡通过（两轮同为 poison），区分力要等绿基线。
- 数据目录 `build/st_data` 随 `rm -rf build` 清理——审计需要历史数据时须另行归档（meta.json 已含再生成所需的全部种子信息，可确定性重建）。
- 其他 NPU 平台/CANN 版本未实测，推断可迁移（框架仅依赖 aclnn 通用调用链与 host 侧 python/C++17）。

## 坏实践

- 用被测 kernel 自证期望值（任务红线，测试失去意义）。
- 期望位流入仓缓存、不随 golden 更新重生成（陈旧期望静默错配）。
- 用例 shape/dtype 硬编码进 C++（违反 manifest 驱动，换分支/加用例就要改代码）。
- 把 whitebox 结构性 CSV 直接当 ST 用例清单（值级注入不可表达、input_data_ranges 为空）。

## 误诊

- 首跑出现 `[ST] manifest 装载失败: manifest JSON 解析失败` 时，先怀疑 BOM/数字解析贪婪/尾随空白——实际根因是自研 stjson::Parser 把 `const char*` 指进构造传入的**临时 std::string**（完整表达式结束即析构，指针悬垂）；修复为 Parser 按值持有字符串（通用 C++ 生存期坑，与本框架的算子域知识无关，如实记录供自查）。
- 同一调试段里 `./test_aclnn ... | head -60; echo "EXIT=$?"` 显示 EXIT=0——那是管道末位 `head` 的退出码，不是测试的（实际 255）；要看测试退出码须去管道或用 PIPESTATUS（通用 shell 坑）。



## 后续增量（2026-09-17，并行策展批：绿基线闭环 + 逐点 NaN/rtol-atol 比对器 + 空张量最小分配）

softplus_v2_grad 算子 Branch-1（full-load）的 aclnn ST（Ascend950 + CANN 9.1.0，同三件套形态：独立 torch golden 离线生成期望 .bin、manifest 驱动、C++ 直连两级接口）补齐原卡失效边界的"绿基线未验证"并给出三个工程细节：

**1. 绿基线闭环（补齐"绿基线（kernel 实现后）下整套框架的过阈/门禁行为未验证"）**：kernel 桩（含哨兵预填）阶段 `run.sh` 端到端跑通、`total=15 PASSED=1 FAILED=14 ERROR=0`（仅空 tensor 用例合理通过）、退出码 1（红基线正确）；kernel 实现后**两次完整运行** `total=15 PASSED=15 FAILED=0 ERROR=0`、退出码 0，两次输出 `IDENTICAL bit-wise`——框架在绿基线下的过阈/确定性门禁行为实测成立。

**2. 逐点比对器（NaN 语义 + rtol/atol）**：期望为 NaN 的元素要求 actual `isnan`（NaN 传播语义逐点对齐），否则 `|a-b| <= atol + rtol*|expected|`（零幅值点由 atol 兜底）：

```cpp
// 逐点比对：expected 为 NaN -> actual 必须 isnan（NaN 传播语义）；
// 否则 |a-b| <= atol + rtol*|expected|（零幅值点用 atol）。返回不匹配元素数。
static int64_t compareElementwise(const std::vector<float>& actual,
                                  const std::vector<float>& expected,
                                  double rtol, double atol,
                                  int64_t printUpTo = 4) {
    int64_t n = static_cast<int64_t>(expected.size());
    int64_t mism = 0;
    int64_t printed = 0;
    for (int64_t i = 0; i < n; ++i) {
        float e = expected[i];
        float a = (i < static_cast<int64_t>(actual.size())) ? actual[i] : 0.0f;
        bool ok;
        if (std::isnan(e)) {
            ok = std::isnan(a);                       // S1: NaN 逐点 isnan 对齐
        } else {
            double diff = std::fabs(static_cast<double>(a) - static_cast<double>(e));
            double tol = atol + rtol * std::fabs(static_cast<double>(e));
            ok = (diff <= tol);
        }
        if (!ok) {
            ++mism;
            if (printed < printUpTo) {
                fprintf(stderr, "    [%ld] expected=%g actual=%g\n",
                        static_cast<long>(i), e, a);
                ++printed;
            }
        }
    }
    return mism;
}
```

**3. 空张量最小分配**：0 元素用例仍 `malloc` 最小 1 字节避免空指针（`allocBytes = (bytes == 0) ? 1 : bytes`），tensor 描述符 shape 仍为声明的 0 元素 shape；比对侧 `n == 0` 短路通过（与原卡"空输出 32B 占位"同一原则的最小化变体）。fp16 原始字节转 float 用 `aclFloat16ToFloat(uint16_t)`（`aclFloat16` 即 `uint16_t`，经 `acl/acl.h → acl_rt.h → acl_base.h → acl_base_rt.h` 链可达，无须自写位级转换）。

### 复现增量验证信号

- 红基线（kernel 桩、含哨兵预填）：`./run.sh` 端到端跑通，`total=15 PASSED=1 FAILED=14 ERROR=0`，RUN_SH_EXIT=1（红基线预期非 0）（来源平台：950；证据等级：tool_observed）。
- 绿基线：kernel 实现后两次完整运行均 `total=15 PASSED=15 FAILED=0 ERROR=0`、RUN1/RUN2 exit=0，两次输出 `IDENTICAL bit-wise`（来源平台：950；证据等级：tool_observed）。
- 构建集成：算子包 `./build.sh` 退出 0；测试 cmake+make 编译通过（来源平台：950；证据等级：tool_observed）。

### 复现增量边界

- 比对器语义是"逐元素 + NaN 传播 + rtol/atol"，不含逐位一致断言（逐位一致仅作为两次运行间的稳定性旁证）；要求 bit-exact 的算子需另加比较模式。
- manifest+bin 方案要求期望可离线确定性生成；含随机性/不确定归约顺序的算子需固定种子并在 golden 侧复现同样归约。
- fp16 容差与 fp32 两档是本算子设计选择，换算子需按精度预算重定。

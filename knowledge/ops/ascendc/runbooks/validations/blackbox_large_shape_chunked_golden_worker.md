---
type: "Runbook"
title: 大shape黑盒用例分块流式生成golden防OOM
description: 黑盒 ST 含 1.44B 元素 fp16 用例时，golden 侧按 8M 元素分块单遍流式写 .bin、C++ 侧 RAII 逐用例清理 /dev/shm，使 111 条大 shape 用例在 16G shm/128G cgroup 下全量跑通不 OOM。复现增量：OOM 事故侧证据链——worker 全量物化被 SIGKILL（exit 137、空 reply、shm 残留 5.4GB）、free 显示宿主内存是假象、v2→v3 修复前后对照与期望抽样核对。
tags: [task.testing, topic.memory, topic.implementation.memory_management]
status: stable
sources: []
created_at: '2026-09-17T00:49:13Z'
updated_at: '2026-09-17T02:41:05Z'
platforms: ["950"]
---

# 大shape黑盒用例分块流式生成golden防OOM

## 现象

对 softplus_v2_grad（aclnnSoftplusV2Grad，AscendC 自定义算子，ascend950）做黑盒 ACLNN ST 全量验证时，先统计了 manifest 的规模画像：517 条用例（L0 8 + L1 500 + L2 9）里有多达 **111 条超过 1 亿元素**（L0 2 条 + L1 109 条，全部 float16，且都不是 expect_error 异常用例——即必须真跑数据生成、golden 现算与逐元素比对），最大单条 1,438,413,792 元素（shape `[78999, 8, 2276]`）。

当时的执行环境是：`/dev/shm` 只有 **16G**（用前 0%），主机内存 754G，cgroup 限额 128GB。做内存账：一条 1.44B 元素 fp16 用例要落三张张量（grad_output、self、expected），`1.44e9 × 2B = 2.88GB/张 × 3 张 ≈ 8.64GB`；111 条大用例的字节总和（非并发）实测统计为 **748.7GB**。也就是说：如果在 host 侧一次性生成全量输入 + 期望再落盘，单条就逼近甚至超出 /dev/shm 容量（8.64GB < 16G 勉强放得下一条），累计必然把 shm 撑爆；Python golden 侧一次性 `torch` 全量张量也必然撞 cgroup 128GB OOM。

实际采用的解法是黑盒 ST 框架的 golden 常驻 worker（`golden_blackbox_worker.py`）按 **8M 元素一块**分块：从全局单一 RNG 流逐块取随机数、逐块调用外部 golden 现算期望、**单遍流式**追加写三个 `.bin` 文件，峰值内存只有 ~CHUNK 量级；C++ 测试侧再用 RAII 析构保证每条用例的三个 `/dev/shm` 临时 `.bin` 在**任意返回路径**（含提前 return 的失败路径）都被 unlink。最终全量 517/517 PASSED、0 FAIL、0 FATAL、无超时、无 OOM，golden worker 干净 EOF 退出。

## 适用场景

- 工程形态：aclnn 黑盒 ST（manifest 逐用例驱动 + 独立 golden oracle 现算期望）的全量执行；测试脚手架工程。
- API 路径：自建 `test_aclnn_blackbox` 类可执行 + 常驻 golden co-process（stdin JSON job 协议）；`aclrtMalloc/aclrtMemcpy` 上传 device；`.bin` 中转文件放 `/dev/shm`。
- 验证过的平台与 CANN 版本：Ascend 950 系列（device 0 真机全量跑通）；CANN 9.1.0（同一开发环境，据安装路径）。
- 适用粒度：`operator_family`（逐元素类 2 入 1 出算子的黑盒大 shape 验证直接套用；更广的算子族见失效边界）。
- 路线优先级：recommended_fastpath（含 >100M 元素用例的全量黑盒验证，不分块就跑不完）。
- 明确不适用 / 未验证场景：需要全局归约/跨块依赖的算子（golden 不能按块独立计算）不能直接套"块间独立"的分块 golden；`/dev/shm` 更小或 golden 中间量更大的环境需重估 CHUNK（未验证）。

## 判别方法（内存账怎么算）

在跑全量前，先用 manifest 统计做一遍规模画像与内存推导（本次实际执行的探针，`python3 << 'EOF'` heredoc 形式）：

```python
import json
d = json.load(open(".../aclnn_cpp_cases_blackbox.json"))
cases = d["cases"]
big = [c for c in cases if c["n_elements"] > 100_000_000]
print("cases >100M elements:", len(big))
from collections import Counter
print("  by level:", dict(Counter(c["level"] for c in big)))
print("  by dtype:", dict(Counter(c["input_dtypes"][0] for c in big)))
print("  expect_error among big:", sum(1 for c in big if c["expect_error"]))
for c in big[:8]:
    print("  %s lvl=%s n=%d dtype=%s shape=%s expect_err=%s" % (
        c["case_id"], c["level"], c["n_elements"], c["input_dtypes"][0],
        c["input_shapes"][0], c["expect_error"]))
# 总账：三张张量（2 输入 + 1 期望）按 dtype 字节累计
total_bytes = 0
for c in big:
    eb = 2 if c["input_dtypes"][0] == "float16" else 4
    total_bytes += c["n_elements"] * eb * 3
print("total bytes for big cases (sum, not concurrent):", total_bytes / 1e9, "GB")
```

观测输出（原文）：

```text
cases >100M elements: 111
  by level: {'L0': 2, 'L1': 109}
  by dtype: {'float16': 111}
  expect_error among big: 0
  aclnnSoftplusV2Grad_L0_003 lvl=L0 n=1438413792 dtype=float16 shape=[78999, 8, 2276] expect_err=False
  ...
total bytes for big cases (sum, not concurrent): 748.713449304 GB
```

结论链：单条大用例 3×2.88GB=8.64GB 需落 `/dev/shm`（16G，单条可容、两条即爆）→ 必须"单条驻留 + 用完即清"；Python 侧全量 golden 张量（1.44B 元素级）必撞 cgroup 128GB → 必须分块。容量核对命令：`df -h /dev/shm; free -g`（实测 shm 16G / Mem 754G available 707G）。

## 解决配方

分块三件套（量纲推导，非拍脑袋）：

1. **CHUNK = 8M 元素**：fp16 时 16MB/张、fp32 时 32MB/张落盘缓冲，golden（PyTorch CPU）中间量峰值 ~0.5GB——`_CHUNK` 注释原文："分块大小（元素数）：8M → fp32 32MB/tensor，golden 中间量峰值 ~0.5GB，安全。"
2. **单遍流式**：`while start < n` 循环里同时追写三个 `.bin`（输入 1、输入 2、期望），不做"先全量算完再写"。
3. **RAII 逐用例清理**：C++ 侧 `BinCleanup` 析构 unlink 三个 shm `.bin`，覆盖所有返回路径；host 侧输入在上传 device 后立刻 `swap` 释放。

执行方式（长跑全量必须后台化）：`run_blackbox.sh` 内部用 `timeout 7200 ./test_aclnn_blackbox <manifest>` 包裹；外层以 `nohup ./run_blackbox.sh > /tmp/blackbox_run.log 2>&1 &` 后台启动再轮询（前台包装命令会撞执行工具的默认 120s 超时被杀，但 nohup 的后台测试进程存活继续跑；轮询 sleep 取 90~100s，或显式给轮询命令更大的 timeout 参数）。

## 设计决策

- **为什么分块放在 golden worker（Python）侧**：内存瓶颈有两个——Python golden 全量张量（cgroup 128GB）与 `/dev/shm` 的 `.bin` 驻留（16G）。device 侧 128G HBM 放得下单用例 2.88GB×2 的输入，瓶颈不在 kernel 侧；分块只需在"生成 + 算期望"这一侧做。
- **为什么单遍流式而不是分块暂存再合并**：三路同时追加写即可保证块序即元素序，省一次全量读回的内存峰值与一倍磁盘占用。
- **RNG 全局单流、块间连续**：`np.random.RandomState` 只 seed 一次，逐块顺序取数——数学上等价于一次性取全量序列，因此分块**不改变**数据内容；同时保证"同 shape 的两路输入必不同""device 上传的输入与 golden 算期望的输入同源（同一份 .bin）"，满足 fp16 两端同源原则。若每用例/每输入重设 seed，同 shape 两输入会退化成相同数据，掩盖输入处理类缺陷。
- **`torch.set_num_threads(min(32, cpu_count))`**：避免超大 chunk 多线程临时缓冲叠加把峰值内存放大。
- **被放弃的方案**：一次性全量生成（内存不可行）；减小用例规模/跳过大 shape（违反全量执行验收标准，属测试侧放松，禁止）。

## 系统信息

- 工具链：CANN 9.1.0（安装根下的 toolkit），算子包 CMake + opbuild，构建产物按 `ascend950` 家族生成（`[ascend950] Generating SoftplusV2Grad_...`）。
- 目标平台：Ascend 950 系列，device 0。
- 目录布局：`operators/softplus_v2_grad_package/tests/aclnn/` 下 `test_aclnn_blackbox.cpp`（847 行，manifest 驱动）、`golden_blackbox_worker.py`（142 行，常驻 co-process）、`prepare_blackbox_manifest.py`（CSV→manifest）、`run_blackbox.sh`（CSV→manifest→构建→`timeout 7200` 执行）；golden oracle 在 `docs/softplus_v2_grad/develop/golden.py`（CPU+PyTorch，fp16 升 fp32，不调被测 kernel）。
- 相关文件清单（与本次实际读写一一对应）：`golden_blackbox_worker.py`（分块生成）、`test_aclnn_blackbox.cpp`（RAII 清理 + 比对）、`prepare_blackbox_manifest.py`（容差/expect_error 标记：fp16 rtol=atol=1e-3、fp32 1e-5）、`aclnn_cpp_cases_blackbox.json`（517 用例 manifest）。

## 完整代码

golden worker 的分块核心（`golden_blackbox_worker.py` 终版全文的关键段，verbatim）：

```python
# ===== 全局单一 RNG 流：seed 只设一次，逐用例逐输入按顺序取，绝不重设 =====
_RNG = np.random.RandomState(20260716)
# 分块大小（元素数）：8M → fp32 32MB/tensor，golden 中间量峰值 ~0.5GB，安全。
_CHUNK = 8_000_000

# 限制 torch 线程，避免超大 chunk 时多线程临时缓冲叠加
torch.set_num_threads(min(32, os.cpu_count() or 32))


def _gen_chunk(cnt, lo, hi, npdt):
    """从全局 RNG 流生成 cnt 元素。lo==hi（含 ±inf/±0）→ 常量；否则 uniform(lo,hi)。"""
    lo = _to_float(lo)
    hi = _to_float(hi)
    if lo == hi:
        return np.full(cnt, lo, dtype=npdt)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo = lo if np.isfinite(lo) else (-1e38 if lo < 0 else 1e38)
        hi = hi if np.isfinite(hi) else (-1e38 if hi < 0 else 1e38)
    return _RNG.uniform(lo, hi, size=cnt).astype(npdt)


def handle(job):
    dt = job["dtype"]
    n = int(job["n"])
    if dt not in _NP:
        raise ValueError(f"unsupported dtype for data-gen+golden: {dt}")
    npdt = _NP[dt]
    tdt = _TORCH[dt]
    ranges = job["ranges"]
    beta = _to_float(job["beta"])
    threshold = _to_float(job["threshold"])
    lo0, hi0 = ranges[0][0], ranges[0][1]
    lo1, hi1 = ranges[1][0], ranges[1][1]

    with open(job["grad_out_bin"], "wb") as fgo, \
         open(job["self_bin"], "wb") as fsf, \
         open(job["expected_bin"], "wb") as fexp:
        start = 0
        while start < n:
            cnt = min(_CHUNK, n - start)
            go = _gen_chunk(cnt, lo0, hi0, npdt)
            slf = _gen_chunk(cnt, lo1, hi1, npdt)
            fgo.write(go.tobytes())
            fsf.write(slf.tobytes())
            go_t = torch.from_numpy(go).to(tdt)
            slf_t = torch.from_numpy(slf).to(tdt)
            exp = _GOLDEN.softplus_v2_grad_golden(go_t, slf_t, beta=beta, threshold=threshold)
            fexp.write(exp.cpu().numpy().astype(npdt).tobytes())
            start += cnt
    return "OK"
```

C++ 侧的 RAII 清理与 host 内存释放（`test_aclnn_blackbox.cpp` `runCase` 内，verbatim）：

```cpp
    // RAII：确保 /dev/shm .bin 在任意返回路径（含 early return）都被清理，防累积撑爆 shm
    struct BinCleanup {
        std::string p1, p2, p3;
        ~BinCleanup() { if (!p1.empty()) unlink(p1.c_str()); if (!p2.empty()) unlink(p2.c_str()); if (!p3.empty()) unlink(p3.c_str()); }
    } binGuard;
```

```cpp
    // 释放 host 输入（device 已有副本）以省内存
    std::vector<uint8_t>().swap(goBytes);
    std::vector<uint8_t>().swap(selfBytes);
```

主循环与收尾（`main`，verbatim 关键段）：`tmpdir` 固定 `/dev/shm`；每用例 `runCase` 后 `fflush(stdout)`；结束打 summary 并 `return (passed == static_cast<int>(all.size())) ? 0 : 1;`。

## 预防

跑全量黑盒前的检查清单：

1. 统计 manifest 的 `n_elements` 分布，列出 >100M 元素用例的条数/level/dtype/是否 expect_error，算三张张量的字节账（×2B 或 ×4B ×3）。
2. `df -h /dev/shm; free -g` 核对 tmpdir 容量与 host 内存；单条大用例 `.bin` 总字节数必须 < shm 容量（留余量），否则换 tmpdir 或再压 CHUNK。
3. 确认 golden worker 是否分块（grep `_CHUNK` / `while start < n`），确认 C++ 侧 `.bin` 是否 RAII 清理（grep `unlink`）。
4. CSV 条数 == manifest 条数 == 执行 summary 的 total（本次 8+500+9=517 三环一致），防转换丢行。
5. 长跑命令一律 `nohup ... &` + 日志文件 + PID，轮询间隔 ≤100s 或显式设大 timeout。

## 验证信号

以下均为本次开发会话中对成品 kernel 全量复跑的实测输出：

- 黑盒全量 summary（来源平台：Ascend 950（device 0）；证据等级：tool_observed）：`total=517  PASSED=517  FAILED=0`，日志到达 `[517/517]`（无子集捷径），`grep -ciE "FATAL|timed out|TIMEOUT|Killed"` = 0，`[golden_worker] eof, exit` 干净收尾。
- 111 条 >100M 元素大 shape 用例全部包含在上述 517 条中且逐条 `[PASS]`（轮询观测大用例期间进程 RSS 稳定、/dev/shm 未爆、无 cgroup OOM kill）（来源平台：Ascend 950；证据等级：tool_observed）。
- 规模画像与内存账（来源平台：Ascend 950 环境 host；证据等级：tool_observed）：`cases >100M elements: 111`、`total bytes ... 748.713449304 GB`、`shm 16G 0 16G`。
- CSV==manifest==执行三环一致：L0 8 + L1 500 + L2 9 = 517（来源平台：Ascend 950 环境；证据等级：tool_observed）。

## 失效边界

- CHUNK=8M 是按 fp16/fp32 逐元素算子 + PyTorch CPU golden 的中间量估的；换 golden 实现（中间量更大）或 dtype（int8/bf16 等）需重算：峰值 ≈ CHUNK × (输入字节 + golden 中间量展开倍数)。
- `/dev/shm` 小于单用例三张 `.bin` 总量的环境（如 8G shm 遇 1.44B fp16 用例）：必须换 tmpdir 到大磁盘（牺牲 IO 速度）或把 `.bin` 也分块消费（C++ 侧流式读，本框架未实现，未验证）。
- golden 不能按块独立计算的算子（含全局归约、跨块依赖如 softmax/layerNorm 全局统计）：本"块间独立"分块公式不适用，需要两遍法（先统计后逐块），未在本轨迹验证。
- 950 之外的 NPU 平台未验证（推断可迁移：分块与平台无关，属 host 侧生成策略；未验证，推断可迁移）。

**未决项**：

- C++ 侧 `.bin` 仍是整文件读入 host 内存（`readFile(goPath)` 一次装 2.88GB）：单用例可行，但若单张 `.bin` 超过 host 可用内存需要改为分块流式上传（未实现）。
- 精确挂钟时长未计量（轮询观测全量 517 用例约半小时量级）；如需回归时长基线，应给 runner 加计时汇总。
## 坏实践

- 在 Python 侧一次性 `np.random.uniform(0, 1, size=1_438_413_792)` 或全量 torch 张量：直接撞 cgroup 128GB OOM（这正是本配方要防的）。
- 每用例或每输入重设 RNG seed：同 shape 两路输入退化为相同数据，输入处理类缺陷被掩盖。
- `.bin` 只在成功路径清理、早退路径泄漏：一条大用例失败后 shm 残留 8.64GB，后续大用例连环失败（RAII 就是防这个）。
- 为跑得动而删大 shape 用例、加 `--max-cases`、或只跑前 N 条：属测试侧放松，全量验收下禁止。

## 误诊

无（本闭环为正向实现 + 全量实测，无错误归因轮次）。

## 后续增量（2026-09-17，并行策展批：OOM 事故侧证据链——worker 全量物化被 SIGKILL 的失败现场与修复前后对照）

同一 blackbox ST 框架纳入最大用例时的**失败侧现场**（主卡叙事以事前内存账为主，本增量补事故发生后的诊断链与修复前后对照）：

**1. 失败现场**：单独验证最大用例（float16，n=1,438,413,792，单张量 2.88GB，三张量 8.64GB）时 golden co-process 静默死掉——`[FAIL] worker reply:`（空回复），`/dev/shm` 残留 5.4GB（.bin 未清理）。把 golden 单独拎出来计时复现：numpy 侧两路输入（各 2.88GB）刚物化完、进入 torch golden 计算前被 SIGKILL——`Killed`、`exit 137`（=128+9）。

**2. 容器内存判别（free 假象）**：`free` 显示 total 754 / available 703（GB）——**宿主视角，对容器内进程无参考意义**；容器 cgroup v2 `memory.max` = 137438953472（128GB）才是硬限制。golden 全量路径要同时持有 2×2.88GB fp16 输入 + torch fp32 升精度中间量 + 2.88GB 期望，峰值远超 128GB，被 OOM killer 击杀。

**3. 修复（worker v2→v3）**：v2 按用例全量物化（`gen_full(n)` 一次性生成两路输入 → torch → golden → 全量落盘）改为 v3 的 8M 元素分块单遍流式（主卡"解决配方"的完整代码即 v3 终版）；同用例修复后 worker `OK`、exit 0、全程 22s（01:44:32→01:44:54），三路 .bin 各 2876827584 字节落 /dev/shm。

**4. 期望抽样核对**：`expected[0] (go=-65504, z~ -0.12, dx~go*0.47): -31408.0`，与前 3 个元素手算一致——分块 golden 的期望值正确性旁证。

### 复现增量验证信号

- 修复前：1.44B 用例 worker reply 为空、`[FAIL]`、/dev/shm 残留 5.4GB；独立复现 `python3 ... Killed`，exit 137；cgroup memory.max=137438953472（来源平台：950 宿主容器；证据等级：tool_observed）。
- 修复后：分块 worker 处理同用例 `[golden_worker] ready, golden loaded, RNG seeded once, chunked` → `OK`，worker exit 0，全程 22s，三路 .bin 各 2876827584 字节落 /dev/shm（来源平台：950；证据等级：tool_observed）。
- 期望值抽样核对：`expected[0] ...: -31408.0`，与前 3 个元素手算一致（来源平台：950；证据等级：tool_observed）。
- 端到端：该用例在框架内单跑 PASS；随后 517 条全量两次执行均 517/517 PASSED、0 FAIL、0 SKIP、退出码 0（来源平台：950；证据等级：tool_observed）。

### 复现增量边界与坏实践增补

- 误诊增补：首次失败先怀疑"worker 协议/路径解析问题"（上一轮确有路径 bug），重跑后才定位到 OOM-kill——空 worker reply + /dev/shm 残留大文件 + exit 137 三点齐备即先查 cgroup 限额再查协议。
- 坏实践增补：用 `free`/`available` 判断容器内可用内存（本例显示 703GB available，实际限额 128GB）；worker 被杀后不清理 /dev/shm 残留 .bin（smoke 后残留 5.4GB 会挤压后续大用例，配套修复为 C++ 侧 RAII unlink 守卫）。
- `/dev/shm` 残留清理与 cgroup 判别亦见 [ST 批量 exit 137：容器 cgroup 限额与 NPU 清理等待](st_batch_exit_137_cgroup_limit_npu_cleanup_wait.md)（cgroup v1 路径与 NPU 清理等待维度）。
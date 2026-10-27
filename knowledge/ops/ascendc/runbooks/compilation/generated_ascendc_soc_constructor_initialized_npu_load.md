---
type: Runbook
title: "Generated AscendC SoC constructor versus initialized-runtime DSO loading"
description: "Distinguish CPU-only constructor abort from a same-byte DSO loaded after NPU initialization; preserve load, numerical and cause boundaries."
tags: [task.debugging, topic.implementation, topic.precision]
status: draft
sources: []
created_at: "2026-10-10T02:38:14Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["AscendCheckSoCVersion", "basic_string::_S_construct null not valid", "ctypes.CDLL constructor abort"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Generated AscendC SoC constructor versus initialized-runtime DSO loading

## 适用场景

A generated MatmulImpl pure-AIC DSO built and linked for DAV_3510 / CANN9.1.0 / bisheng15.0.5 aborts at CPU-only `ctypes.CDLL`, before a kernel call. Dependency inspection resolves its libraries and `ldd -r` reports no unresolved symbols. Retained disposition metadata reports rc133, `std::logic_error`, `basic_string::_S_construct null not valid`. The direct CPU abort raw-log locator is now **0 bytes**; this review cannot independently confirm the exception text from that raw log. Treat it as a metadata-reported observation, not a fresh loader replay.

The generated registration constructor calls `AscendCheckSoCVersion` before `RegisterAscendBinary`. Its complete runtime function implementation was not available in the examined SDK, so attributing the null argument to a specific internal function is **unproved**. Do not recast this as a link error, a numerical failure, or a confirmed universal requirement for all AscendC DSOs.

## Cause-specific discriminator

1. Preserve the actual loader error, generated registration order and dependency resolution. Bind compiler/runtime versions, original DSO SHA and loader process.
2. Distinguish missing symbols, an abort during a constructor, and a failed NPU runtime initialization. A TsdOpen/set-device failure is a separate observation.
3. Under an independently authorized device window, initialize the real NPU runtime first, then load the **identical DSO bytes**. Verify actual DSO hash and process maps; a successful CPU import of a changed binary is not this test.
4. If that load succeeds, report the observed deployment discriminator. Preserve the CPU-only failure and unproved internal cause. Continue the original full-denominator numerical and performance requirements for those exact bytes; loader success is not precision.

No device launch is authorized by this card. Use the caller's current resource authority and bound, rather than reviving a stopped guard or loading a peer's binary.

## Validation and failure boundaries

One same-DSO episode used SHA256 `4f9e8e7fad8e46d2d4a101bb59a164aed48415a24a812e456f9dc2b2fa9e0a18` in both contexts. CPU-only CDLL failure is reported by the retained disposition metadata, with the missing direct raw log disclosed above; loading after actual NPU initialization succeeded with the exact file hash and process maps. Build success and constructor failure remain separate.

The subsequent own WHITE candidate returned49/50 original cases:15 passed the original CPU grader and34 failed. Candidate/native compared equal on all49. The last case was interrupted by an owner stop and is NOT_RUN, not a numerical failure. Q was not measured. This observation demonstrates only loading behavior and a partial numerical mismatch; it does not grant full precision, performance or native-as-CPU oracle substitution.

One numerical source and one successful build campaign supplied this DSO. There was no fresh cold reproduction, isolated constructor-cause proof, cross-version validation or measured iteration saving. A future consumer must retain the original arithmetic/input/dtype/grader and independently validate its own DSO.

Adjacent mechanisms: [unresolved libcust_opapi symbol](libcust_opapi_l0op_contiguous_undefined_symbol.md) and [GE initialization / tiling interference](561002_torch_npu_ge_init_tiling_interference.md) require their own signals; their presence is not evidence that either caused this abort.

## Evidence access and authenticated reproduction

The episode evidence in sources is temporarily **not anonymously accessible** in the team Wiki. It is valid only with an authorized clone and an exact recheck; no password, token, PT tensor or DSO is published by this card. The fixed evidence commit is `fc8acb1b2a11d55f4c6837a4fab26071fd68de55`.

| 文件（同一固定提交） | Git blob | SHA256 |
|---|---|---|
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/CUBE02_BUILD_CPU_LOAD_DISPOSITION.json` | `ebf2da32c7231c5b823014fdb5e9169bce569ca6` | `b1a7aef172a5b2c6707b40e21debef6588fada0da9dc717c5109828771dfb1c5` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/CUBE02_CPU_ABI_LOAD_RAW.txt` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/CUBE02_CPU_LOAD_DIAGNOSIS_RAW.txt` | `09833ffe7fe97a793ea9d1bd70abc36bbba80f9d` | `2251ce133f4dcc3ce3844d4adbc4888c9bc7413b3f278539892632e3477c1ff8` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/CUBE02_DSO_LINK_IDENTITY_RAW.txt` | `e2ae33daf00c0d6b9c9607aaa6d154107d1a0a8a` | `50308337f0c9cde3506643b62c431c4401b40bd242a8b82bc50f6a215974aa04` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/CUBE_BUILD02_RAW.txt` | `5b7520be1d0fb2f9e10051d4018333b7752f13c9` | `9d3de6cb0a65db268ffe433c07d8cc7e2eaefb8eb32d0daa42785c94f4119cef` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/LIVE_HOST_CONSUMER_BINDING.json` | `0fc93e2c7a5f1a70aae707866d71755007702e1a` | `5bd06402750143f48e2005f22784f5aab3d5ed5a451b43a0efdf75c937f8f05b` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/OWNER_STOP_PARTIAL_TERMINAL.json` | `b24bcd7957b09d3156c86e35dae98900b80df7a2` | `a05f56c722109788c43f2ffe4d94450aa6e485a1e8dded2333d95182cd1d55bf` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/OWNER_STOP_PRECISION_ROWS.jsonl` | `c25456e6fce73b838733b35028480bf4295923e7` | `b064cfadd02332ffc3ab30d44d5d903a140ede06939932a25d3311e74f5a7c0d` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/README.md` | `299afbcf500abd8655a28b6b4be91639c5abe343` | `8b8ebcec602e7df4ed5c2776cdb0627d6d2ff8333938d2222d096f26d9d36a3f` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/SOURCE02_PRE_CUTOFF_FREEZE_0203.json` | `80172c7e8e576be9a9d76abf33ca1268ae424478` | `14859379feaa7eb58f2ed6e9dc45022ec5faa6f44bc36cd25eac6d09a04acc3a` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/SOURCE_MANIFEST.json` | `de4023ff83991d4b6f7160780e1e5043db590d51` | `1047b26f0f043b979a54bc26e5e502518652e9c7f359d9e772d2ed9c0a6967f5` |
| `projects/torch-to-a5-mfu-56/evidence/POSTMERGE_KNOWLEDGE_STOP_20261010T0355Z/constructor_source_evidence/consumer_cube02.py` | `5efb11eb72e71893c58ad132105d71d830735203` | `b7f5d8d6d368a33529e96692cabc2441d54180bd08a3383b29b722396b6b69a4` |

In an authorized clone, fetch main, recover bytes with `git show <fixed-commit>:<path>`, verify `git rev-parse <fixed-commit>:<path>` against the Git blob, then verify SHA256. SOURCE_MANIFEST.json binds original sizes and hashes. The empty abort file has SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; the exact exception text is known only from retained disposition metadata. The generated constructor excerpt, dependency report, loaded DSO maps/hash and49-row terminal have their own direct retained sources. These historical bytes were copied and inspected on local CPU; they do not establish new device execution or the unknown internal null origin.

Before upstream/public promotion, replace private episode sources with authorized public sources or rewrite the empirical scope. Stable team review is separate from public reproducibility. An active resource STOP forbids a replay even if this card describes a previous initialized load.

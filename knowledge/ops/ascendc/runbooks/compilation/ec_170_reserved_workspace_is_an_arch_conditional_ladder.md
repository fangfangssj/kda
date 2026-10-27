---
type: Runbook
title: "RESERVED_WORKSPACE is an arch-conditional LADDER, not a constant"
description: "`RESERVED_WORKSPACE` 不是常量而是按 `__NPU_ARCH__` 分 8 臂的条件梯子，grep 到的第一个值（2MB）是别的 arm；a5/a3 上实际是 16MB。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-170"]
---

# RESERVED_WORKSPACE is an arch-conditional LADDER, not a constant

**EC-170** — **`RESERVED_WORKSPACE` is an arch-conditional LADDER, not a constant** — `ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h:269` (sha256 `f3633d71…5275a62c`) defines it 8× as arms of one `#if/#elif __NPU_ARCH__` chain spanning `:269`-285, and a plain grep returns the `1001` arm FIRST (`:270`, 2 MB). On BOTH our targets — a5 `3510` and a3 `2201` — it is **16 MB**, exactly what EC-68/PB-53 already encode; a "2 MB on this CANN" report about a5/a3 is the signature of this misread, not a platform difference. Same trap on the accessor: `GetUserWorkspace` lives once per arch in `dav_<arch>/kernel_operator_common_impl.h` (a5 = `ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_common_impl.h:59`, sha256 `5199996e…d8c33013`); `dav_m510`/`dav_m310`/`dav_c220` are OTHER arches' copies and a citation into the wrong one looks entirely valid. Rule: resolve `__NPU_ARCH__` first, read that arm in that arch's dir, record arch + toolkit version + an `ext:` path and digest with any number. ⚠ SCOPE: measured on the CANN **9.1.0-beta.1** devkit ONLY — declared `cann=9.1.0-beta.1`, `unverified_on: cann=9.1.T500`. The undemonstrated `REGIST_MATMUL_OBJ` workspace-ownership mechanism this entry used to carry is now `CAND-MATMUL-WORKSPACE-EXCLUSIVE`; EC-170 makes no fault-code claim.

`applies_to: soc=all; cann=9.1.0-beta.1; bisheng=n/a; kernel_type=ACLRT_LAUNCH / standalone-pybind`
`verified_on: direct header read of the CANN 9.1.0-beta.1 devkit (innerversion V100R001C25B114, recorded at ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/ascend_all_cann_install.info:3) on the dev host, 2026-08-14. The ladder and its per-arch values are MEASURED, on that toolkit and no other`
`unverified_on: cann=9.1.T500 (the toolkit the originating op ran on — the ladder was read on 9.1.0-beta.1 headers and has NOT been re-read there; do not carry these values to a T500 target without re-reading)`

⚠️ **Scope, stated before the content because the entry was over-scoped once already**: this entry declares `cann=9.1.0-beta.1` — the single toolkit the headers were read on — not `9.1.x`. A `9.1.T500` consumer is out of scope and the KB_INDEX row's `cann=` tag excludes it from recall on that target. The workspace-ownership mechanism this entry used to carry as "Part 2" was never demonstrated by a passing run and has been moved to the candidate tier as `CAND-MATMUL-WORKSPACE-EXCLUSIVE`.

- **The reading trap (measured).** `ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h:269` — sha256 `f3633d71934af3b3e1b0c5d034da49e19f47ec772d752e48782f4ef65275a62c`, 750 lines — defines `RESERVED_WORKSPACE` **eight times**, as arms of a single `#if / #elif __NPU_ARCH__` ladder spanning lines 269–285 of that exact file (the eight definitions land on 270, 272, 274, 276, 278, 280, 282, 284):

  | `__NPU_ARCH__` | `RESERVED_WORKSPACE` |
  |---|---|
  | 1001 | 2 MB |
  | 2002 | 2 MB |
  | 2201 — a3, arch 220x | **16 MB** |
  | 3002 | 16 MB |
  | 3102 | 16 MB |
  | 3510 (a5, Ascend950PR, arch 351x) / 5102 | **16 MB** |
  | 3003 | 0 |
  | 3113 | 0 |

  A plain `grep -n RESERVED_WORKSPACE` on that file returns the **`1001` arm first — line 270, 2 MB**. Reading that as "the value on my CANN" is a first-hit error and it is **silent**: nothing about the matched line announces that it is one arm of a ladder. On **both** of our targets — a5 (`3510`) and a3 (`2201`) — the value is **16 MB**, which is exactly what EC-68 and PB-53 already encode. A "this CANN uses 2 MB" report about an a5 or a3 kernel is therefore the signature of this misread, not a platform difference.
- **The same trap on the accessor.** `GetUserWorkspace` is defined once **per arch** under `asc/impl/basic_api/dav_<arch>/kernel_operator_common_impl.h`. For a5 that file is `dav_3510` — `ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_common_impl.h:59`, sha256 `5199996e7a71d0ae7d34432cd6cea4748ba1146d7cb275b91ddd7876d8c33013`, 5825 bytes — where the function opens at `:51` and `:59` is its `return GetSysWorkSpacePtr() + RESERVED_WORKSPACE;`. `dav_m510`, `dav_m310`, `dav_c220`, `dav_m200` … are **other arches'** directories, each carrying its own copy of the same function at its own line number. A citation into the wrong `dav_*` directory looks entirely valid — same symbol, same expression, plausible line — and is simply not about your chip.
- **Rule**: never quote this constant BY VALUE across platforms, and never cite `dav_*` without checking which arch the directory is. Resolve `__NPU_ARCH__` for your target first, then read the arm that arch selects, in that arch's own `dav_<arch>/`. Any number you record must carry its arch **and** its toolkit version — and, since these headers live outside the repo, an `ext:` full path plus a digest, so a later reader can tell whether the file they are holding is the file that was read.
- **Evidence boundary**: this is a direct read of the devkit headers present on the dev host at CANN 9.1.0-beta.1, digests above. It is **not** re-verified against CANN 9.1.T500, the toolkit the originating op ran on — see `unverified_on`. **This entry does not amend EC-68's or PB-53's 16 MB figures**; it confirms them for a3 and a5 and explains how a 2 MB reading gets produced. It makes **no** claim about workspace aliasing or about producing any fault code — that half is now `CAND-MATMUL-WORKSPACE-EXCLUSIVE` and is undemonstrated.
- **Cross-ref**: [`EC-68`](ec_68_a_kernel_that_owns_its_workspace_via_getuserworksp.md) (`SetSysWorkspaceForce` first — the entry that owns `507015` on this path), [`PB-53`](pb_53_standalone_pybind_mix_cube_vec_507014_silent_hang.md) (the `507014` double-bootstrap hang), [`PB-54`](pb_54_passing_getsysworkspaceptr_to_regist_matmul_obj_un.md) (the `507015` wrong-base fixpipe fault — the two carry **different** codes and must not be quoted as one family under a single code), `CAND-MATMUL-WORKSPACE-EXCLUSIVE` (the undemonstrated workspace-ownership mechanism split out of this entry), `CAND-KFC-standalone-bootstrap-teardown`. backend=ascendc.

## 补充证据（2026-10-10 合并自 `reserved_workspace_arch_conditional_ladder_not_constant.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## EC-170 — EC-170: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=all；CANN 9.1.0-beta.1；kernel_type=ACLRT_LAUNCH / standalone-pybind（仅该 toolkit 头文件实测；9.1.T500 未验证，勿外推取值）。
- 验证来源：开发主机上 CANN 9.1.0-beta.1 devkit 头文件直接读取（innerversion V100R001C25B114，2026-08-14；ladder 及各 arch 取值在该 toolkit 上实测，且仅在该 toolkit）。

**EC-170** — **`RESERVED_WORKSPACE` is an arch-conditional LADDER, not a constant** — `ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h:269` (sha256 `f3633d71…5275a62c`) defines it 8× as arms of one `#if/#elif __NPU_ARCH__` chain spanning `:269`-285, and a plain grep returns the `1001` arm FIRST (`:270`, 2 MB). On BOTH our targets — a5 `3510` and a3 `2201` — it is **16 MB**, exactly what EC-68/PB-53 already encode; a "2 MB on this CANN" report about a5/a3 is the signature of this misread, not a platform difference. Same trap on the accessor: `GetUserWorkspace` lives once per arch in `dav_<arch>/kernel_operator_common_impl.h` (a5 = `ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_common_impl.h:59`, sha256 `5199996e…d8c33013`); `dav_m510`/`dav_m310`/`dav_c220` are OTHER arches' copies and a citation into the wrong one looks entirely valid. Rule: resolve `__NPU_ARCH__` first, read that arm in that arch's dir, record arch + toolkit version + an `ext:` path and digest with any number. ⚠ SCOPE: measured on the CANN **9.1.0-beta.1** devkit ONLY — declared `cann=9.1.0-beta.1`, `unverified_on: cann=9.1.T500`. The undemonstrated `REGIST_MATMUL_OBJ` workspace-ownership mechanism this entry used to carry is now `CAND-MATMUL-WORKSPACE-EXCLUSIVE`; EC-170 makes no fault-code claim.

- Former A5Ops ID: `EC-170`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=9.1.0-beta.1; bisheng=n/a; kernel_type=ACLRT_LAUNCH / standalone-pybind`
`verified_on: direct header read of the CANN 9.1.0-beta.1 devkit (innerversion V100R001C25B114, recorded at ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/ascend_all_cann_install.info:3) on the dev host, 2026-08-14. The ladder and its per-arch values are MEASURED, on that toolkit and no other`
`unverified_on: cann=9.1.T500 (the toolkit the originating op ran on — the ladder was read on 9.1.0-beta.1 headers and has NOT been re-read there; do not carry these values to a T500 target without re-reading)`

⚠️ **Scope, stated before the content because the entry was over-scoped once already**: this entry declares `cann=9.1.0-beta.1` — the single toolkit the headers were read on — not `9.1.x`. A `9.1.T500` consumer is out of scope and the KB_INDEX row's `cann=` tag excludes it from recall on that target. The workspace-ownership mechanism this entry used to carry as "Part 2" was never demonstrated by a passing run and has been moved to the candidate tier as `CAND-MATMUL-WORKSPACE-EXCLUSIVE`.

- **The reading trap (measured).** `ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/asc/impl/basic_api/utils/kernel_utils_constants.h:269` — sha256 `f3633d71934af3b3e1b0c5d034da49e19f47ec772d752e48782f4ef65275a62c`, 750 lines — defines `RESERVED_WORKSPACE` **eight times**, as arms of a single `#if / #elif __NPU_ARCH__` ladder spanning lines 269–285 of that exact file (the eight definitions land on 270, 272, 274, 276, 278, 280, 282, 284):

  | `__NPU_ARCH__` | `RESERVED_WORKSPACE` |
  |---|---|
  | 1001 | 2 MB |
  | 2002 | 2 MB |
  | 2201 — a3, arch 220x | **16 MB** |
  | 3002 | 16 MB |
  | 3102 | 16 MB |
  | 3510 (a5, Ascend950PR, arch 351x) / 5102 | **16 MB** |
  | 3003 | 0 |
  | 3113 | 0 |

  A plain `grep -n RESERVED_WORKSPACE` on that file returns the **`1001` arm first — line 270, 2 MB**. Reading that as "the value on my CANN" is a first-hit error and it is **silent**: nothing about the matched line announces that it is one arm of a ladder. On **both** of our targets — a5 (`3510`) and a3 (`2201`) — the value is **16 MB**, which is exactly what EC-68 and PB-53 already encode. A "this CANN uses 2 MB" report about an a5 or a3 kernel is therefore the signature of this misread, not a platform difference.
- **The same trap on the accessor.** `GetUserWorkspace` is defined once **per arch** under `asc/impl/basic_api/dav_<arch>/kernel_operator_common_impl.h`. For a5 that file is `dav_3510` — `ext:~/Ascend/cann-9.1.0-beta.1/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_common_impl.h:59`, sha256 `5199996e7a71d0ae7d34432cd6cea4748ba1146d7cb275b91ddd7876d8c33013`, 5825 bytes — where the function opens at `:51` and `:59` is its `return GetSysWorkSpacePtr() + RESERVED_WORKSPACE;`. `dav_m510`, `dav_m310`, `dav_c220`, `dav_m200` … are **other arches'** directories, each carrying its own copy of the same function at its own line number. A citation into the wrong `dav_*` directory looks entirely valid — same symbol, same expression, plausible line — and is simply not about your chip.
- **Rule**: never quote this constant BY VALUE across platforms, and never cite `dav_*` without checking which arch the directory is. Resolve `__NPU_ARCH__` for your target first, then read the arm that arch selects, in that arch's own `dav_<arch>/`. Any number you record must carry its arch **and** its toolkit version — and, since these headers live outside the repo, an `ext:` full path plus a digest, so a later reader can tell whether the file they are holding is the file that was read.
- **Evidence boundary**: this is a direct read of the devkit headers present on the dev host at CANN 9.1.0-beta.1, digests above. It is **not** re-verified against CANN 9.1.T500, the toolkit the originating op ran on — see `unverified_on`. **This entry does not amend EC-68's or PB-53's 16 MB figures**; it confirms them for a3 and a5 and explains how a 2 MB reading gets produced. It makes **no** claim about workspace aliasing or about producing any fault code — that half is now `CAND-MATMUL-WORKSPACE-EXCLUSIVE` and is undemonstrated.
- **Cross-ref**: EC-68 (`SetSysWorkspaceForce` first — the entry that owns `507015` on this path), PB-53 (the `507014` double-bootstrap hang), PB-54 (the `507015` wrong-base fixpipe fault — the two carry **different** codes and must not be quoted as one family under a single code), `CAND-MATMUL-WORKSPACE-EXCLUSIVE` (the undemonstrated workspace-ownership mechanism split out of this entry), `CAND-KFC-standalone-bootstrap-teardown`. backend=ascendc.

---

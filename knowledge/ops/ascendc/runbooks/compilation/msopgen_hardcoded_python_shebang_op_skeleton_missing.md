---
type: Runbook
title: "`cp: cannot stat './ops_<soc>/cmake'` in a custom-op build means `msopgen` never ran — its hard-coded `#!/root/anaconda3/bin/python` shebang is absent from standard images"
description: "`cp: cannot stat './ops_<soc>/cmake'` in a custom-op build means `msopgen` never ran — its hard-coded `#!/root/anaconda3/bin/python` shebang is absent from standard images"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-299"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-299 — EC-87: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all；CANN 9.1.0；op_class=all（任何通过 msopgen 生成 op 工程骨架的构建）。
- 验证来源：soc=Ascend910_9382（A3）；CANN 9.1.0；镜像 ascend-pytorch-hebing v26.0.0/py3.11（2026-07-30）。

**EC-299** — `cp: cannot stat './ops_<soc>/cmake'` in a custom-op build means **`msopgen` never ran**: its shebang is a hard-coded `#!/root/anaconda3/bin/python` absent from standard ascend-pytorch images, so the generated op-project skeleton is missing and the failure surfaces TWO layers later. Fix container-side (`ln -s $(which python3) /root/anaconda3/bin/python`) to keep CANN + upstream repo byte-identical.

- Former A5Ops ID: `EC-87`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=9.1.0; bisheng=n/a; op_class=all (any build that generates an op project via msopgen)`
`verified_on: soc=Ascend910_9382 (A3); cann=9.1.0; image ascend-pytorch-hebing v26.0.0/py3.11 (2026-07-30)`
**Symptom** (the message you actually see — two layers away from the cause):
cp: cannot stat './ops_Ascend910_9382/cmake': No such file or directory
cp: cannot stat './ops_Ascend910_9382/CMakePresets.json': No such file or directory
sed: can't read ./ops/CMakePresets.json: No such file or directory
**Root cause**: `$ASCEND_HOME_PATH/bin/msopgen` begins with a hard-coded
#!/root/anaconda3/bin/python
which does not exist in the standard ascend-pytorch container images. Any build step that generates an op-project skeleton via `msopgen gen -i <op>.json -c ai_core-<soc> -out <dir>` therefore produces **nothing**, and the failure only surfaces later when a subsequent `cp`/`sed` tries to consume the skeleton that was never created. The real error (`bad interpreter: No such file or directory`) is one scroll further up and is easily lost in a long build log.
**Detection**: `head -1 $(which msopgen)` — if the interpreter path does not exist, this is the bug. Confirm with `msopgen --help` (fails the same way).
**Fix** — repair the interpreter container-side so the upstream source tree stays unmodified:
```bash
mkdir -p /root/anaconda3/bin
ln -s "$(which python3)" /root/anaconda3/bin/python
```
Prefer this over editing the vendor script or the project's build scripts: it leaves both the CANN install and the upstream repo byte-identical, which matters when the whole point of the run is "does the UNMODIFIED upstream build?".
**Evidence**: A3, 2026-07-30. DeepEP `csrc/deepep/compile_ascend_proj.sh` calls `BuildOps()` → `msopgen gen ... -out ops_${soc_version}`, then `cp -r ./ops_${soc}/cmake ./ops`. Build died at the `cp` with exit 2; the `ops` tree genuinely lacks `cmake/` + `CMakePresets.json` in the repo (they are msopgen-generated, not vendored — `ops2/` ships its own, which is why a build limited to `ops2/` succeeds and masks the problem). After the symlink, the full build ran to completion and produced `custom_opp_*.run` + the pybind extension.
**Cross-ref**: PB-58 (the other silent-misdirection trap found in the same bring-up); runbook `src/references/moonep/graft/A3_CV_FUSED_MOE_RUNBOOK.md`. backend=ascendc.

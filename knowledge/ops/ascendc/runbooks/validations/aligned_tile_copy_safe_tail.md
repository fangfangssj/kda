---
type: Runbook
title: 完整块与安全尾块分开选搬运路径，循环可保持统一
description: 以普通 AdamW 的变长循环说明 DataCopy 与精确字节尾搬运的边界；复制路径分域不要求拆成两个独立循环，也不承诺固定提速。
tags:
- task.implementation
- topic.data_movement
- topic.alignment
status: draft
sources: []
created_at: '2026-10-17T18:12:04Z'
updated_at: '2026-10-22T13:17:58Z'
platforms: []
---

# 完整块与安全尾块分开选搬运路径，循环可保持统一

## 适用场景

连续分块的多输入 kernel，完整 tile 的源、目的起点和长度满足所选 DataCopy 重载的要求，末 tile 的有效长度可能较小。当前实现由每核的 `myTiles_`、`curLen_` 和循环索引计算有效 count；这些元数据必须与实际分核保持一致。

## 源码中的统一循环

原实现把完整块与最后的变长块放在同一个 `Process` 循环，只在 `CopyIn/CopyOut` 内按 `count == elementNumPerTile_` 选择普通 DataCopy 或按真实字节数的 DataCopyPad。`Compute(count)` 必须只消费有效 count；本卡不改变其数学操作。

```cpp
    __aicore__ inline void Process()
    {
        int64_t n = myTiles_;
        for (int64_t i = 0; i < n; ++i) {
            int64_t tileId = blockIdx_ + i * blockNum_;
            int64_t off = tileId * elementNumPerTile_;
            int64_t count = curLen_ - i * elementNumPerTile_;
            if (count > elementNumPerTile_) count = elementNumPerTile_;
            CopyIn(off, count);
            Compute(count);
            CopyOut(off, count);
        }
    }
private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        auto varL = inQVar_.AllocTensor<T>();
        auto gradL = inQGrad_.AllocTensor<T>();
        auto mL = inQM_.AllocTensor<T>();
        auto vL = inQV_.AllocTensor<T>();
        if (count == elementNumPerTile_) {
            AscendC::DataCopy(varL, varGm_[offset], count);
            AscendC::DataCopy(gradL, gradGm_[offset], count);
            AscendC::DataCopy(mL, mGm_[offset], count);
            AscendC::DataCopy(vL, vGm_[offset], count);
        } else {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(varL, varGm_[offset], cp, pad);
            AscendC::DataCopyPad(gradL, gradGm_[offset], cp, pad);
            AscendC::DataCopyPad(mL, mGm_[offset], cp, pad);
            AscendC::DataCopyPad(vL, vGm_[offset], cp, pad);
        }
        inQVar_.EnQue(varL);
        inQGrad_.EnQue(gradL);
        inQM_.EnQue(mL);
        inQV_.EnQue(vL);
    }
```

```cpp
    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto yL = outQY_.DeQue<T>();
        if (count == elementNumPerTile_) {
            AscendC::DataCopy(yGm_[offset], yL, count);
        } else {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[offset], yL, cp);
        }
        outQY_.FreeTensor(yL);
    }
```

## 搬运与分段合同

完整块 gate 只有在 `elementNumPerTile_` 的字节数及所有输入输出偏移均满足目标重载要求时才充分；不能只比较 count。尾块的 `blockLen` 使用 `count*sizeof(T)`，不能向上凑整跨越 GM 边界。为 UB 分配空间做对齐，与从 GM 读取的有效字节数是两件事。

这里“最后一个尾块”应限定到该分配段；多段或二维布局不保证全局只有一块尾部。零长度路径应在 host/tiling 或 Process 之前闭合，不能把 count=0 传给未确认支持的搬运重载。

## 验证信号

来源为普通 ApplyAdamW 导出实现 `csrc/ops/apply_adam_w/op_kernel/apply_adam_w_kernel.cpp`，文件 SHA-256 为 `12a6a42d920c4c667b1ee17a6dc5dea8d6483373c4d1bba6699eb2e6f80d326f`。本卡记录可直接审阅的源码结构。公开材料见[普通算子源码与单条评测证据](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/ordinary_adamw_gather_evidence_20261020.zip)（ZIP SHA-256 `44c0ccfe24752744aef8854f9a7b7dce191e069539342b41e0b0c4979aac054e`）；包内根目录为 `ordinary_optimization/`。公开附件提供证据入口，技术证据是包内下列文件。

`apply_adam_w_source_excerpt.txt` 保留 `Process`（原 L113–131）、`CopyIn`（L134–162）和 `CopyOut`（L257–267）的完整代码单位，只删除非许可注释，并非可独立编译工程；原行号、完整源文件／原范围／公开节录哈希见 `apply_adam_w_source_binding.json` 的 `/original_source`、`/published_excerpt/ranges`。单条脱敏历史记录为 `apply_adam_w_historical_record.json`：`/benchmark` 标识 official-tasks 1.1.0、A3（910c），`/result` 为 compile_pass=true、80/80，`/artifact_binding` 绑定原 report、选中原记录、原源包和源码 SHA-256。恢复 receipt 与该原记录的同提交关联经过核对，未重新打开原包。

上述历史数值仅属于原导出完整 artifact，不能据此认定本卡分支各自得到覆盖，也不认证实际加载二进制、CANN 版本、单机制因果或性能收益。本轮没有重新编译或运行；`platforms: []` 保持为空。

可从完整循环和两个 copy 方法核对同一 offset/count 的传递、四个输入的对称尾处理及输出的有效字节数。当前源码的统一循环构成“必须拆成两个独立循环”的具体反例；源注释声称减少 drain/restart，但本卡没有独立时间线或配对计时，不把该解释写成已量化收益。

## 失效边界与采用检查

每条输入、输出可有不同 stride 或 dtype，不能机械复用一个字节数。队列 ownership、复用完成性与计算 count 仍按[双缓冲槽位闭合](../../guides/synchronization/double_buffer_template_queue_occupancy_and_event_closure.md)核对，不能因完整块使用 DataCopy 就删依赖。采用时覆盖完整块、余数 1、接近块长的尾部、最后核、空工作与多输入，并在同环境对比完整调用成本。

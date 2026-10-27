# Mmad辅助配置说明

- [SetHF32Mode](set_hf32_mode.md)：用于设置Mmad计算是否开启HF32模式，开启该模式后L0A Buffer/L0B Buffer中的FP32数据将在参与Mmad计算之前被舍入为HF32。
- [SetHF32TransMode](set_hf32_trans_mode.md)：设置HF32模式取整的具体方式，需要先使用SetHF32Mode开启HF32取整模式。
- [SetMMColumnMajor/SetMMRowMajor](set_mm_column_major_set_mm_row_major.md)：控制Mmad/MmadWithSparse优先通过M/N的哪个方向。

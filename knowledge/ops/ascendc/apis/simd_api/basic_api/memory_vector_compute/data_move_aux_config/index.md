# 数据搬运辅助配置接口

- [NdDmaDci](nd_dma_dci.md)：刷新NDDMA DataCache，确保多核场景下DataCopy多维搬运前缓存数据为最新状态。
- [ResetLoopModePara](reset_loop_mode_para.md)：重置loop mode参数，与SetLoopModePara搭配使用，在使能loop mode的数据搬运结束后调用。
- [SetLoopModePara](set_loop_mode_para.md)：使能并设置DataCopy/DataCopyPad的loop mode参数，支持GM与UB之间的双向数据搬运循环模式。
- [SetPadValue](set_pad_value_isasi.md)：设置DataCopyPad非对齐搬运时需要填充的数值，支持GM到UB的填充数据自定义。

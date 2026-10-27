# SoftMax接口

- [AdjustSoftMaxRes](adjust_soft_max_res.md)：调整SoftMax计算结果为指定值，当maxTensor中存在指定值时，按行将对应softMaxRes结果替换为自定义值，用于SoftMax后处理。
- [IsBasicBlockInSoftMax](is_basic_block_in_soft_max.md)：用于判断SoftMaxTiling结构是否符合基本块特征。
- [SimpleSoftMax](simple_soft_max.md)：使用预先计算好的sum和max数据对输入tensor按行做Softmax计算，内部不含reduce过程。
- [SoftMax](soft_max.md)：对输入tensor按行做Softmax计算，内部包含reducemax、exp、reducesum和div过程，可输出sum和max中间结果，支持ND和NZ格式。
- [SoftMax/SimpleSoftMax Tiling](soft_max_simple_soft_max_tiling.md)：用于获取SoftMax/SimpleSoftMax Tiling参数。
- [SoftMax Tiling使用说明](soft_max_tiling_usage.md)：Ascend C提供一组SoftMax Tiling API，方便用户获取SoftMax kernel计算时所需的Tiling参数。
- [SoftmaxFlash](softmax_flash.md)：Softmax增强版本，可根据上一次Softmax计算的sum和max更新本次结果，适用于last轴切轴场景，不开启update时等同于SoftMax。即将废弃，建议使用SoftmaxFlashV2。
- [SoftmaxFlash Tiling接口](softmax_flash_tiling_interface.md)：注意：该接口后续即将废弃，新开发内容不要使用该接口。 用于获取SoftmaxFlash Tiling参数。
- [SoftmaxFlashV2](softmax_flash_v2.md)：SoftmaxFlash增强版本，对应FlashAttention-2算法，支持通过isUpdate参数控制是否根据前一次的sum和max更新当前结果，支持ND和NZ格式。
- [SoftmaxFlashV2 Tiling接口](softmax_flash_v2_tiling_interface.md)：用于获取SoftmaxFlashV2接口所需的Tiling参数。
- [SoftmaxFlashV3](softmax_flash_v3.md)：SoftmaxFlash增强版本，对应Softmax PASA算法。对输入tensor尾轴切分并按update模式计算softmax的均值、最大值和指数和，仅支持ND格式输入。
- [SoftmaxFlashV3 Tiling接口](softmax_flash_v3_tiling_interface.md)：用于获取SoftmaxFlashV3接口所需的Tiling参数。
- [SoftmaxGrad](softmax_grad.md)：对输入tensor按行做softmax梯度反向计算，支持ND和NZ格式，通过isFront参数控制是否仅计算前向求和结果。
- [SoftmaxGradFront](softmax_grad_front.md)：对输入tensor按行做softmax梯度前向计算，计算grad与src乘积的按行求和结果并按datablock广播输出，支持ND和NZ格式。
- [SoftmaxGrad Tiling接口](softmax_grad_tiling_interface.md)：用于获取SoftmaxGrad Tiling参数。

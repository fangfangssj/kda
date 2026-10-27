# mask寄存器计算

- [all_mask](all_mask.md)：头文件路径为："tensor_api/arch/vector/experimental/mask_reg_compute.h"，命名空间为asc::te::experimental。
- [deinterleave(掩码解交织)](deinterleave.md)：头文件路径为："tensor_api/experimental/arch/vector/mask_reg_compute.h"，命名空间为asc::te::experimental。
- [make_mask](make_mask.md)：头文件路径为："tensor_api/experimental/arch/vector/mask_reg_compute.h"，接口命名空间为asc::te::experimental。
- [mask的用法](mask_usage.md)：reg_tensor同时保存数据寄存器reg和与数据关联的掩码寄存器mask。计算接口根据输入reg_tensor的mask选择参与计算的有效元素，并将计算所使用的mask传递给返回的reg_tensor。local_tensor::store的单源direct、pack和pack_quarter模式继续使用源reg_tensor中保存的mask控制写回。
- [none_mask](none_mask.md)：头文件路径为："tensor_api/arch/vector/experimental/mask_reg_compute.h"，命名空间为asc::te::experimental。
- [update_mask](update_mask.md)：头文件路径为："tensor_api/experimental/arch/vector/mask_reg_compute.h"，命名空间为asc::te::experimental。

- [interleave（掩码交织）](interleave.md)：头文件路径为："tensor_api/experimental/arch/vector/mask_reg_compute.h"，命名空间为asc::te::experimental。

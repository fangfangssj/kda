# 矩阵计算的搬入

- [copy（Global Memory到L1 Buffer数据搬运）](copy_gm_to_l1.md)：头文件路径为：tensor_api/tensor.h。
- [copy（L1 Buffer到BiasTable Buffer数据搬运）](copy_l1_to_biastable.md)：头文件路径为：tensor_api/tensor.h。
- [copy（L1 Buffer到Fixpipe Buffer数据搬运）](copy_l1_to_fixbuf.md)：头文件路径为：tensor_api/tensor.h。
- [copy（L1 Buffer到L0A Buffer数据搬运）](copy_l1_to_l0a.md)：头文件路径为：tensor_api/tensor.h。
- [copy（L1 Buffer到L0B Buffer数据搬运）](copy_l1_to_l0b.md)：头文件路径为：tensor_api/tensor.h。
- [copy（L1 Buffer到L0ScaleA Buffer数据搬运）](copy_l1_to_l0scalea.md)：头文件路径为：tensor_api/tensor.h。
- [copy（L1 Buffer到L0ScaleB Buffer数据搬运）](copy_l1_to_l0scaleb.md)：头文件路径为：tensor_api/tensor.h。
- [copy（L1 Buffer到Unified Buffer数据搬运）](copy_l1_to_ub.md)：头文件路径为：tensor_api/tensor.h。
- [矩阵计算的搬入总体说明](overall_description.md)：矩阵计算的搬入接口主要实现Global Memory到L1 Buffer和L1 Buffer到L0A Buffer/L0B Buffer/L0ScaleA Buffer/L0ScaleB Buffer/BiasTable Buffer/Fixpipe Buffer的数据高效传输。

# 矢量数据搬运

- [asc_copy](asc_copy.md)：将数据从Unified Buffer（UB）搬运到UB。支持高维切分。
- [asccopygm2ub](asc_copy_gm2ub/index.md)：共 2 张知识卡（如 asc_copy_gm2ub、asc_copy_gm2ub）。
- [asccopygm2ubalign](asc_copy_gm2ub_align/index.md)：共 2 张知识卡（如 asc_copy_gm2ub_align、asc_copy_gm2ub_align）。
- [asccopyub2gm](asc_copy_ub2gm/index.md)：共 2 张知识卡（如 asc_copy_ub2gm、asc_copy_ub2gm）。
- [asccopyub2gmalign](asc_copy_ub2gm_align/index.md)：共 2 张知识卡（如 asc_copy_ub2gm_align、asc_copy_ub2gm_align）。
- [asc_copy_ub2l1](asc_copy_ub2l1.md)：提供数据搬运功能，将数据从Unified Buffer (UB)搬运到L1 Buffer。
- [asc_copy_ub2ub](asc_copy_ub2ub.md)：将数据从Unified Buffer（UB）搬运到UB。
- [asc_ndim_copy_gm2ub](asc_ndim_copy_gm2ub.md)：多维数据搬运接口，将数据从Global Memory (GM)搬运到Unified Buffer（UB）。相比于基础数据搬移接口，可以更加自由配置搬入的维度信息。 本功能最多能处理5个维度，每个维度都可以单独配置Stride，且支持随路Padding功能。相关配置参数有很多，需要通过不同的接口传入。
- [asc_set_copy_pad_val](asc_set_copy_pad_val.md)：和asc_copy_gm2ub_align接口配合使用，设置连续搬运数据块左右两侧需要填补的数据值。
- [asc_set_gm2ub_loop1_stride](asc_set_gm2ub_loop1_stride.md)：将数据从Global Memory (GM)搬运到Unified Buffer（UB）时，通过调用该接口设置内层循环中相邻迭代数据块间的间隔。
- [asc_set_gm2ub_loop2_stride](asc_set_gm2ub_loop2_stride.md)：将数据从Global Memory (GM)搬运到Unified Buffer（UB）时，通过调用该接口设置外层循环中相邻迭代数据块间的间隔。
- [asc_set_gm2ub_loop_size](asc_set_gm2ub_loop_size.md)：将数据从Global Memory (GM)搬运到Unified Buffer（UB）时，通过调用该接口设置数据搬运流程中的循环次数。
- [asc_set_ndim_loop_stride](asc_set_ndim_loop_stride.md)：本接口用于设置asc_ndim_copy_gm2ub进行NDDMA多维数据搬运时，loop0至loop4各循环维度的源操作数步长和目的操作数步长。
- [asc_set_ndim_pad_count](asc_set_ndim_pad_count.md)：本接口用于配置NDDMA高维切分搬运中loop1至loop4各维左右两侧的填充元素数。配置结果供后续调用asc_ndim_copy_gm2ub时使用。
- [asc_set_ndim_pad_value](asc_set_ndim_pad_value.md)：本接口用于设置asc_ndim_copy_gm2ub在常数填充模式下使用的填充值。当padding_mode设置为true时，后续调用asc_ndim_copy_gm2ub将使用该值填充搬运数据的边界区域。
- [asc_set_ub2gm_loop1_stride](asc_set_ub2gm_loop1_stride.md)：本接口用于设置asc_copy_ub2gm_align使用循环搬运模式时，内层循环中相邻两次迭代源操作数数据块起始地址之间的偏移量，以及目的操作数数据块起始地址之间的偏移量。
- [asc_set_ub2gm_loop2_stride](asc_set_ub2gm_loop2_stride.md)：本接口用于设置asc_copy_ub2gm_align使用循环搬运模式时，外层循环中相邻两次迭代源操作数数据块起始地址之间的偏移量，以及目的操作数数据块起始地址之间的偏移量。
- [asc_set_ub2gm_loop_size](asc_set_ub2gm_loop_size.md)：本接口用于设置asc_copy_ub2gm_align使用循环搬运模式时的两层循环迭代次数。loop1_size用于设置内层循环loop1的迭代次数，loop2_size用于设置外层循环loop2的迭代次数。

- [asc_ndim_copy_dci](asc_ndim_copy_dci.md)：头文件路径为："c_api/vector_datamove/vector_datamove.h"。

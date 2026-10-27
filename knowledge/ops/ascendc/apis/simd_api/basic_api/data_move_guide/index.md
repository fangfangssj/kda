# 数据搬运导览

- [GmL1OrL0DataMove](gm_l1_or_l0_data_move.md)：提供Global Memory与L1 Buffer及L0 Buffer间数据搬运接口概述，涵盖连续搬运、高维切分、格式转换及矩阵计算结果搬出等场景。
- [GmUbDataMove](gm_ub_data_move.md)：提供Global Memory与Unified Buffer间数据搬运接口概述，支持连续搬运、高维切分、切片搬运、格式转换及非对齐搬运等场景。
- [L1与BiasTable Buffer数据搬运](l1_bias_table_data_move.md)：DataCopy能够实现L1 Buffer至BiasTable Buffer的连续数据搬运，数据搬运时格式和内容保持不变。
- [L1与Fixpipe Buffer数据搬运](l1_fixpipe_data_move.md)：DataCopy能够实现L1 Buffer至Fixpipe Buffer的连续数据搬运，数据搬运时格式和内容保持不变。
- [L1与L0数据搬运](l1_l0_data_move.md)：L1 Buffer与L0 Buffer之间的数据搬运提供了灵活的接口支持，能够充分适配多样化的计算场景（见表1）。其中，L1 Buffer-\>L0A Buffer、L1 Buffer-\>L0B Buffer的数据搬运用于矩阵计算（参考矩阵计算的搬入），L0C Buffer-\>L1 Buffer用于矩阵计算的搬出，完成矩阵计算的输出流程（参考矩阵计算的搬出）。
- [L1OrL0CUbDataMove](l1_or_l0_c_ub_data_move.md)：提供L1 Buffer及L0C Buffer与Unified Buffer间数据搬运接口概述，涵盖连续搬运、高维切分、格式转换及随路量化激活等场景。
- [概览](overview/index.md)：数据搬运导览，介绍AI Core分层存储架构中的数据通路、TPosition映射、搬运功能分类及通用约束等核心概念。（共 4 张知识卡：数据搬运概念（概述）、数据通路（概述）、总体约束说明（概述））
- [UB与UB数据搬运](ub_ub_data_move.md)：Unified Buffer（UB）之间的数据搬运提供了灵活的接口支持，共有两组接口：DataCopy提供了基础的数据搬运能力，支持连续和非连续数据搬运；Copy除了支持连续和非连续数据搬运，还提供了mask操作和DataBlock间隔操作的能力，支持在接口内部和接口外部设置mask。具体接口功能如下表所示。

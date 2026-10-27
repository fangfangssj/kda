# matmul tuning cases

- [AIV核上的ND2NZ格式转换](aiv_nd2nz_conversion.md)：矩阵乘算子中输入矩阵内轴非256字节对齐时，在AIV核上完成ND到NZ格式转换以提升随路搬运效率。
- [Matmul高阶API开启IBShare模板共享A和B矩阵数据](matmul_ibshare_ab.md)：本案例呈现了在融合算子场景中，使用Matmul高阶API进行矩阵乘法计算时，A矩阵和B矩阵同时启用IBShare对性能的提升效果。
- [Matmul高阶API开启IBShare模板共享B矩阵数据](matmul_ibshare_b.md)：本案例呈现了在矩阵乘算子场景中，使用Matmul高阶API进行矩阵乘法计算，B矩阵开启IBShare对算子性能的提升效果。IBShare功能通过共享L1 Buffer上相同的A矩阵或B矩阵数据，减少重复的MTE2数据搬运开销，提升算子性能。该功能支持A矩阵和B矩阵其中一个矩阵开启IBShare，也支持A矩阵和B矩阵同时开启IBShare。
- [Matmul高阶API开启L2 Cache切分](matmul_l2_cache_split.md)：本案例呈现了在Matmul计算过程中，输入和输出的数据总量超过L2 Cache大小时，通过L2 Cache数据切分对算子性能的提升效果。开启L2 Cache切分的完整样例请参考L2 Cache切分的算子样例。
- [Matmul高阶API开启MDL模板](matmul_mdl_template.md)：开启MDL模板使MTE2一次性搬入多个基本块到L1 Buffer，提升带宽利用率，减少搬运次数。
- [Matmul高阶API开启MTE2 Preload](matmul_mte2_preload.md)：本案例呈现了在矩阵乘算子场景中，使用Matmul高阶API进行矩阵乘法计算，开启MTE2 Preload对算子性能的提升效果。通过MatmulConfig中的doMTE2Preload参数开启矩阵M或N方向的预加载功能，预加载即在MTE2间隙提前加载A矩阵/B矩阵数据，开启预加载功能后，可以减少MTE2间隙，提升算子性能。doMTE2Preload参数的详细介绍请参考MatmulConfig。
- [Matmul高阶API开启多核K轴错峰访问内存](matmul_multi_core_k_stagger.md)：多核执行Matmul时开启K轴错峰访问内存，使不同核从不同起始地址访问数据，缓解地址冲突。
- [Matmul高阶API开启多核切K](matmul_multi_core_split_k.md)：当M、N轴较小无法有效切分时，开启多核切K功能实现K轴切分，使算子多核并行执行。
- [Matmul高阶API开启NBuffer33模板](matmul_nbuffer33.md)：MTE2 Bound场景下开启NBuffer33模板，将A矩阵切分为3x3基本块错开搬运流水，平衡MTE2与FixPipe数据流量。
- [Matmul性能优化策略总览](matmul_optimization_overview.md)：本节提供了一系列包含Matmul计算的算子性能调优案例，开发者可根据实际应用场景，参考相关案例中的优化方法和思路，应用于具体实践中。案例分为如下五类，各分类的简介请参见如下表格，详细内容请阅读后续章节。
- [Matmul高阶API开启纯Cube模式](matmul_pure_cube.md)：本案例呈现了在矩阵乘算子场景中，开启Matmul高阶API的纯Cube模式对算子性能的提升效果。如下图所示，Matmul API默认使用MIX模式，即用户从AIV侧发起消息，通过消息通信框架中转消息后，在AIC侧执行Matmul计算。这套消息处理机制会带来额外的Scalar性能开销。相较于MIX模式，纯Cube模式可以直接跳过消息通信框架，完成Matmul计算，提升算子性能。
- [Matmul高阶API开启Tiling全量常量化](matmul_tiling_constant.md)：将Matmul Tiling参数常量化到编译期，减少运行时Scalar计算开销，降低指令头开销并避免阻塞MTE2流水。
- [Matmul算子优化Tiling策略](matmul_tiling_strategy.md)：本案例对Matmul算子进行性能分析和优化。Matmul算子实现的功能是矩阵乘法，其中主要包含数据搬入和搬出流水，Cube计算流水。
- [Matmul高阶API开启UnitFlag](matmul_unit_flag.md)：开启UnitFlag为MMAD与FIXPIPE提供512B细粒度同步，实现计算与搬出流水并行，提升算子性能。

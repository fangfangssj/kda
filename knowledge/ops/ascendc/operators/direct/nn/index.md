# 神经网络直调算子

按 ops-nn 算子仓目录组织神经网络类 Direct Invoke 知识。

- [激活](activation/index.md)：提供 Softmax、SwiGlu 等激活算子。
- [卷积](conv/index.md)：提供二维、三维反向滤波和深度卷积算子。
- [Foreach](foreach/index.md)：提供张量列表逐元素融合算子。
- [索引](index/index.md)：提供 Gather、Scatter 等索引算子。
- [损失](loss/index.md)：提供交叉熵等融合损失算子。
- [矩阵乘](matmul/index.md)：提供量化矩阵乘算子。
- [归一化](norm/index.md)：提供 RMSNorm 及融合动态量化算子。
- [优化器](optim/index.md)：提供 AdamW 等参数更新算子。
- [池化](pooling/index.md)：提供自适应池化与形态学池化算子。
- [量化](quant/index.md)：提供反量化、激活与量化融合算子。
- [循环神经网络](rnn/index.md)：提供 GRU、LSTM 的多阶段直调实现。

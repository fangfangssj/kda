# 比较与选择

- [Compare](compare.md)：逐元素比较两个RegTensor大小，根据比较模式将结果写入MaskReg对应比特位，支持EQ/GT/LT/GE/NE/LE模式。
- [Compares](compares.md)：逐元素比较RegTensor中每个元素与标量的大小关系，结果为真则MaskReg对应比特位置1。
- [比较与选择概述](overview.md)：比较与选择支持以下接口：
- [Select](select.md)：根据mask比特位值从两个源操作数中选取元素，比特位为1选srcReg0，为0选srcReg1。
- [Squeeze](squeeze.md)：将srcReg中被mask选择的有效元素依次复制到dstReg中连续排列，剩余位置置0。

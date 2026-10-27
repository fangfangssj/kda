# MaskReg计算

- [DeInterleave](de_interleave.md)：将源操作数src0和src1中的元素解交织存入目的操作数dst0和dst1中。
- [Interleave](interleave.md)：将源操作数src0和src1中的元素交织存入目的操作数dst0和dst1中。
- [Move](move.md)：将src中的元素复制到dst中，有mask时仅复制有效元素，无效元素填0。
- [MoveMask](move_mask.md)：从SetVectorMask设置的掩码寄存器中读取Mask值，按模板参数T对应格式转换后写入MaskReg。
- [Pack](pack.md)：根据低位或高位模式，将源操作数src的偶数位bit提取到dst的低半部分或高半部分。
- [Select](select.md)：根据mask的比特位值从src0或src1中选取元素，bit为1选src0，bit为0选src1。
- [UnPack](un_pack.md)：根据低位或高位模式，将src的低半部分或高半部分展开到dst，每bit展开为2bit。

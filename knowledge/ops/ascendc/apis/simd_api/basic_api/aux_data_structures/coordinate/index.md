# Coordinate

- [Coordinate简介](coordinate_intro.md)：Coordinate本质上是一个元组（tuple），用于表示张量在不同维度的位置信息，即坐标值。Coordinate（坐标）和Layout（布局）、Index（内存位置索引）之间存在紧密的关联：
- [Crd2Idx](crd2_idx.md)：将多维坐标通过布局(Layout)转换为内存位置索引，Layout包含Shape和Stride信息。
- [MakeCoord](make_coord.md)：将传入的数据打包成Coord数据结构。

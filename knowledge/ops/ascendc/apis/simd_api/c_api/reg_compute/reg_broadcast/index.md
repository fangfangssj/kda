# 广播操作

- [asc_duplicate](asc_duplicate.md)：将源操作数src的最低位元素（即下标为0的元素）广播到目的操作数dst中被mask筛选的位置。mask仅用于筛选dst中的元素位置，不影响从src中读取的元素；dst中未被mask筛选的位置置零。
- [asc_duplicate_highest](asc_duplicate_highest.md)：将src的最高位元素广播到dst中被mask筛选的位置，dst中未被mask筛选的位置被置为0。
- [asc_duplicate_highest_merge](asc_duplicate_highest_merge.md)：merge模式下，将src的最高位元素广播到dst中被mask筛选的位置，dst中未被mask筛选的元素保留原值。
- [asc_duplicate_merge](asc_duplicate_merge.md)：merge模式下，将src的最低位元素广播到dst中被mask筛选的位置，dst中未被mask筛选的元素保留原值。
- [asc_duplicate_scalar](asc_duplicate_scalar.md)：将标量值广播到目的操作数中。支持带输出参数和带返回值两种接口形式，每种形式均支持以下两类重载：
- [asc_duplicate_scalar_merge](asc_duplicate_scalar_merge.md)：merge模式下，将标量值value广播到dst中被mask筛选的位置，dst中未被mask筛选的元素保留原值。

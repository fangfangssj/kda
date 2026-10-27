# 排序计算

- [asc_bitsort](asc_bitsort.md)：位图排序函数，一次迭代可以完成32个元素的排序，数据按如下描述结构进行保存：
- [asc_get_vms4_sr](asc_get_vms4_sr.md)：此接口用于获取执行asc_mrgsort4操作后的4个队列中，每个队列已经理过的Region Proposal个数。使用时，需要将asc_mrgsort4中的if_exhausted_suspension参数配置为true，该配置模式下某条队列耗尽后，MrgSort4指令停止。
- [asc_mrgsort4](asc_mrgsort4.md)：将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序。

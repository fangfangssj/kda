# 排序组合（ISASI）

- [GetMrgSortResult](get_mrg_sort_result.md)：与MrgSort接口配合使用，在队列耗尽停止后获取4个队列已处理的Region Proposal个数。
- [MrgSort](mrg_sort.md)：将已排序的最多4条队列合并为1条队列，结果按score域由大到小排序，通常处理Sort32预处理的数据。
- [MrgSort4（排序与合并）](mrg_sort4.md)：将已经排好序的最多4条Region Proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。
- [排序与合并概述](overview.md)：Region Proposal（简称RP）是目标检测算法中用于定位潜在目标区域的候选框。Region Proposal采用特定的数据结构进行存储和管理，以支持高效的排序和筛选操作。在业务场景下，为了从大量Region Proposals中筛选出最有可能包含目标的区域，需要根据每个RP的置信度score进行排序。
- [ProposalConcat](proposal_concat.md)：将连续元素合入Region Proposal内对应位置，每次迭代会将16个连续元素合入到16个Region Proposals的对应位置里。 Region Proposal说明： 目前仅支持两种数据类型：half、float。 每个Region Proposal占用连续8个half/float类型的元素，约定其格式：
- [ProposalExtract](proposal_extract.md)：与ProposalConcat功能相反，从Region Proposals内将相应位置的单个元素抽取后重排，每次迭代处理16个Region Proposals，抽取16个元素后连续排列。
- [RpSort16（排序与合并）](rp_sort16.md)：根据Region Proposals中的score域对其进行排序（score大的排前面），每次排16个Region Proposals。
- [Sort32](sort32.md)：一次迭代内对32个数按score域降序排列，结果以score和index的结构存储在目的张量中。

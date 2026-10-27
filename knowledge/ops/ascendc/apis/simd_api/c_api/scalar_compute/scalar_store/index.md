# 数据搬出

- [asc_store_dev](asc_store_dev.md)：不经过DCache向GM地址上写数据。 当多核操作GM地址时，如果数据无法对齐到Cache Line，经过DCache的方式下，由于按照Cache Line大小进行读写，会导致多核数据随机覆盖的问题。此时，可以采用不经过DCache直接读写GM地址的方式，从而避免上述随机覆盖的问题。

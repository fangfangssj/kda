# 数据搬入

- [asc_load_dev](asc_load_dev.md)：不经过DCache直接从GM地址读取整型数据。 当多核操作GM地址且数据无法对齐到Cache Line时，经过DCache读写可能引入Cache Line粒度的数据覆盖。此时，可使用本接口绕过DCache读取GM数据。

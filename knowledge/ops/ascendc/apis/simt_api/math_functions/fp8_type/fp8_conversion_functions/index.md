# fp8数据转换函数

- [__asc_cvt_float2_to_fp8x2](_asc_cvt_float2_to_fp8x2.md)：将float2的两个分量按CAST_RINT模式转换为FP8类型（e4m3或e5m2）的8位浮点数，支持饱和与非饱和溢出处理，结果以位级打包存储。
- [__asc_cvt_float_to_fp8](_asc_cvt_float_to_fp8.md)：将float类型数据按照CAST_RINT模式转换为指定类型（float8_e4m3_t和float8_e5m2_t）的8位浮点数，并根据指定的饱和模式（饱和或非饱和）进行溢出处理。转换结果以位级形式存储为__asc_fp8_storage_t类型，该类型为unsigned char，用于存储float8_e4m3_t或float8_e5m2_t类型的数据。
- [__e4m3x22float2](_e4m3x22float2.md)：将float8_e4m3x2_t类型数据的两个分量转换为浮点数，返回转换后的float2类型数据。
- [__e5m2x22float2](_e5m2x22float2.md)：将float8_e5m2x2_t类型数据的两个分量转换为浮点数，返回转换后的float2类型数据。
- [__float22hif82_rh](_float22hif82_rh.md)：将float2类型数据的两个分量遵循CAST_HYBRID模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
- [__float22hif82_rh_sat](_float22hif82_rh_sat.md)：饱和模式下，将float2类型数据的两个分量遵循CAST_HYBRID模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
- [__float22hif82_rna](_float22hif82_rna.md)：将float2类型数据的两个分量遵循CAST_ROUND模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
- [__float22hif82_rna_sat](_float22hif82_rna_sat.md)：饱和模式下，将float2类型数据的两个分量遵循CAST_ROUND模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
- [__half22hif82_rh](_half22hif82_rh.md)：将half2类型数据的两个分量遵循CAST_HYBRID模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
- [__half22hif82_rh_sat](_half22hif82_rh_sat.md)：饱和模式下，将half2类型数据的两个分量遵循CAST_HYBRID模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
- [__half22hif82_rna](_half22hif82_rna.md)：将half2类型数据的两个分量遵循CAST_ROUND模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
- [__half22hif82_rna_sat](_half22hif82_rna_sat.md)：饱和模式下，将half2类型数据的两个分量遵循CAST_ROUND模式转换为hifloat8_t精度，返回转换后的hifloat8x2_t类型数据。
- [__hif822float2](_hif822float2.md)：将hifloat8x2_t类型数据的两个分量转换为浮点数，返回转换后的float2类型数据。
- [__hif822half2](_hif822half2.md)：将hifloat8x2_t类型数据的两个分量转换为半精度浮点数，返回转换后的half2类型数据。

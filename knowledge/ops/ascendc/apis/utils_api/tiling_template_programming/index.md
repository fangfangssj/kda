# Tiling模板编程

- [ASCENDC_TPL_SEL_PARAM](ascendc_tpl_sel_param.md)：Tiling模板编程时，开发者通过调用此接口自动生成并配置TilingKey。 使用该接口需要包含定义模板参数和模板参数组合的头文件。详细内容请参考Tiling模板编程。
- [GET_TPL_TILING_KEY](get_tpl_tiling_key.md)：Tiling模板编程时，开发者通过调用此接口自动生成TilingKey。该接口将传入的模板参数通过定义的位宽，转成二进制，按照顺序组合后转成uint64数值，即TilingKey。 使用该接口需要包含定义模板参数和模板参数组合的头文件。详细内容请参考Tiling模板编程。
- [模板参数定义](template_params_definition.md)：通过以下函数原型进行模板参数ASCENDC_TPL_ARGS_DECL和模板参数组合ASCENDC_TPL_ARGS_SEL（即可使用的模板）的定义。详细内容请参考Tiling模板编程。

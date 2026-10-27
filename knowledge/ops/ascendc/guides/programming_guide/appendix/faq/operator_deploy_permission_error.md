---
type: Guide
title: 算子包部署时出现权限不足报错
description: 部署自定义算子包时，出现如下报错信息：
tags:
- task.lookup
- topic.implementation
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/programming_guide/appendix/faq/operator_deploy_permission_error.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 算子包部署时出现权限不足报错<a name="ZH-CN_TOPIC_0000002096022474"></a>

## 现象描述<a name="section151611254194612"></a>

部署自定义算子包时，出现如下报错信息：

```
[WARNING] The directory /usr/local/Ascend/cann/opp does not have sufficient permissions. Please check and modify the folder permissions (e.g., using chmod), or use the --install-path option to specify an installation path and source the generated set_env.bash.
...
[ERROR] create /usr/local/Ascend/cann/opp/vendors/customize/framework failed
```

## 问题根因<a name="section417961104715"></a>

当前操作用户缺少对部署路径下vendors目录的写权限。

自定义算子包默认安装路径`${INSTALL_DIR}/opp/vendors`的目录权限与CANN软件包安装用户和安装配置有关：root用户安装CANN，`${INSTALL_DIR}/opp/vendors`权限为755；非root用户携带--install for all参数安装CANN，该目录权限为755，非root用户不带--install for all参数安装CANN时，该目录权限为750。

例如，root用户安装CANN软件包后，HwHiAiUser属组用户在对应目录部署自定义算子包，因为其他用户没有写权限，会出现上述报错信息，提示权限不足导致自定义算子包部署失败。

## 处理步骤<a name="section12149153520416"></a>

-   方法一（推荐）：使用--install-path参数指定用户自有且权限受控的安装目录（参考[指定目录安装](../../advanced_programming/aclnn_operator_development/compilation_and_deployment/basic_process.md#custom-install-path)）。部署用户需要对指定的安装路径有读、写和执行权限。安装后执行生成的`set_env.bash`脚本，由该脚本配置算子运行时查找路径。

    ```
    ./custom_opp_<target os>_<target architecture>.run --install-path=<path>
    source <path>/vendors/<vendor_name>/bin/set_env.bash
    ```

-   方法二：如需部署到默认安装路径，请联系CANN软件包安装用户或管理员，由其统一部署自定义算子包。确需授权其他用户部署时，应遵循最小权限原则，仅向可信用户授予必要权限。

    **安全提示：**请勿将共享的vendors目录权限设置为777，也不要对该目录递归开放所有用户的写权限。父目录写权限可能允许非授权用户新增或重命名算子目录、替换加载配置，影响其他用户的算子加载。运行用户仅需使用算子包时，不应授予部署目录的写权限。

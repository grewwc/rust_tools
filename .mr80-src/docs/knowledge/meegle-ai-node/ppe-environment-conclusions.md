---
title: Aeolus PPE 平台契约与验证结论
description: 应用鉴权、BITS 构建、镜像配方、ENV 部署及验收边界。
last_verified: 2026-11-05
---

# Aeolus PPE 平台契约与验证结论

应用身份已完成固定 SHA 的 BITS/SCM 制品构建、ICM 镜像构建、任务专属 PPE 双集群首次部署及第二 SHA 普通升级。两个集群实际版本、依赖、规格及 Pod healthz 已独立回查；浏览器业务路由和本轮真实删除回查仍未验收。

实现与恢复规则见 [运行约定](ppe-deployment-implementation.md)，完整产品验收见 [端到端验收](ppe-product-e2e-validation.md)。本文保留可复用的接口结论；具体运行、资源及失败请求台账由执行记录管理，不作为通用部署输入。

## 应用身份与权限

平台使用 ByteCloud AK/SK 经官方 SDK 换取应用 JWT：AK/SK 在 staging/prod 从同一个 KMS alias `conciliator/credentials/bytecloud_credentials` 读取 JSON 对，dev 使用 `BYTECLOUD_AK`、`BYTECLOUD_SK`。JWT 通过 `X-Jwt-Token` 发送；ByteCloud CN 网关为 `https://bc-cn-gw.bytedance.net`，业务平台由 `domain` 指定。密钥不进入业务请求或部署记录。

[SdkMaatTokenProvider](../../../src/conciliator/integrations/maat/auth.py) 负责 SDK 凭据及 JWT 刷新，HTTP 客户端限制凭据发送到指定 HTTPS origin 并拒绝重定向。

认证主体与请求操作人分别处理。当前 BITS `Username`、ICM body `username` 为经授权的 `huaguojin`，由服务端固定，不能从客户端任意指定；这不等于使用个人 JWT，也不授予资源权限。曾把服务账号 ID 填入 ICM `username` 导致 SCM 下载 403；修正后构建成功，但缓存使全部依赖的独立下载权限仍未充分验证。

ENV 接口没有声明 `username`，代理字段为 `X-Real-User-Token`；当前实现仅使用应用 JWT。个人 CLI 登录态仅用于人工诊断和文档查询。

| 授权范围 | 核对方式 |
| --- | --- |
| API 接入 | 在服务账号详情核对具体平台 API 接入权限；可调用接口不等于可操作资源 |
| PPE 操作 | 目标 PSM 的 `tce.operator_ppe.cn` 及工单权限，与 ENV 管理员权限分别核对 |
| 来源敏感配置 | `tce.cluster.query_sensitive_data`；读取线上配置时按角色详情核对 `tce.viewer`，环境变量仅驻内存 |
| 资源创建 | 实际路径涉及时核对 `groot.resource.manager.cn`；已存在 PSM 可能要求直接 owner，父节点继承不能替代 |
| SCM/代码仓 | 主仓构建、各依赖仓下载及代码访问分别核验；成功的缓存构建不能替代 |

申请通过 [IAM 资源入口](https://cloud.bytedance.net/iam/acls/list?x-resource-account=public&x-bc-region-id=bytedance) 选择实际 PSM/节点、角色和服务账号授权对象，审批后使用同一后台身份核验。不能根据 `inf.hae.boe` 名称或个人 dry-run 成功推断后台所需权限。

直接 SCM `POST /api/v2/versions/cicd_create` 的应用请求曾返回 HTTP 200/code 0，但业务 `state=failed` 且无 version ID，提示无发布权限；补操作人未解决。新请求走已验证的 BITS 正式 OpenAPI，直接 SCM adapter 仅供历史恢复。SCM 接入需向 Oncall 核对服务账号注册、白名单及操作人仓库发版权限，必要时使用 [BPM 33104](https://bpm.bytedance.net/apply?cid=33104)；现有错误不能断言被拒的是应用还是操作人。

## BITS 构建契约

`BitsScmClient` 使用应用 JWT 与真人 `Username`，正式写接口为：

```text
POST https://bits.bytedance.net/api/v1/pipelines/open/{pipeline_id}/run
GET  https://bits.bytedance.net/api/v1/pipelines_open/{pipeline_id}
GET  https://bits.bytedance.net/api/v1/pipelines_open/runs/{run_id}
```

- 当前固定 SCM-only 流水线为 `1251261261058`；body 使用 `pipeline_id, custom_vars, group_id, group_version`，变量 `custom.commit_hash` 的 `value.text` 为完整 SHA。
- 当前变量组和版本由流水线详情获取，不硬编码旧版本。
- 预检必须只有一个 `job_atoms/scm_compile_beta`，仓库为 `data/dp/aeolus_ppe`，`pub_base=commit_base`，`revision={{custom.commit_hash}}`；含其他构建/TCE 步骤的流水线不能替代。
- 原运行查询返回 `pipeline_run` 及 snake_case 字段。运行成功后仍独立核验其 SCM version ID、SHA、类型、架构与 `build_ok`，不能读取仓库 latest 代替精确版本详情。
- Console `/api/v1/pipelines/{id}/run` 与 `X-PLATFORM-PROXY-USER` 的失败不能推广到上述正式 OpenAPI。

## ICM 镜像配方

SCM 制品本身不直接产出容器镜像。完整配方以 [aeolus_cn.json](../../../src/conciliator/integrations/ppe/aeolus_cn.json) 为准，不复制高层 ENV CLI 的默认主仓、Python 2 配方或资源池。

| 项目 | 固定约定 |
| --- | --- |
| 代码/主制品仓 | `dp/aeolus` / `data/dp/aeolus_ppe`（SCM repo `572600`），主版本对应本轮 SHA |
| PSM/镜像架构 | `data.aeolus.api_v3` / `x86_64` |
| 基础镜像 | `debian.bookworm.python310:latest`；核验实际解析版本 |
| 启动脚本/根目录 | `toutiao/app/data/aeolus/dlv_control/bootstrap.sh` / `/opt/tiger` |
| 包安装 | 保留完整系统包及 pip3 配置，包含 `python-dev-is-python3`；旧 `python-dev` 在该基线无安装候选 |
| 区域/标签 | `China-North-LF` / `ppe` |

成功配方参考 ICM `1.0.1.7372`，固定 11 项 SCM 依赖及路径，仅替换主仓制品。其他成功镜像若主仓、启动脚本或 manifest 不同，不能整份复制。

ICM `repos[].remote_id` 为字符串，`packages[].package_type` 指定安装器，`arch_map` 为对象；实际创建响应为列表。目录文档曾标注不同类型，解析失败或超时先对账精确版本，不能重复提交。

## ENV/TCE 接口与标识

以下路径相对 ByteCloud CN 网关：

| 能力 | 接口 | domain/约束 |
| --- | --- | --- |
| 精确 SCM 版本 | `GET /api/repos/572600/versions/{version}` | `scm;v1`，读取指定版本 detail |
| ICM 构建 | `POST /api/v2/images/TCE/build_multi_images/` | `icm;v1` |
| 精确镜像 | `GET /api/v2/images/TCE/version_detail/?name={psm}&version={version}` | `icm;v1`，核验 LF ok 和配方 |
| ENV 预检 | `POST /api/v4/dsl/pre_check/common?standard_env=online_cn` | `env_platform;v1`，is_break=true 拦截 |
| ENV 部署/升级 | `PUT /api/v4/dsl/env/{name}?type=ppe&standard_env=online_cn` | `env_platform;v1`，action 与 meta 需匹配 |
| ENV 工单 | `GET /api/v4/dsl/deployment/{id}` | `env_platform;v1`，保存 id_str 避免整数精度丢失 |
| 来源集群 | `GET /api/v4/suggest/cluster_by_id?cluster_id={id}&env_type=ppe&standard_env=online_cn` | `env_platform;v1`，匹配 ID/名称/PSM，new_env_vars 不落盘 |
| 服务/集群归属 | `GET /api/v4/instance_meta?env={name}&env_type=ppe&service={psm}&service_type=tce&detail=true` | `env_platform;v1`，instance_meta ID、TCE service ID、cluster ID 不可混用 |
| TCE 工单 | `GET /open-apis/v1/deployment/{id}/` | `tce;v1` |
| 实例核验 | `GET /open-apis/v1/services/{service_id}/clusters/{cluster_id}/`、`GET /open-apis/v1/clusters/{cluster_id}/instances/`、`GET /open-apis/v1/pods/{pod_name}/` | `tce;v1`，独立检查版本与健康 |

来源参考环境为 `ppe_aeolus_huaguojin_test`，`online` / `query_online_mirror` 来源 ID 为 `219404613` / `219404612`。来源 ID 仅用于配置读取及 `base_cluster_id`，不作为任务资源 ID。目标为 LF / `PPE/ipv6`，每集群 8 CPU / 32 GiB / 1 实例；参考环境的 VQS 不属于当前部署目标。

### ENV 版本升级的镜像字段

创建/补缺集群在 `meta.clusters[]` 各集群配置内指定 `image_tag`、`image_version`，并提供来源、位置、资源和启动配置。升级则将共享镜像放在服务 meta，集群仅传数字 ID：

```json
{
  "action": "upgrade",
  "type": "tce",
  "psm": "data.aeolus.api_v3",
  "meta": {
    "image_tag": "tce",
    "image_version": "<已核验成功版本>",
    "clusters": [{"id": 220397362}, {"id": 220397363}],
    "enable_dsd": false
  }
}
```

示例 ID 仅说明类型，不能用于其他任务；上例省略完整 `scm_repos` 和外层，不可直接提交。实测升级时把镜像放进每个集群，即使 ID 为数字也报 `cluster_list item 0 should provide id or name`；改成服务 meta 后双集群升级通过。

`GET /api/v4/service/{psm}/image/{version}?env=ppe` 曾因 manifest 的 `remote_id` 数字/字符串解析冲突失败；显式指定成功镜像的部署路径已通过，该查询失败不能直接推断镜像不可用。

## 完成判断与验收边界

预检返回非拦截的 `param invalid no need check` 或 `ClusterNotFound` 表示部分检查未执行，不能记为配额检查通过。

就绪需结合 ENV Succeed、TCE finished、目标集群 running、实际镜像及依赖匹配、实例规格和直连健康接口。`meta.status=need_cluster` 不单独判失败。健康响应没有证明浏览器携带 `X-TT-ENV`、未回落到基线或下游命中 PPE；部署完成、入口待验证、人工业务验证分别表达。

同任务普通升级与明确失败后的制品复用重试已验证。完整产品操作、浏览器/下游路由、本轮真实回收、热部署、运行中部分失败恢复和全部依赖无缓存下载仍按 [验收指南](ppe-product-e2e-validation.md) 补验。

## 官方来源

- [BITS OpenAPI](https://cloud.bytedance.net/docs/bits/docs/6480709f2207580224c6b2d2/659661b732f88702fa48a58e)、[接口样例](https://cloud.bytedance.net/docs/bits/docs/6480709f2207580224c6b2d2/67d29ca7e7fd5a0511dd3bc6)。
- [ENV 更新](https://cloud.bytedance.net/api/v2/openapi/llms/env_platform/v1/api/a0637178c1b99fbdd87fb7bbdcc26132e679945e.md)、[预检](https://cloud.bytedance.net/api/v2/openapi/llms/env_platform/v1/api/1bd1087649712ee2350e197b993a982cf53aa9ff.md)、[TCE 实例查询](https://cloud.bytedance.net/api/v2/openapi/llms/tce/v1/api/b1a1b3d4b9c11b538ab844edff635049713bb160.md)。
- [SCM 发布要求](https://cloud.bytedance.net/docs/scm/docs/63a91ebec127d00222697c9c/63b785ba881600021d8f387a)、[服务账号管理](https://cloud.bytedance.net/api/v1/cloud_developer/docs/console/cn/63c4cbbace57ff0223cc0786.md)、[IAM 申请](https://cloud.bytedance.net/api/v1/cloud_developer/docs/console/cn/68d27490abf01204ebcb06c8.md)。
- [TCE 接入](https://cloud.bytedance.net/api/v1/cloud_developer/docs/tce/cn/63ad8cc6f2894002294ddbce.md)、[权限](https://cloud.bytedance.net/api/v1/cloud_developer/docs/tce/cn/63ad87accaaad3021d15363a.md)、[敏感配置](https://cloud.bytedance.net/api/v1/cloud_developer/docs/tce/cn/6a82b7b18d7c430503133481.md)、[资源创建授权](https://cloud.bytedance.net/api/v1/cloud_developer/docs/console/cn/67220d2441dc5e03216131ff.md)。
- [ENV 隔离与请求标识](https://cloud.bytedance.net/api/v1/cloud_developer/docs/bits/cn/650964c5de579c02f3e968b7.md)。

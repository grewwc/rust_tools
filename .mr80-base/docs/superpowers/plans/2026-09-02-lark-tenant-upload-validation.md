# 飞书应用身份上传 Markdown 验证方案

## 目标

本轮只验证一件事：Conciliator 使用自建应用的 `tenant_access_token`，调用飞书
Drive OpenAPI，把一个原生 Markdown 文件上传到“Conciliator”共享文件夹。

- 应用：`cli_aa1d2aa842385bee`
- 文件夹 token：`K0zefVjqUlsicAdu4TjccNhjnjh`
- 接口：`POST /open-apis/drive/v1/files/upload_all`
- 认证：`Authorization: Bearer <tenant_access_token>`

本轮不验证下载、内容回读、SHA-256、L2、文件覆盖，也不串联
`/resolve -> /reject -> retry`。

## 代码入口

- `src/conciliator/integrations/lark/client.py`：获取并缓存
  `tenant_access_token`，以及发送带认证头的请求。
- `src/conciliator/integrations/lark/drive.py`：组装
  `multipart/form-data`，调用 `upload_all`。
- `tests/fixtures/lark-upload-validation.md`：固定的验证文件。
- `scripts/verify_lark_upload.py`：可重复执行的真实 OpenAPI 验证入口。
- `tests/unit/test_lark_drive.py`：验证 token 交换、缓存和 multipart 请求契约。

生产部署不保存长期 tenant token。服务从 KMS 读取 App Secret，在运行时换取
`tenant_access_token` 并按有效期缓存。本地联调时可以临时把 API Explorer 中的
tenant token 注入脚本进程，验证结束后立即清掉。

## 本地复用方法

先运行不访问飞书的契约测试：

```bash
.venv/bin/python -m pytest tests/unit/test_lark_drive.py -q
```

再从飞书 API 调试台复制当前应用的 `tenant_access_token`。不要把 token 写进仓库、
`.env`、命令行参数、测试输出或日志。可以在终端里静默读取，再导出给当前 shell：

```bash
read -s LARK_TENANT_ACCESS_TOKEN
export LARK_TENANT_ACCESS_TOKEN
.venv/bin/python scripts/verify_lark_upload.py
unset LARK_TENANT_ACCESS_TOKEN
```

脚本会生成带 UTC 时间戳的文件名，形如
`conciliator-lark-upload-20260902-093000.md`。成功时输出 `file_token`、文件名和
字节数，但不会输出 tenant token 或 App Secret。上传请求不会自动重试，避免在
响应丢失时意外产生重复文件；需要重跑时使用新的时间戳文件名。

上传成功后，用 Chrome 打开共享文件夹，确认同名 Markdown 文件已经出现：

<https://bytedance.larkoffice.com/drive/folder/K0zefVjqUlsicAdu4TjccNhjnjh>

## 2026-09-02 验证记录

已完成：

1. 飞书开发者后台显示 `drive:file:upload` 已按“应用身份”开通，当前修改均已发布。
2. API Explorer 当前处于 `tenant_access_token` 模式。
3. token 只注入本地验证进程，没有写入文件或打印到输出。
4. 单元测试确认请求使用 `multipart/form-data`，字段为 `file_name`、
   `parent_type=explorer`、`parent_node`、`size` 和文件二进制内容。
5. 第一次真实请求到达飞书 Drive 后，返回
   `HTTP 403 / code 1061004 / forbidden`。
6. 目标文件夹为“仅协作者可访问”；在共享面板中搜索 `Conciliator`，可以找到对应
   智能体。点击搜索结果会直接加入协作者，没有二次确认。
7. 应用成为文件夹协作者后，使用同一个 API Explorer tenant token 重跑成功：
   - 文件名：`conciliator-lark-upload-20260902093637.md`
   - file token：`W8K0bH28EoF0uexlsOClmrKegoo`
   - 文件大小：377 字节
8. Chrome 已在目标文件夹中看到同名 `.md` 文件，作者显示为 `Conciliator`。

结论：`tenant_access_token + drive:file:upload + 文件夹协作者权限` 可以跑通
`upload_all`。第一次 `403` 的直接原因是应用身份没有目标文件夹的访问权；成为
协作者后，无需 user token 即可上传。

## 验收标准

- 契约测试通过。
- 真实上传命令退出码为 0，响应包含非空 `file_token`。
- Chrome 中能在目标文件夹看到脚本输出的同名 `.md` 文件。
- 仓库、进程日志和测试输出中没有 App Secret 或 tenant token。

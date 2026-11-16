# Conciliator 飞书上传验证

这份文件用于验证 Conciliator 能否通过应用身份调用飞书 OpenAPI，并把原生 Markdown 文件上传到指定的共享文件夹。

- 鉴权方式：`tenant_access_token`
- 上传接口：`POST /open-apis/drive/v1/files/upload_all`
- 文件类型：原生 `.md` 文件

看到这份文件，说明上传请求已经成功。

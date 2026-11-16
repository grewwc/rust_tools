# 本地测试环境 TODO

记录日期：2026-11-09。

## 飞书用户授权入口与测试回调

状态：**暂停实施，等待后续安排**。本次只记录 TODO，不修改卡片、`.env` 或远端配置。

现象：文档评估提示应用无法读取文档、用户尚未授权，但卡片没有可点击的授权入口。

已确认：

- 后端 `/api/aeolus/tasks/by-node` 已提供 `authorization_required` 和 `lark_authorized`。
- 当前卡片尚未使用这些字段展示授权操作。
- 本地配置的 `LARK_OAUTH_REDIRECT_URI` 仍为生产地址 `https://aeolus-conciliator.byted.org/oauth/lark/callback`。
- 后端已有 `/oauth/lark/start`、`/oauth/lark/callback`，以及授权成功后恢复被授权阻断的评估的逻辑。

待办：

- [ ] 在确实因文档授权被阻断时显示“去飞书授权”，避免把普通文档错误或临时故障都归类为未授权。
- [ ] 补齐卡片的授权状态类型和链接生成/获取契约，使用任务操作者的 `user_key`。
- [ ] 通过用户点击打开授权页，授权成功后继续查询任务状态并显示恢复进度。
- [ ] 确定浏览器可访问的测试后端地址，使授权起点、回调与 state cookie 保持同一站点，并核对 HTTPS/浏览器限制。
- [ ] 同时核对 Agent 引用文档代理地址；当前代理 origin 从 `LARK_OAUTH_REDIRECT_URI` 推导，生产回调会使本地任务的引用文档请求也访问生产。
- [ ] 将本地 OAuth 回调配置和测试飞书 App 后台登记地址统一为测试地址。
- [ ] 验证未授权、授权取消、授权失效、身份不匹配，以及应用身份已经能读文档的情况。
- [ ] 验证授权后自动继续评估，不重复发送评估请求，不自动进入 Coding。
- [ ] 构建并检查测试卡片；实际发布与飞书后台配置变更留待恢复实施时处理。

相关实现：

- `frontend/src/features/ai_node_web_aeolus_card/App.tsx`
- `src/conciliator/api/aeolus.py`
- `src/conciliator/api/oauth.py`
- `src/conciliator/services/lark_user_token_service.py`
- `src/conciliator/services/aeolus_service.py`

验收目标：应用身份无权读取测试文档时，用户可从卡片完成测试 App 授权；回调落到测试后端，随后原任务继续评估。

## 部署测试配置收口

状态：**worker 环境及启动入口已完成，其余待办保留**。检查依据见[2026-11-09 配置检查记录](local-test-configuration-review-2026-11-09.md)。

- [x] 补齐 `deployment-worker/.venv` 的 v2 模块，使用专用环境启动并验证注册、心跳和领取轮询。
- [x] 保存 IDE Run 的 `deployment-worker` 入口；重启后确认已有 VQS 实例保持原进程和端口，健康及监听归属检查通过。
- [ ] 验证数据库、后端和 worker 的整体冷启动；本次只重启了 worker。
- [ ] 核对测试卡片构建 profile 与产物 API 地址；当前根工程生成配置是生产 origin，`npm run build` 默认也使用 prod。
- [ ] 只读核验远端测试 AI 节点 webhook、测试卡片实际版本及 API origin，再确定是否需要变更。
- [ ] 使用测试 App 验证实际需求文档、引用文档读取，以及默认 Markdown 上传目录权限；需要时改用测试目录。
- [ ] 使用新的测试 Coding 提交完成部署、MAAT 业务页面验证及释放恢复；已有 VQS 本机健康不能代替全链路验收。

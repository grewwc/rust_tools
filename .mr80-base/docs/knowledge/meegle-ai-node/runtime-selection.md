# PPE 与 Devbox 运行时选择

两种部署目标都展示服务端候选、推荐依据和手动选择框。客户端只提交候选 ID、配置 revision、
轮次和完整 SHA；不能指定任意解释器、PSM、构建配方或机器路径。

Meego 网页卡片与飞书消息中的 Review 卡片均提供运行配置选择。飞书 PPE 使用独立表单，
选项绑定候选 ID 和 revision，回调签名绑定候选集合与本轮 SHA；不兼容候选不提供提交入口。
Devbox 与 PPE 都只采用服务端明确推荐，证据不足时要求手选。重试沿用原部署配置。
已发送的消息不会自动修改；新通知或现有消息的有效回调会返回新版卡片。

## 识别依据

`RuntimeDetectionService` 使用已有 NextCode 应用身份的 `GetFile`，
在本轮 Coding 的精确 SHA 读取固定文件，不执行仓库中的脚本：

- `.python-version`：固定 Python major.minor；补丁版本不冒充 worker 的版本能力。
- `pyproject.toml` 的 `project.requires-python`、`setup.cfg` 的 `options.python_requires`，
  以及 `setup.py` 中静态字符串 `python_requires`：按 PEP 440 计算约束交集。
- 根 `requirements.txt` 中无条件的 `antlr4-python2-runtime` / `antlr4-python3-runtime`：
  已核实的 Python 2/3 依赖约束；不根据通用包版本猜 Python 小版本。
- `run/prod_run.sh` 中显式 `python2/python3 -m bytedunicorn`：展示启动提示。

基础分支、`base_branches` 和 `base_release_max` 只展示弱提示，不自动选定、不限制手选。
当前尚不递归读取 `-r` 引用文件、不解析动态 setup 声明、不查询依赖包的远端元数据。

2026-11-12 用 Codebase CLI 和后端应用身份的实际 `GetFile` 按精确 SHA 核验；
应用身份检查使用隔离内存台账，没有修改业务任务或触发部署：

| 仓库 / SHA | 根依赖证据 |
| --- | --- |
| Aeolus / `0340e0c863072607d731ce5ee48098bfbf68bae2` | `antlr4-python2-runtime==4.7.2` |
| Aeolus / `9de29907307a62714e3c5a8cc3ad80912be1ae8c` | `antlr4-python3-runtime==4.9.3` |
| VQS / `dd854438c71627b76419aabca52ade725437eb60` | `antlr4-python3-runtime==4.7.2` |

只有明确证据与部署候选形成唯一兼容匹配时默认推荐。多个 Python 3 小版本候选、
证据不足或约束冲突时不默认选择。强约束排除的候选在前端禁用，服务端仍会校验并拒绝。
弱提示可以被用户手选覆盖；未声明小版本的镜像只标记 Python 3，不宣称满足特定小版本约束。
Aeolus PPE 普通配方的基础镜像明确为 Python 3.10；VQS 普通基础镜像仅标明 Python 3。

## 保存与恢复

- 识别结果保存到任务 evaluation 的 `_runtime_detection`，包含仓库、完整 SHA、规则版本、
  规范化约束和安全依据；不保存文件正文或凭据。成功结果复用；读取失败最多每 60 秒重试，
  显示手选提示。仓库、SHA 或规则版本变化后重新读取，旧请求的晚到结果不能覆盖新 SHA。
- PPE 与 Devbox 的部署 data 冻结 `runtime_profile_id`、识别依据和配置版本。
  PPE 重试读取原部署的完整模板；不因当前推荐或服务端模板变化切换运行时。旧记录按原模板恢复。
- 前端将任务、轮次、SHA、机器、候选配置 revision 纳入选择范围；
  SHA 或配置变化后重新推荐，普通轮询保留手选。部署请求 ID 绑定所选候选及 revision。
- PPE 同任务环境更换配方须先释放旧环境。Devbox 仍按原规则先释放本任务旧实例。

## 验证

后端定向测试覆盖声明、依赖、未知/冲突、手选覆盖弱提示、强约束拒绝、SHA 变化、
迟到结果隔离、失败重读、两个目标共享依据、配置冻结和 PPE 更换运行时前释放。
在最新 main 的独立 worktree 上，本次部署/API/worker 范围共 372 项测试通过，
包含仅声明 Python 3 的镜像遇到补丁版本约束时的 6 项回归；前端 40 项测试通过。
开发 bundle、TypeScript、Ruff 和 diff 格式检查通过。
浏览器隔离验收使用构建产物和受控 API，在 1280/375 宽度检查两个选择框、真实点击请求参数和 SHA 重置，均通过。
这些检查不触发真实远端部署或卡片发布。

扩大至 AeolusService 的回归曾出现 `test_recovery_collects_blocked_run_without_starting_another_session`
的 `TaskLeaseLost`，False/True 参数均出现过。该测试未启用部署服务和本次识别器，
使用同一 SQLite StaticPool 连接；本次不修改其租约流程，不能宣称扩大回归全部通过。

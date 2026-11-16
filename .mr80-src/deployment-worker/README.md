# Conciliator deployment worker

运行在 Linux DevBox 上，通过 HTTP 领取任务、启动 Aeolus/VQS 并上报结果。
worker 需要 Python 3.11+、httpx、Git、可读取的 `/proc` 及业务运行账号的 doas 权限。
每个 v2 worker 连接一个后端，每次部署使用独立 checkout、进程组、动态端口和 journal。

接入顺序：准备 Devbox 和业务运行环境 → 安装 worker → 修改个人配置 → 只读预检 → 启动并在卡片验收。
接已有线上服务时无需在 Devbox 上运行 Conciliator 后端、MySQL 或执行数据库迁移。

## 接入前提

- Linux Devbox 能直接访问目标后端，并有足够的 CPU、内存和磁盘运行所选业务。
- 已安装 Python 3.11+、`uv`、Git、doas；`/proc` 可读，以便核对业务进程和监听端口的归属。
- Git 凭据可非交互读取需要部署的主仓及其全部子模块；worker 不接收或管理 Git token。
- 已开通所选业务的 doas 权限：Aeolus 使用 `data.aeolus.api_v3`，VQS 使用 `data.vqs.api`。
- MAAT 能访问 Devbox 的 `advertised_host` 和业务动态端口；仅 Devbox 能访问后端还不够。

## 下载本仓库

先将 Conciliator 代码下载到自己的 Devbox，worker 安装包位于本仓库的 `deployment-worker/`：

```bash
git clone git@code.byted.org:xiahaitao/Conciliator.git
cd Conciliator
```

已有仓库时先同步包含本次修复的版本；验证未合入的 MR 时检出对应 MR 分支。
后续命令均在本仓库根目录执行。业务源码由 worker 按任务提交自动拉取，无需把别人的业务 checkout 路径复制过来。

## 准备业务运行环境

接入示例使用 **Aeolus Python 3.10、VQS Python 3.7**；worker 自身仍使用独立的 Python 3.11+ 环境。
先按对应业务版本准备解释器、Python 包、SCM 运行库及本机库，部署过程不会安装业务依赖。
只保留自己需要且已准备好的 service/profile，避免卡片出现不可用选项。

环境路径统一写在个人 worker JSON 中，无需额外的 manifest、文件摘要清单或手工环境版本号。
示例中的 `/opt/runtimes/...` 都是占位路径，须替换为自己的实际路径，不会由安装命令创建。

### SCM parser 与依赖路径

`dependency_paths.aeolus_expr_parse` 指向准备好的 SCM parser **包目录本身**，
不是它的父目录或整个 SCM 产物目录。例如安装在 `/opt/tiger` 下时，填写真实的包路径，不要猜测目录名。
worker 将它复制到独立 checkout，跳过根目录下同名旧子模块；其他子模块仍使用部署提交固定的 SHA。
应包含对应业务需要的解析器模块，例如 `clickhouse_mini_common_expr/clickhouse_mini_common_expr/parser`。

`dependency_paths` 是“目录名 → 准备好的目录”的复制映射；需要其他运行库时按需增加映射。
示例中的 VQS 3.7 只指定 SCM parser；`aeolus_kms`、`sysconf`、`envmanager` 由该版本主仓的子模块提供，不重复配置。
示例保留 Aeolus/VQS 所需的 `extra_paths`。若部署的其他版本不再声明这些子模块，才补充对应运行库映射。
`extra_paths` 保留通过 Python 搜索路径加载的运行库，不能简单替换成 `dependency_paths`。
其他已由 Git 子模块提供的同名源码不会被运行库覆盖。运行库按版本保存，避免在部署或复制期间原地修改。

## 安装与配置

首次安装：

```bash
uv venv --python 3.11 deployment-worker/.venv
uv pip install --python deployment-worker/.venv/bin/python ./deployment-worker
umask 077
mkdir -p .deployment-worker
cp -n deployment-worker/examples/deployment_worker.v2.example.json .deployment-worker/worker.json
```

编辑复制出的 `.deployment-worker/worker.json`，以[配置示例](examples/deployment_worker.v2.example.json)为准。
接入只需核对以下信息：

| 配置 | 用途 |
| --- | --- |
| `protocol_version` | 保持示例中的 `2` |
| `api_base` | 后端地址；生产使用 `https://aeolus-conciliator.byted.org`，不附加 `/api` 或 webhook 路径 |
| `worker_id` | 自己的稳定机器身份；1～64 个英文字母、数字、下划线或短横线，不能与其他人的 worker 重复 |
| `advertised_host` | MAAT 可访问的 Devbox IP 或域名；这是业务监听地址，不是后端地址 |
| `python_path` / `python_version` | 每套 profile 的业务解释器和 major.minor 版本，实际解释器必须匹配；Aeolus 示例为 3.10，VQS 为 3.7 |
| `dependency_paths` | 对应业务的 SCM parser 等准备好的依赖目录 |
| `extra_paths` | 业务运行库的 Python 搜索路径；示例保留 Aeolus/VQS 所需路径，按实际安装位置替换，不要因它不是复制映射就删除 |

`py310` / `py37` 是卡片中可选的运行配置名称。环境版本标识默认使用这个名称，配置摘要由 worker 自动计算，均不需手填。
默认使用 `worker_id` 作为卡片显示名称，并将状态保存在配置文件目录下的同名子目录。

### 按需使用的配置

无需把下表字段全部补到示例中，只有对应场景才配置：

| 配置 | 使用场景或默认行为 |
| --- | --- |
| `display_name` / `root_path` | 自定义卡片名称或状态目录；默认分别为 worker ID 和配置文件目录下的 worker ID 子目录 |
| `base_branches` / `base_release_max` | 可选的基础分支提示；不限制手动选择，也不作为自动推荐依据 |
| `source_path` | 已有本地业务 Git 仓库时作为拉取加速来源；获取不到任务提交会回退远端，无需为了接入另外建业务仓库 |
| `vmid` | 需要关联外部机器身份时使用；普通 Devbox 接入无需填写 |
| `service_identity` / `start_argv` / `readiness_*` | 非标准业务启动或健康检查覆盖项；标准 Aeolus/VQS 使用仓库内置默认值，无需填写 |

默认服务身份为 Aeolus `data.aeolus.api_v3`、VQS `data.vqs.api`，端口由 worker 动态分配。
不要配置固定 `port`、`service_port` 或 `maat_url`；部署身份、会话和配置摘要均由系统生成。

## 更新已安装的 worker

更新仓库源码后，强制将当前包重新安装到专用虚拟环境：

```bash
uv pip install --python deployment-worker/.venv/bin/python \
  --reinstall-package conciliator-deployment-worker ./deployment-worker
deployment-worker/.venv/bin/python -c \
  'from conciliator_deployment_worker.profiles import runtime_profiles; from conciliator_deployment_worker.multi_worker import MachineWorker'
deployment-worker/.venv/bin/conciliator-deployment-worker --help
```

不同源码修订可能仍使用相同包版本号。缺少 `profiles` 或 `multi_worker` 通常表示
环境中仍是旧包；重新安装后确认模块和 CLI 可用，再停止旧监督进程并启动新进程。
使用同一份 v2 配置，保持 worker 身份、root、注册地址、journal 和端口账本不变。
重启后确认注册和心跳正常，并检查原业务实例的进程身份、监听归属及健康响应。
尽量等在途部署完成后重启；若在 checkout/启动过程中重启，该笔部署会被标记失败，需要释放后重新部署。
不要用后端 `.venv` 替代专用环境，也不要删除 journal 或端口账本来解决启动问题。

## 线上接入前检查

先使用 worker 自己的解释器检查线上入口。下面只有 GET，不注册、不领取任务：

```bash
deployment-worker/.venv/bin/python - <<'PY'
import httpx

import json
from pathlib import Path

config = json.loads(Path(".deployment-worker/worker.json").read_text())
with httpx.Client(base_url=config["api_base"],
                  timeout=15, trust_env=False, follow_redirects=False) as client:
    for path in ("/healthz", "/readyz"):
        response = client.get(path)
        response.raise_for_status()
        assert response.json().get("status") == "ok", path
        print(path, response.status_code, "ok")
PY
```

两个入口都应返回 HTTP 200 和 `status=ok`。worker 使用 `trust_env=False`，不会使用
`HTTP_PROXY` / `HTTPS_PROXY` 等环境代理；只用带代理的浏览器或 curl 测通不能证明 worker 可连接。
TLS 证书验证保持开启；不要用关闭校验或更换成非受信地址来绕过接入问题。

注册只上报运行配置摘要，不验证业务解释器和依赖是否真实可用。正式部署前可对选定配置做只读检查：

```bash
deployment-worker/.venv/bin/python - <<'PY'
import asyncio
import json
from pathlib import Path
from conciliator_deployment_worker.profiles import runtime_profiles
from conciliator_deployment_worker.multi_worker import verify_runtime

config = json.loads(Path(".deployment-worker/worker.json").read_text())
for repository, profiles in runtime_profiles(config).items():
    for name, profile in profiles.items():
        asyncio.run(verify_runtime(profile))
        print(repository, name, "ok")
PY
```

此检查核对实际解释器版本和依赖目录；旧配置若保留文件摘要则继续校验，不安装依赖或启动业务服务。
目录检查通过并不证明其中包含所有 Python 模块；还需核对 SCM parser 的实际内容及其与业务版本的兼容性。
它不代替 Git/doas 权限、MAAT 到 `advertised_host` 的连通性或真实业务部署验收。

## 启动与注册验收

**正式启动后会自动领取任务，不是只做注册探测。** 确认目标后端、配置和已有实例归属后再启动：

```bash
deployment-worker/.venv/bin/conciliator-deployment-worker \
  --config .deployment-worker/worker.json
```

首次建议前台运行，确认接入后可用现有进程管理工具托管。临时后台运行也可在仓库根目录执行：

```bash
umask 077
nohup deployment-worker/.venv/bin/conciliator-deployment-worker \
  --config .deployment-worker/worker.json \
  > .deployment-worker/worker.output 2>&1 < /dev/null &
echo $! > .deployment-worker/worker.pid
```

先停止前台进程再转后台，同一个配置不要重复启动。此方式不提供开机自启或异常自动重启。
停止时先确认 PID 仍对应自己的 worker，再发送 SIGTERM；不要使用按名称批量 kill 的命令。
个人配置、状态目录和输出文件留在本机，不提交仓库。

启动流程为：注册 → 恢复该后端返回的部署记录 → journal 对账心跳 → 周期心跳和任务领取。
注册会生成新的会话并将 `ready` 设为 false；对账成功后才设为 true。
不要为测试接口而向已有在线 worker 重复发送注册请求，否则会使旧会话失效。

接入验收须同时满足：

1. `POST /api/deployment-workers/register` 返回 HTTP 200、`protocol_version=2` 和非空会话；不公开完整响应或会话值。
2. 使用该会话的 `POST /api/deployment-workers/{worker_id}/heartbeat` 返回 HTTP 200，常驻 worker 完成对账后返回 `ready=true`。
3. 心跳持续成功，卡片显示 worker 在线且具名运行配置可选；超过 30 秒无新心跳会被判离线。

只验证注册/心跳时不要调用 `/claim`、上报部署结果或执行释放命令；保持 `reconciled=false`
可以验证会话连通性，但不会使新注册的 worker 具备领取任务资格，也不能宣称常驻接入已经完成。
确认上述常驻接入验收通过后，再从最新成功 Review 轮次发起业务部署。
推荐读取 Coding 精确 SHA 的 Python 声明和依赖；只有唯一兼容候选时自动推荐，
证据不足或多个候选时手动选择。明确不兼容的配置由服务端拒绝，基础分支仅作提示。
配置与 SHA 随部署固定，完整契约见[运行时选择](../docs/knowledge/meegle-ai-node/runtime-selection.md)。
`PYTHONPATH` 首项由 worker 自动设置为独立 checkout（即 `AEOLUS_ROOT`），
后续 `extra_paths` 仅按业务依赖需要设置，不要把共享的其他业务 checkout 加入路径。
配置摘要会上报，机器路径不上传；机器部署数量仅用于判断压力。

`advertised_host` 在注册地址、部署快照与服务结果中必须一致；
有未释放部署时禁止修改注册地址。监听归属无法通过 `/proc` 确认时，worker 不会宣告就绪。
本机健康、MAAT 入口和人工业务验证分别检查，详见 [Devbox 部署](../docs/knowledge/meegle-ai-node/devbox-deployment.md)。

### 从本地后端切换到线上

先在原后端确认在途任务及业务实例已完成释放或明确对账，再停止旧 worker。
为线上准备单独的配置文件和 `root_path`，保留原配置、journal 和端口账本；不要直接把原状态目录
指向另一后端。
同一 worker 身份在目标后端已有未释放部署时，需要恢复与目标记录匹配的 journal，不能删除记录或伪造空对账。

常见错误：DNS/连接失败先检查 Devbox 网络；401/403 检查线上入口访问策略；400/422 核对 v2 配置及字段；
409 核对同机身份、旧会话、注册地址与 journal；500 需从服务端检查数据库版本和表结构。
现有 CLI 没有 worker token/签名配置，`session_id` 不是独立身份认证；不能通过随意填一个 token 解决 401/403。

## 释放与升级

通过任务卡片释放指定实例。worker 核对进程身份、停止所属进程组、确认监听结束后回报；
独立 checkout 可清理，日志、journal 和端口账本保留用于恢复与对账。
不要删除 `ports.json` 或 journal，也不要通过杀掉端口上的任意进程宣称释放。

升级包后重启监督进程，现有实例按 journal 恢复；同一个状态目录不能同时运行两个管理器。
机器根目录保存端口账本，逐实例状态保存在 `instances/<deployment_id>/`，不能混用其他格式的状态目录。

本机停服确认后，后端自动软删除本次部署的 MAAT 环境和无引用代理，并回查有效记录已消失。
可继续 Coding；软删除不保证网关路由撤销，远端资源 ID 和旧端口预留仍保留。
worker 接口的地址、会话和实例一致性检查不构成身份认证，正式入口保护见[生产配置清单](../docs/production-configuration-checklist.md)。

## 常见问题

| 现象 | 检查和处理 |
| --- | --- |
| 卡片没有自己的 Devbox | 核对 `api_base` 是否与卡片环境一致；检查 worker 是否常驻且完成对账。无心跳超过 30 秒会离线 |
| 注册成功但不能选运行配置 | 确认上报了实际准备好的 service/profile，并核对卡片展示的代码证据与兼容性；证据不足时手动选择 |
| `ModuleNotFoundError: aeolus_expr_parse...` | 核对 `dependency_paths.aeolus_expr_parse` 是否指向兼容的完整 SCM parser 包目录；升级 worker 后重新部署，不直接覆盖旧实例 |
| `prepared parser requires a fresh checkout` | 旧目录已经初始化过 parser 子模块；从卡片释放该实例后重新部署，不删除 journal 或手动覆盖源码 |
| `prepared runtime` 检查失败 | 核对业务解释器、Python 版本和依赖目录；历史配置若保留文件校验，还需核对原摘要或 manifest |
| `missing instance journal` / 对账失败 | 核对 worker 身份、目标后端、root 和端口账本是否对应；保留现场，不用清空状态绕过 |
| 本机健康通过但 MAAT 入口失败 | 检查 MAAT 到 `advertised_host` 动态端口的连通性、业务身份及业务配置；本机健康不等于完整业务验收 |

## 旧配置兼容

已有配置中的 `runtime_revision`、`files`、`runtime_manifest` 继续生效，但新接入无需配置这些字段。
其中 `files` 仅用于可选文件摘要校验；`runtime_manifest` 是旧版独立文件路径，二者不能同时配置。
`extra_python_paths`、VQS 的 `dependency_root` 保留旧格式兼容，新配置分别使用 `extra_paths` 和 `dependency_paths`。

更新代码不要求迁移旧配置。确需调整配置时，先备份、完成在途部署并执行只读预检，
再重启 worker；配置变更会改变配置摘要。保留旧 manifest 供已有实例恢复，
不要修改实例的 assignment、journal 或端口账本，也不需要修改数据库。

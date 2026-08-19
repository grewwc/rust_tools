# 模型注册表（models/）

`a` 的模型配置拆分为本目录下的独立 JSON 文件，每个文件对应一个模型
（替代原根目录单文件 `models.json`）。加载器（`src/bin/ai/model_names.rs`）
按文件名排序读取所有 `*.json`，跳过 `README.md` 等非 JSON 文件。

## 约定

- 文件名 = `NN-<key>.json`：`NN` 为两位序号（`01` 起），**保留历史顺序**——
  顺序有语义（`default_candidate_rank` 用索引做平手裁决，影响默认模型选择），
  新增模型请追加到末尾（下一个序号），不要插入中间。
- 每个文件内容是一个模型对象（也支持 JSON 数组形式，便于用户侧合并文件）。
  字段含义见 `ModelDef` 的文档注释（`src/bin/ai/model_names.rs`）。
- 删除模型 = 删除对应文件；调整偏好 = 改对应字段。

## 用户覆盖

- 新格式（推荐）：`~/.config/rust_tools/models/` 目录，约定同上；目录存在时优先。
- 旧格式（兼容）：`~/.config/rust_tools/models.json` 单文件覆盖。
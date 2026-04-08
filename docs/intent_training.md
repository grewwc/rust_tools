# Intent Training

本项目的本地意图识别改为使用外部 `TF-IDF + Logistic Regression` 模型文件。

## 一句话理解

- `training_corpus.json`：人维护的训练语料
- `train_intent_model.py`：训练脚本
- `intent_model.json`：机器生成的模型参数文件，不建议手改

`intent_model.json` 里主要是 `bias`、`idf`、`weights` 这类线性模型参数，适合程序读取，不适合人工维护。日常要改效果时，优先改语料，然后重新训练生成模型文件。

## 文件

- 训练语料：`config/intent/training_corpus.json`
- 训练脚本：`scripts/train_intent_model.py`
- 训练产物：`config/intent/intent_model.json`

## 日常工作流

推荐按下面流程迭代：

1. 先在 `config/intent/training_corpus.json` 的 `samples` 数组中追加真实样本，每条样本至少包含 `text` 和 `core`
   例如：`{ "text": "帮我看一下这个报错", "core": "seek_solution" }`
2. 执行训练脚本生成新的 `config/intent/intent_model.json`
3. 运行相关测试，确认典型问句没有回归
4. 手工试几条边界问句，观察是否还存在误判

如果只是想修正分类效果，不要直接手改 `intent_model.json`。

## 训练

在仓库根目录执行：

```bash
python3 scripts/train_intent_model.py
```

可选参数：

```bash
python3 scripts/train_intent_model.py \
  --corpus config/intent/training_corpus.json \
  --output config/intent/intent_model.json \
  --epochs 220 \
  --learning-rate 0.55 \
  --l2 0.0002
```

常见做法：

- 先直接跑默认命令，看训练日志
- 如果语料增多后欠拟合，再调大 `--epochs`
- 如果权重抖动太大或容易过拟合，再适当调大 `--l2`

## 语料格式

`training_corpus.json` 包含四部分：

- `labels`：分类标签顺序
- `feature_config`：字符 n-gram 参数和最大特征数
- `runtime_rules`：搜索、否定、资源类型等修饰符规则
- `samples`：训练样本，字段为 `text` 和 `core`

### labels

当前标签为：

- `query_concept`
- `request_action`
- `seek_solution`
- `casual`

这几个标签的职责建议保持稳定。新增标签会影响训练脚本、运行时模型和下游逻辑，不建议随手改。

### feature_config

- `char_ngram_min` / `char_ngram_max`：字符 n-gram 范围
- 当前用 `2..4`，是为了兼顾中文短句和英文短语

如果没有明确证据，不建议频繁调整这个区间。对效果影响更大的通常是语料，而不是这里的参数。

### runtime_rules

这部分不是 LR 自动学出来的，而是运行时直接使用的辅助规则：

- `search_patterns`：判断是不是“搜索/推荐/列举类”请求
- `negation_patterns`：识别否定语气
- `resource_keywords`：提取 `skill/tool/doc/file` 等资源类型

这部分适合人工维护，因为它本来就是业务规则。

### samples

每条样本至少包含：

```json
{
  "text": "帮我审查这段代码",
  "core": "request_action"
}
```

建议：

- 每类样本都要有足够数量
- 尽量混合中文、英文、短句、口语化表达
- 多补“容易混淆”的边界样本，而不是只补典型样本

例如这些边界样本很重要：

- “有什么工具可以调试？”
- “帮我找几个 review skill”
- “为什么 tool call 会失败”
- “这是什么”

## 运行时

- 默认模型路径：`config/intent/intent_model.json`
- 可通过 `~/.configW` 配置覆盖：

```ini
ai.intent.model_path=/absolute/path/to/intent_model.json
```

- `ai.intent_model` 仍保留给 `thinking gate` / `skill router` 的轻量 LLM 使用，不参与本地意图分类

运行时行为：

- 如果 `ai.intent.model_path` 指向的模型文件存在，就优先加载它
- 如果你重新训练了模型，只要覆盖这个文件或改配置路径即可
- 不需要重新改 Rust 代码里的权重

## 验证

训练后至少做两类验证：

### 自动验证

可以跑当前已有的意图识别相关测试：

```bash
cargo test --bin a intent_recognition
```

如果只想快速看模型文件是否能被加载：

```bash
cargo test --bin a test_default_model_loads
```

### 手工验证

建议至少手测以下几类句子：

- 问候：`hello`、`你好`
- 概念：`什么是 trait object`
- 请求执行：`帮我重构这个函数`
- 解决问题：`怎么处理这个报错`
- 搜索类边界：`有哪些 skill 可以做 review`

## 调优建议

- 修改语料后重新执行训练脚本
- 如果要提升中文短句效果，优先补充真实用户问句样本
- 如果要减少误判，优先补充边界样本，例如“推荐几个 tool”“这是什么”“帮我看下报错”
- 如果某个类别总被误判，先补那个类别的反例和近邻样本，再考虑调训练参数
- 如果训练集精度很高但实际效果一般，通常说明语料分布不真实，需要补真实输入而不是继续堆参数

## 不建议做的事

- 不要手工修改 `intent_model.json` 里的 `weights`、`idf`、`bias`
- 不要只看训练集精度就判断模型可用
- 不要把所有效果问题都归因到模型参数，很多时候是语料不够或标签边界不清晰

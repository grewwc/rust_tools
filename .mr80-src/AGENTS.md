# Repository knowledge routing

This file is a discovery index, not a replacement for the user's coding instructions.

## Meegle / Meego AI node

For tasks involving any of the following terms, read `docs/knowledge/meegle-ai-node/README.md` before inspecting or changing code:

- Meegle, Meego, 飞书项目, AI 节点
- aeolus 编码助手, Coding 卡片, `API_BASE`
- `/hooks/meegle/ai-node`, `/api/aeolus/tasks/*`
- LPM, `meegle-plugin`, plugin point configuration, card release/publish

Use the routing table in that index and load only the relevant topic pages. Treat source code and current CLI output as authoritative if a knowledge page is stale.

Never read or print secret values from `.env`, LPM credentials, `plugin.config.json`, logs, or local caches. Remote point configuration and plugin publishing are externally visible operations and require explicit user confirmation before execution.

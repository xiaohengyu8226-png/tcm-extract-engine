# tcm-extract-engine

中医自动化抽取与自反馈自优化引擎。从 tcm-rag 抽离的独立项目，零依赖 tcm-rag。

## 定位

- 自动化抽取（C/A/B）：Profiler+Planner(C) → Extractors(A) → Validator+Verifier(B)。
- 自反馈自优化（GEPA 飞轮）：反思式变异 + Pareto 选择 + system-aware 合并 + 跨书飞轮。
- Artifact-Driven 8 阶段编排：parse → profile → plan → extract → validate → verify → review → publish。

不包含：Web 前端、Excel 导出、人工审核发布（这些留在 tcm-rag）。

## 安装

```bash
uv sync                # 基础依赖
uv sync --extra mineru # 需要真实 PDF 解析时（GPU）
uv sync --extra dev    # 开发/测试
```

## 使用

```bash
# 单本书跑到核验完成（止于 WAITING_REVIEW）
tcm-engine book run-book <pdf>

# 批量
tcm-engine book run <dir>

# 断点续跑 / 从某阶段重跑 / 重试
tcm-engine book resume <book_id>
tcm-engine book rerun <book_id> --from extract
tcm-engine book retry <book_id>

# GEPA 进化（单书 / 跨书 / 全模板）
tcm-engine gepa evolve <book_id> -t general_rule.md
tcm-engine gepa evolve-multi <book_id>... -t general_rule.md
tcm-engine gepa evolve-all <book_id>...

# 人工晋升胜出约束块到 prompts/a/<template>
tcm-engine gepa promote <run_dir> --reviewer <you>

# 查看产物与趋势
tcm-engine gepa runs
tcm-engine gepa trend
```

## 配置

首次使用请复制模板并填入真实值（`.env` 已在 `.gitignore` 中，切勿提交密钥）：

```bash
cp .env.example .env
```

从项目根 `.env` 读取（与 tcm-rag 同名环境变量）：`OPENAI_BASE_URL` / `OPENAI_API_KEY` /
`OPENAI_MODEL`，以及三模型独立覆盖 `PROMPT_*` / `EXTRACT_*` / `VERIFY_*`。

## 目录约定

- `workspace/books/<book_id>/`：单本书 Artifact 工作目录 + manifest.json。
- `workspace/gepa/`：GEPA 进化产物与趋势台账。
- `prompts/{a,b,c}/`：A/B/C 提示词模板（PromptTemplateRegistry 读取）。

# ETH Oracle · 设计与进度

更新：2026-10-20

## 1. 目标

构建一个可复现的 ETH 次日与未来 7 天方向预测研究系统：

- 多源数据：行情、技术指标、宏观、链上、资金费率、情绪/新闻。
- 每日自动结算、重训、评估、失败归因、生成日报。
- 本地 GUI：真实 K 线 + 日/周预测线 + 历史命中/失败标记 + 指标 + 日报。
- 统计诊断：Wilson CI、二项检验、Brier/Brier skill、ECE/MCE、可靠性图、sharpness。
- 跨标的验证：同一套特征/流程在 ETH/BTC/SOL 上做样本外走步回测。
- 结果诚实：展示基线、超额、显著性、校准和策略诊断，不伪造高胜率。
- 代码进入 GitHub 公开仓库，带测试和自动运行脚本。

## 2. 架构

```text
Binance / FRED / DefiLlama / Alternative.me / Wikipedia / RSS
        │
        ▼
app/data/*  ──► data/cache/dataset.csv + last_refresh.json
        │
        ▼
app/features.py  (154 features, next-day target)
        │
        ▼
app/model.py  (HGB + RF + ET + LR, calibrator, HGB regressor)
        │
        ├── app/evaluation.py  (metrics, walk-forward weights, threshold)
        ├── app/analysis.py    (failure labels, Markdown report)
        └── app/db.py          (SQLite: predictions/model_runs/news/reports)
        │
        ▼
app/main.py (FastAPI) ──► app/static/* (dashboard)
```

## 3. 关键设计决策

- **目标**：下一 UTC 日线收盘方向（close-to-close），以及 7 天后收盘方向；预测价 = 当前收盘 × (1 + 回归预测收益)。
- **防泄漏**：特征只用 `as_of` 及之前；日目标用 `close.shift(-1)`，周目标用 `close.shift(-7)`；周模型训练截止 `as_of - 7d`；走步回测每 30 天重训，验证集取训练集末尾 120 天；不随机打乱时间序列。
- **模型**：HGB 为主，RF/ET 提供非线性稳健性，LR 提供线性信号；验证集准确率决定初始权重，实盘每日用近 90 天表现动态调权。
- **校准**：Platt 校准器在验证集上拟合；概率只作为相对置信度，不宣称真实概率。
- **置信阈值**：在近 180 天、覆盖率 ≥25% 的约束下搜索 0.50–0.80，避免为了高准确率只选极少数样本。
- **失败归因**：规则化、多标签；同一笔错误可命中多个标签，因此占比合计可超过 100%。
- **存储**：SQLite WAL；预测以 `(mode, as_of_date, horizon_days)` 唯一；旧库自动迁移，日/周预测互不覆盖。
- **调度**：UTC 日线 08:00 CST 收盘，08:10 CST 跑每日任务（日/周模型与回测增量更新）。
- **统计诚实**：同时报告准确率、基线、二项检验 p 值、95% CI、Brier skill、ECE 和 sharpness；不把 0.5pp 的噪声包装成优势。

## 4. 已完成

- [x] 环境：Python 3.12.13 venv；pandas 3.0.6 / sklearn 1.9.1 / FastAPI 0.148。
- [x] 数据源联调：Binance、FRED、F&G、DefiLlama、Wikipedia、RSS；GDELT 默认关闭。
- [x] 数据缓存与降级：`data/cache/`、`last_refresh.json`、源状态。
- [x] 特征工程：154 个特征，覆盖技术/宏观/链上/情绪/资金费率/交互。
- [x] 模型：集成训练、Platt 校准、动态权重、特征重要性、top drivers。
- [x] 走步回测：365 天、366 条预测、每 30 天重训。
- [x] 每日流程：结算 → 拉数 → 重训 → 权重/阈值 → 预测 → 日报。
- [x] SQLite：predictions / model_runs / news_items / reports。
- [x] FastAPI：health/overview/candles/predictions/metrics/failures/news/report/model/run/job。
- [x] GUI：深色仪表盘、K 线、EMA、预测虚线、正确/错误 marker、滚动胜率、失败分布、新闻、日报。
- [x] 测试：特征、模型、指标、归因、日报、SQLite、API（合成数据，无网络）。
- [x] 运行脚本：server/daily/launchd 安装与卸载。
- [x] README、截图、GitHub Actions。
- [x] 首轮真实数据回测与日报。
- [x] 统计显著性（Wilson CI、精确二项检验）与概率校准（可靠性图、ECE、Brier skill、sharpness）。
- [x] 未来 7 天预测：周目标、周模型、周回测、周权重/阈值、数据库 horizon 隔离与迁移。
- [x] 多标的泛化验证：ETH/BTC/SOL 同一套特征走步回测，结果写入 `data/cache/asset_validation.json` 并由 `/api/validation` 展示。
- [x] GUI：日/周预测切换、显著性卡片、可靠性图、多标的验证表、预测周期列。
- [x] 31+ 测试：新增二项检验、校准、周目标、周模型、horizon 隔离/迁移、validation API。

## 5. 首轮真实结果

数据：`2017-08-17 → 2026-10-18`，3350 行，154 特征。日回测：`2025-10-18 → 2026-10-18`，366 条已结算。周回测：52 个不重叠 7 日窗口。

| 指标 | 日 | 周 |
|---|---:|---:|
| Accuracy | 51.37% | 34.62% |
| Baseline | 50.82% | 51.92% |
| Edge | +0.55% | -17.31% |
| 二项检验 p（vs 50%） | 0.638 | 0.036 |
| 95% Wilson CI | 46.26%–56.45% | 23.15%–48.20% |
| Brier | 0.2508 | 0.2778 |
| Brier skill vs 0.5 | -0.0033 | -0.1110 |
| ECE | 0.0136 | 0.2094 |
| AUC | 0.5088 | 0.3644 |
| 近 30 次 | 53.33% | — |
| 近 90 次 | 53.33% | — |
| 策略累计（无成本） | -24.95% | — |

结论：日模型只有不显著的微弱边际；周模型在首轮 52 个样本上显著低于随机，是失败分支。两者都适合研究/观察，不适合直接实盘。

多标的（365 天、每 30 天重训）：

| 标的 | Accuracy | Baseline | Edge | p（vs 50%） | Brier skill |
|---|---:|---:|---:|---:|---:|
| ETH | 51.37% | 50.82% | +0.55% | 0.638 | -0.0033 |
| BTC | 51.64% | 51.37% | +0.27% | 0.565 | -0.0023 |
| SOL | 50.55% | 52.19% | -1.64% | 0.875 | -0.0043 |

三个标的均无显著优势，支持“ETH 上的小幅超额是噪声/过拟合”的判断。

## 6. 踩坑与经验

- FRED 对 Chrome/自定义 UA 会挂起，`curl/8.7.1` UA + 直连优先可用；代理对 FRED 反而超时。
- QuickQ 代理 `127.0.0.1:10900`；Binance/宏观/链上大多可用；FRED 直连更好。
- GDELT 共享 IP 经常 429/超时，默认关闭；否则每日多等 20–30 秒且仍拿不到数据。
- CryptoSlate RSS 偶发 403；单源失败不影响整体。
- pandas 3 需要 `pct_change(fill_method=None)`；大量列插入会有 PerformanceWarning，已在 pytest 配置中忽略。
- sklearn 1.9 的 `LogisticRegression(penalty=...)` 已弃用，已移除该参数。
- macOS launchd 任务必须使用绝对 Python 路径并设置 `WorkingDirectory`；相对路径会找不到 `app`。
- 静态 K 线库使用 Lightweight Charts v4.2.0，已本地 vendor，不依赖 CDN。

## 7. 风险与限制

- 次日方向信噪比极低；回测区间/市场 regime 改变会显著影响准确率。
- 周模型仅 52 个样本，首轮准确率 34.6%，显著低于随机；样本扩大后结论可能变化，但当前不可用。
- 未计交易成本、滑点、资金费率；策略曲线不是可交易净值。
- 新闻情绪为英文词典打分，历史新闻 tone 缺失。
- 部分宏观序列发布有滞后（如高收益债利差），已用前向填充，但可能引入轻微陈旧信息。
- 特征数远大于样本增量，存在过拟合风险；已用正则/早停/验证集，但无法消除。
- 单机 SQLite + 单进程适合个人研究，不适合多用户并发写入。

## 8. 下一步

- [x] 概率校准（可靠性曲线 / ECE / Brier skill / sharpness）与显著性检验。
- [x] ETH/BTC/SOL 跨标泛化验证。
- [ ] 周模型失败归因：检查方向反转、regime 与样本重叠，决定修复或下线。
- [ ] 加入交易成本/滑点后的策略净值与按置信度分层的收益。
- [ ] 用滚动窗口做概率校准稳定性检查。
- [ ] 增加特征漂移和模型性能告警。
- [ ] 评估按波动率/趋势 regime 分层的子模型。
- [ ] 补充期权 skew、ETF flows、链上转账等数据。
- [ ] 将新闻情绪升级为金融领域模型或 LLM 结构化打分。
- [ ] 在 GitHub Actions 之外增加定时数据快照（可选）。

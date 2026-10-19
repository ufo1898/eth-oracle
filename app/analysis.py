from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

FEATURE_LABELS = {
    "ret_1d": "1日涨跌",
    "ret_5d": "5日涨跌",
    "ret_20d": "20日涨跌",
    "ret_30d": "30日涨跌",
    "ret_60d": "60日涨跌",
    "vol_7d": "7日波动率",
    "vol_30d": "30日波动率",
    "vol_90d": "90日波动率",
    "atr_14_pct": "14日ATR占比",
    "rsi_6": "RSI(6)",
    "rsi_14": "RSI(14)",
    "rsi_28": "RSI(28)",
    "macd_hist_norm": "MACD柱",
    "bb_pctb_20": "布林%B(20)",
    "bb_width_20": "布林带宽(20)",
    "stoch_k_14": "随机指标K",
    "adx_14": "ADX趋势强度",
    "mfi_14": "资金流量MFI",
    "volume_z_20": "成交量Z分数",
    "taker_buy_ratio": "主动买入占比",
    "close_sma_20": "相对20日均线",
    "close_sma_50": "相对50日均线",
    "close_sma_200": "相对200日均线",
    "sma_50_200": "50/200日均线差",
    "above_sma_200": "是否在200日均线上方",
    "dist_high_20": "距20日高点",
    "dist_low_20": "距20日低点",
    "dist_ath": "距历史高点",
    "funding_level": "资金费率(基点)",
    "funding_z_30": "资金费率Z分数",
    "fng_level": "恐惧贪婪指数",
    "fng_change_7d": "恐惧贪婪7日变化",
    "fng_percentile_252": "恐惧贪婪一年分位",
    "btc_ret_1d": "BTC 1日涨跌",
    "btc_ret_5d": "BTC 5日涨跌",
    "ethbtc_ret_1d": "ETH/BTC 1日涨跌",
    "ethbtc_sma_ratio_20": "ETH/BTC相对20日均线",
    "ethbtc_trend": "ETH/BTC趋势",
    "spx_ret_1d": "标普500 1日涨跌",
    "spx_ret_5d": "标普500 5日涨跌",
    "nasdaq_ret_1d": "纳斯达克1日涨跌",
    "dxy_ret_1d": "美元指数1日涨跌",
    "dxy_ret_5d": "美元指数5日涨跌",
    "us10y_change_1d": "美债10年收益率变化",
    "vix_level": "VIX水平",
    "vix_change_1d": "VIX变化",
    "hy_spread_change_1d": "高收益债利差变化",
    "tvl_log_change_7d": "ETH链上TVL 7日变化",
    "stablecoin_log_change_7d": "稳定币供应7日变化",
    "stablecoin_log_change_30d": "稳定币供应30日变化",
    "wiki_log_change_7d": "维基百科关注度7日变化",
    "wiki_z_30": "维基百科关注度Z分数",
    "news_tone_level": "新闻情绪",
    "news_tone_change_1d": "新闻情绪变化",
    "risk_on_composite": "风险偏好综合分",
    "rsi14_x_trend": "RSI×趋势",
    "funding_x_ret5": "资金费率×5日涨跌",
    "ethbtc_x_btc": "ETH/BTC×BTC",
    "vol_x_trend": "波动×趋势",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def classify_failure(
    prediction: dict[str, Any],
    features: dict[str, Any] | None,
    news_score: float | None = None,
) -> tuple[list[str], str]:
    features = features or {}
    categories: list[str] = []
    details: list[str] = []
    prob = _num(prediction.get("prob_up")) or 0.5
    confidence = max(prob, 1 - prob)
    actual_ret = _num(prediction.get("actual_return"))
    predicted_up = prob >= 0.5
    actual_up = (actual_ret or 0.0) > 0

    if confidence < 0.55:
        categories.append("低置信度")
        details.append(f"模型置信度仅 {confidence:.0%}")
    elif confidence >= 0.65:
        categories.append("过度自信")
        details.append(f"模型以 {confidence:.0%} 置信度判断{'上涨' if predicted_up else '下跌'}")

    if actual_ret is not None and abs(actual_ret) < 0.005:
        categories.append("小幅噪声")
        details.append(f"实际仅变动 {actual_ret:+.2%}，方向接近随机")

    atr = _num(features.get("atr_14_pct"))
    vol30 = _num(features.get("vol_30d"))
    if (atr is not None and atr > 0.055) or (vol30 is not None and vol30 > 0.06):
        categories.append("高波动")
        if atr is not None:
            details.append(f"ATR占比 {atr:.2%}")

    spx = _num(features.get("spx_ret_1d"))
    nasdaq = _num(features.get("nasdaq_ret_1d"))
    vix_change = _num(features.get("vix_change_1d"))
    hy_change = _num(features.get("hy_spread_change_1d"))
    if (
        (spx is not None and abs(spx) > 0.015)
        or (nasdaq is not None and abs(nasdaq) > 0.02)
        or (vix_change is not None and abs(vix_change) > 2.0)
        or (hy_change is not None and abs(hy_change) > 0.15)
    ):
        categories.append("宏观冲击")
        bits = []
        if spx is not None:
            bits.append(f"标普 {spx:+.2%}")
        if vix_change is not None:
            bits.append(f"VIX {vix_change:+.1f}")
        if bits:
            details.append("宏观：" + "、".join(bits))

    fng = _num(features.get("fng_level"))
    if fng is not None and (fng <= 20 or fng >= 80):
        categories.append("情绪极端")
        details.append(f"恐惧贪婪指数 {fng:.0f}")

    funding = _num(features.get("funding_level"))
    if funding is not None and abs(funding) > 4.0:
        categories.append("资金费率极端")
        details.append(f"资金费率 {funding:+.2f}bp")

    above50 = _num(features.get("above_sma_50"))
    if above50 is not None:
        if predicted_up and above50 < 0.5:
            categories.append("逆势看多")
            details.append("价格位于50日均线下方")
        elif (not predicted_up) and above50 >= 0.5:
            categories.append("逆势看空")
            details.append("价格位于50日均线上方")

    btc = _num(features.get("btc_ret_1d"))
    ethbtc = _num(features.get("ethbtc_ret_1d"))
    if btc is not None and abs(btc) > 0.025:
        categories.append("BTC联动")
        details.append(f"BTC {btc:+.2%}")
    if ethbtc is not None and abs(ethbtc) > 0.02:
        categories.append("ETH/BTC轮动")
        details.append(f"ETH/BTC {ethbtc:+.2%}")

    if news_score is not None and abs(news_score) > 0.25:
        categories.append("消息面冲击")
        details.append(f"新闻情绪 {news_score:+.2f}")

    if actual_ret is not None and abs(actual_ret) > 0.05:
        categories.append("大幅单边行情")
        details.append(f"实际波动 {actual_ret:+.2%}")

    # Preserve order but remove duplicates.
    categories = list(dict.fromkeys(categories))
    if not categories:
        categories = ["模型特征失效"]
    direction_text = "上涨" if predicted_up else "下跌"
    actual_text = "上涨" if actual_up else "下跌"
    reason = f"模型预测{direction_text}，实际{actual_text}"
    if actual_ret is not None:
        reason += f"（{actual_ret:+.2%}）"
    if details:
        reason += "。失效因素：" + "；".join(details[:5])
    reason += "。"
    return categories, reason


def _p_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def _sig_label(value: Any, observed: Any, reference: float | None = None) -> str:
    try:
        p_value = float(value)
        rate = float(observed)
    except (TypeError, ValueError):
        return "n/a"
    if p_value >= 0.05:
        return "不显著"
    target = 0.5 if reference is None else float(reference)
    return "显著高于基准" if rate > target else "显著低于基准"


def _prediction_lines(prediction: dict[str, Any], title: str) -> list[str]:
    direction = "上涨" if prediction.get("direction") == "UP" else "下跌"
    horizon = int(prediction.get("horizon_days") or 1)
    label = "周" if horizon > 1 else "日"
    return [
        f"- {title}：**{prediction.get('as_of_date')} 收盘 → {prediction.get('target_date')} 收盘**（{label}）",
        f"- 方向：**{direction}**｜上涨概率 {float(prediction.get('prob_up') or 0):.1%}"
        f"｜置信度 {float(prediction.get('confidence') or 0):.1%}",
        f"- 预测收盘价：**${float(prediction.get('predicted_close') or 0):,.2f}**"
        f"（预期 {float(prediction.get('predicted_return') or 0):+.2%}）",
    ]


def build_daily_report(
    *,
    report_date: str,
    latest_prediction: dict[str, Any] | None,
    metrics: dict[str, Any],
    failure_breakdown: list[dict[str, Any]],
    resolved_today: list[dict[str, Any]],
    model_info: dict[str, Any] | None,
    news_summary: dict[str, Any],
    news_items: list[dict[str, Any]],
    refresh_status: dict[str, Any],
    optimization: dict[str, Any],
    weekly_prediction: dict[str, Any] | None = None,
    weekly_metrics: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# ETH Oracle 日报 · {report_date}")
    lines.append("")
    lines.append(f"> 自动生成于 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}｜研究用途，不构成投资建议")
    lines.append("")

    lines.append("## 下一日预测")
    if latest_prediction:
        lines.extend(_prediction_lines(latest_prediction, "预测区间"))
        drivers = latest_prediction.get("drivers") or []
        if drivers:
            bits = [f"{feature_label(d['feature'])}（{'利多' if d['direction'] == 'up' else '利空'}）" for d in drivers[:5]]
            lines.append("- 主要驱动：" + "、".join(bits))
    else:
        lines.append("- 暂无预测，等待首次运行。")
    lines.append("")

    lines.append("## 未来 7 天预测")
    if weekly_prediction:
        lines.extend(_prediction_lines(weekly_prediction, "预测区间"))
        drivers = weekly_prediction.get("drivers") or []
        if drivers:
            bits = [f"{feature_label(d['feature'])}（{'利多' if d['direction'] == 'up' else '利空'}）" for d in drivers[:5]]
            lines.append("- 主要驱动：" + "、".join(bits))
    else:
        lines.append("- 暂无周预测。")
    lines.append("")

    lines.append("## 胜率统计")
    if metrics and metrics.get("n"):
        lines.append(
            f"- 全部已结算：**{metrics.get('accuracy'):.1%}**（{metrics.get('n')} 次）"
            f"｜基线 {metrics.get('baseline_accuracy'):.1%}｜超额 {metrics.get('edge'):+.1%}"
        )
        w30 = (metrics.get("windows") or {}).get("30")
        w90 = (metrics.get("windows") or {}).get("90")
        if w30:
            lines.append(f"- 近 30 次：{w30['accuracy']:.1%}（{w30['n']} 次）")
        if w90:
            lines.append(f"- 近 90 次：{w90['accuracy']:.1%}（{w90['n']} 次）")
        lines.append(
            f"- Brier {metrics.get('brier')}｜AUC {metrics.get('auc')}"
            f"｜策略累计 {metrics.get('strategy', {}).get('cumulative_return', 'n/a')}"
        )
    else:
        lines.append("- 暂无已结算预测。")
    if weekly_metrics and weekly_metrics.get("n"):
        lines.append(
            f"- 周预测：**{weekly_metrics.get('accuracy'):.1%}**（{weekly_metrics.get('n')} 次）"
            f"｜基线 {weekly_metrics.get('baseline_accuracy'):.1%}｜超额 {weekly_metrics.get('edge'):+.1%}"
            f"｜Brier {weekly_metrics.get('brier')}"
        )
    lines.append("")

    lines.append("## 统计显著性与概率校准")
    if metrics and metrics.get("n"):
        significance = metrics.get("significance") or {}
        calibration = metrics.get("calibration") or {}
        ci = significance.get("ci95") or [None, None]
        lines.append(
            f"- 准确率 {metrics.get('accuracy'):.1%}，95% CI "
            f"{ci[0]:.1%}–{ci[1]:.1%}，SE {significance.get('se_pp', 'n/a')}pp，n={metrics.get('n')}"
            if ci[0] is not None and ci[1] is not None
            else f"- 准确率 {metrics.get('accuracy'):.1%}，n={metrics.get('n')}"
        )
        vs_coin = significance.get("vs_coinflip") or {}
        vs_base = significance.get("vs_baseline") or {}
        if vs_coin:
            lines.append(
                f"- 对随机 50% 的二项检验：p={_p_value(vs_coin.get('p_value'))}"
                f"（{_sig_label(vs_coin.get('p_value'), metrics.get('accuracy'))}）"
            )
        if vs_base:
            lines.append(
                f"- 对基线 {metrics.get('baseline_accuracy'):.1%} 的二项检验：p={_p_value(vs_base.get('p_value'))}"
                f"（{_sig_label(vs_base.get('p_value'), metrics.get('accuracy'), metrics.get('baseline_accuracy'))}）"
            )
        if calibration:
            lines.append(
                f"- 校准：ECE {calibration.get('ece')}｜Brier skill vs 0.5 "
                f"{calibration.get('brier_skill_vs_half')}｜平均预测概率 {calibration.get('mean_prob')}"
                f"｜实际上涨率 {calibration.get('observed_up_rate')}"
            )
        if vs_coin and vs_coin.get("p_value") is not None and vs_coin["p_value"] >= 0.05:
            lines.append("- 解读：当前准确率与随机/基线无法在统计上区分，不应把小幅超额当作稳定优势。")
        elif vs_coin and vs_coin.get("p_value", 1) < 0.05 and metrics.get("accuracy", 0.5) < 0.5:
            lines.append("- 解读：当前准确率显著低于随机，模型方向可能系统性地错误，不应直接使用。")
    else:
        lines.append("- 暂无足够样本。")
    if weekly_metrics and weekly_metrics.get("n"):
        wsig = weekly_metrics.get("significance") or {}
        wcoin = wsig.get("vs_coinflip") or {}
        wcal = weekly_metrics.get("calibration") or {}
        lines.append(
            f"- 周预测：准确率 {weekly_metrics.get('accuracy'):.1%}，n={weekly_metrics.get('n')}"
            f"，对随机 50% 的二项检验 p={_p_value(wcoin.get('p_value'))}"
            f"（{_sig_label(wcoin.get('p_value'), weekly_metrics.get('accuracy'))}）"
        )
        if wcal:
            lines.append(
                f"- 周校准：ECE {wcal.get('ece')}｜Brier skill vs 0.5 {wcal.get('brier_skill_vs_half')}"
                f"｜平均预测概率 {wcal.get('mean_prob')}"
            )
    lines.append("")

    lines.append("## 昨日/最近结算复盘")
    if resolved_today:
        for item in resolved_today[:8]:
            mark = "✅" if item.get("correct") == 1 else "❌"
            horizon = int(item.get("horizon_days") or 1)
            label = "周" if horizon > 1 else "日"
            lines.append(
                f"- {mark} [{label}] 预测 {item.get('as_of_date')} → {item.get('target_date')}："
                f"方向 {'上涨' if float(item.get('prob_up') or 0) >= 0.5 else '下跌'}，"
                f"实际 {float(item.get('actual_return') or 0):+.2%}。{item.get('failure_reason') or ''}"
            )
    else:
        lines.append("- 本次没有新结算的预测。")
    lines.append("")

    lines.append("## 失败原因分布")
    lines.append("> 同一笔错误可命中多个标签，因此占比合计可超过 100%。")
    lines.append("")
    if failure_breakdown:
        for item in failure_breakdown[:8]:
            lines.append(f"- {item['category']}：{item['count']} 次（{item['share']:.0%}）")
    else:
        lines.append("- 暂无失败样本。")
    lines.append("")

    lines.append("## 消息面")
    lines.append(
        f"- 新闻情绪：{news_summary.get('score', 0):+.3f}"
        f"（样本 {news_summary.get('count', 0)} 条，正面 {news_summary.get('positive', 0)} / 负面 {news_summary.get('negative', 0)}）"
    )
    for item in news_items[:5]:
        lines.append(f"- [{item.get('source')}] {item.get('title')}（情绪 {float(item.get('sentiment') or 0):+.2f}）")
    if not news_items:
        lines.append("- 暂无新闻数据。")
    lines.append("")

    lines.append("## 模型与数据")
    if model_info:
        lines.append(
            f"- 模型版本 {model_info.get('version')}｜训练区间 {model_info.get('train_start')} → {model_info.get('train_end')}"
            f"｜样本 {model_info.get('n_samples')}｜特征 {model_info.get('n_features')}"
        )
        lines.append(f"- 集成权重：{model_info.get('weights')}")
        lines.append(f"- 验证集准确率：{model_info.get('val_metrics', {}).get('accuracy')}")
    if refresh_status:
        lines.append(
            f"- 数据截至 {refresh_status.get('last_date')}｜共 {refresh_status.get('rows')} 行"
            f"｜刷新于 {refresh_status.get('refreshed_at')}"
        )
        failed = [s for s in refresh_status.get("sources", []) if not s.get("ok")]
        if failed:
            lines.append("- 数据源告警：" + "、".join(f"{s.get('source')}（{s.get('error', 'failed')}）" for s in failed[:5]))
    lines.append("")

    lines.append("## 自动优化动作")
    lines.append(f"- 动态集成权重：{optimization.get('weights')}")
    lines.append(f"- 高置信度阈值：{optimization.get('threshold')}")
    weekly_optimization = optimization.get("weekly") or {}
    if weekly_optimization:
        lines.append(f"- 周模型权重：{weekly_optimization.get('weights')}｜阈值 {weekly_optimization.get('threshold')}")
    lines.append(
        f"- 流程：结算待定预测 → 重新拉取数据 → 重训模型 → 更新权重/阈值 → 生成下一日与未来 7 天预测。"
    )
    lines.append("")
    lines.append("---")
    lines.append("*ETH Oracle 是统计研究工具。加密资产波动极大，任何模型都无法稳定预测次日方向。*")
    return "\n".join(lines)

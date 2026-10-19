/* global LightweightCharts */
(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    priceChart: null,
    candleSeries: null,
    volumeSeries: null,
    ema20Series: null,
    ema50Series: null,
    predictionSeries: null,
    weeklyPredictionSeries: null,
    accuracyChart: null,
    accuracySeries: null,
    cumulativeSeries: null,
    baselineSeries: null,
    loading: false,
    updating: false,
    lastLoaded: null,
    latestPrediction: null,
    latestPredictions: { 1: null, 7: null },
    horizon: 1,
  };

  const FEATURE_LABELS = {
    ret_1d: "1日涨跌",
    ret_5d: "5日涨跌",
    ret_20d: "20日涨跌",
    ret_30d: "30日涨跌",
    ret_60d: "60日涨跌",
    vol_7d: "7日波动率",
    vol_30d: "30日波动率",
    vol_90d: "90日波动率",
    atr_14_pct: "14日ATR占比",
    rsi_6: "RSI(6)",
    rsi_14: "RSI(14)",
    rsi_28: "RSI(28)",
    macd_hist_norm: "MACD柱",
    bb_pctb_20: "布林%B",
    stoch_k_14: "随机指标K",
    adx_14: "ADX趋势",
    mfi_14: "MFI资金流",
    volume_z_20: "成交量Z分数",
    taker_buy_ratio: "主动买入占比",
    close_sma_20: "相对20日均线",
    close_sma_50: "相对50日均线",
    close_sma_200: "相对200日均线",
    sma_50_200: "50/200日均线差",
    above_sma_200: "200日均线之上",
    dist_high_20: "距20日高点",
    dist_low_20: "距20日低点",
    dist_ath: "距历史高点",
    funding_level: "资金费率(bp)",
    funding_z_30: "资金费率Z分数",
    fng_level: "恐惧贪婪指数",
    fng_change_7d: "恐惧贪婪7日变化",
    fng_percentile_252: "恐惧贪婪一年分位",
    btc_ret_1d: "BTC 1日涨跌",
    btc_ret_5d: "BTC 5日涨跌",
    ethbtc_ret_1d: "ETH/BTC 1日涨跌",
    ethbtc_sma_ratio_20: "ETH/BTC相对20日均线",
    ethbtc_trend: "ETH/BTC趋势",
    spx_ret_1d: "标普500 1日",
    spx_ret_5d: "标普500 5日",
    nasdaq_ret_1d: "纳斯达克1日",
    dxy_ret_1d: "美元指数1日",
    dxy_ret_5d: "美元指数5日",
    us10y_change_1d: "美债10年变化",
    vix_level: "VIX水平",
    vix_change_1d: "VIX变化",
    hy_spread_change_1d: "高收益债利差变化",
    tvl_log_change_7d: "TVL 7日变化",
    stablecoin_log_change_7d: "稳定币7日变化",
    stablecoin_log_change_30d: "稳定币30日变化",
    wiki_log_change_7d: "关注度7日变化",
    wiki_z_30: "关注度Z分数",
    news_tone_level: "新闻情绪",
    news_tone_change_1d: "新闻情绪变化",
    risk_on_composite: "风险偏好综合",
    rsi14_x_trend: "RSI×趋势",
    funding_x_ret5: "资金费率×5日",
    ethbtc_x_btc: "ETH/BTC×BTC",
    vol_x_trend: "波动×趋势",
  };

  const money = (value) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return `$${Number(value).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };
  const pct = (value, digits = 1) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return `${(Number(value) * 100).toFixed(digits)}%`;
  };
  const signedPct = (value, digits = 2) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    const number = Number(value) * 100;
    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}%`;
  };
  const num = (value, digits = 2) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(digits);
  };
  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  const label = (key) => FEATURE_LABELS[key] || String(key || "").replaceAll("_", " ");
  const shortDate = (value) => (value ? String(value).slice(0, 10) : "—");
  const relativeTime = (iso) => {
    if (!iso) return "";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return "";
    const diff = Math.max(0, Date.now() - then);
    const minutes = Math.floor(diff / 60000);
    if (minutes < 60) return `${Math.max(1, minutes)} 分钟前`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} 小时前`;
    return `${Math.floor(hours / 24)} 天前`;
  };

  const toast = (message, type = "") => {
    const el = $("#toast");
    el.textContent = message;
    el.className = `toast show ${type}`;
    window.clearTimeout(toast.timer);
    toast.timer = window.setTimeout(() => {
      el.className = "toast";
    }, 4200);
  };

  async function getJSON(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (_) {
        // ignore
      }
      throw new Error(`${response.status} ${detail}`);
    }
    return response.json();
  }

  function initPriceChart() {
    const container = $("#price-chart");
    if (!container || state.priceChart) return;
    const chart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { type: "solid", color: "transparent" },
        textColor: "#8b9bb4",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(148, 163, 184, 0.07)" },
        horzLines: { color: "rgba(148, 163, 184, 0.07)" },
      },
      rightPriceScale: {
        borderColor: "rgba(148, 163, 184, 0.16)",
        scaleMargins: { top: 0.08, bottom: 0.24 },
      },
      timeScale: {
        borderColor: "rgba(148, 163, 184, 0.16)",
        rightOffset: 6,
        barSpacing: 8,
        fixLeftEdge: false,
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { color: "rgba(168, 85, 247, 0.5)", width: 1, style: 2 },
        horzLine: { color: "rgba(168, 85, 247, 0.4)", width: 1, style: 2 },
      },
      localization: {
        priceFormatter: (price) => `$${Number(price).toLocaleString("en-US", { maximumFractionDigits: 0 })}`,
      },
    });

    state.priceChart = chart;
    state.candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      priceLineVisible: false,
    });
    state.volumeSeries = chart.addHistogramSeries({
      priceScaleId: "volume",
      priceFormat: { type: "volume" },
      lastValueVisible: false,
      priceLineVisible: false,
    });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    state.ema20Series = chart.addLineSeries({
      color: "rgba(34, 211, 238, 0.58)",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      title: "EMA20",
    });
    state.ema50Series = chart.addLineSeries({
      color: "rgba(245, 158, 11, 0.48)",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      title: "EMA50",
    });
    state.predictionSeries = chart.addLineSeries({
      color: "#a855f7",
      lineWidth: 3,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      title: "次日预测",
    });
    state.weeklyPredictionSeries = chart.addLineSeries({
      color: "#22d3ee",
      lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
      title: "周预测",
    });

    const resize = () => chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    window.addEventListener("resize", resize);
    if (window.ResizeObserver) new ResizeObserver(resize).observe(container);
  }

  function initAccuracyChart() {
    const container = $("#accuracy-chart");
    if (!container || state.accuracyChart) return;
    const chart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { type: "solid", color: "transparent" },
        textColor: "#8b9bb4",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: "rgba(148, 163, 184, 0.05)" },
        horzLines: { color: "rgba(148, 163, 184, 0.06)" },
      },
      rightPriceScale: {
        borderColor: "rgba(148, 163, 184, 0.14)",
        scaleMargins: { top: 0.12, bottom: 0.12 },
      },
      timeScale: { borderColor: "rgba(148, 163, 184, 0.14)", rightOffset: 4, barSpacing: 5 },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      localization: {
        priceFormatter: (price) => `${(Number(price) * 100).toFixed(0)}%`,
      },
    });
    state.accuracyChart = chart;
    state.baselineSeries = chart.addLineSeries({
      color: "rgba(148, 163, 184, 0.32)",
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    state.cumulativeSeries = chart.addLineSeries({
      color: "rgba(34, 211, 238, 0.72)",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    state.accuracySeries = chart.addLineSeries({
      color: "#a855f7",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: true,
    });
    const resize = () => chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    window.addEventListener("resize", resize);
    if (window.ResizeObserver) new ResizeObserver(resize).observe(container);
  }

  function renderOverview(data) {
    const latest = data.latest_prediction;
    const dataset = data.data || {};
    const refresh = dataset.refresh || {};
    const failedSources = (refresh.sources || []).filter((source) => !source.ok);
    const badge = $("#data-badge");
    if (dataset.last_date) {
      badge.textContent = `数据 ${dataset.last_date} · ${dataset.rows} 行${failedSources.length ? ` · ${failedSources.length} 源降级` : ""}`;
      badge.className = failedSources.length ? "badge warn" : "badge";
      badge.title = failedSources.map((source) => `${source.source}: ${source.error || "cached"}`).join("\n");
    } else {
      badge.textContent = "尚未初始化";
      badge.className = "badge warn";
    }

    const model = data.model || {};
    const modelBadge = $("#model-badge");
    if (model.version) {
      const valAcc = model.val_metrics && model.val_metrics.accuracy;
      modelBadge.textContent = `模型 ${model.version}${valAcc != null ? ` · 验证 ${(valAcc * 100).toFixed(0)}%` : ""}`;
      modelBadge.title = `训练 ${model.train_start || "?"} → ${model.train_end || "?"} · ${model.n_samples || "?"} 样本 · ${model.n_features || "?"} 特征`;
      modelBadge.className = "badge muted";
    } else {
      modelBadge.textContent = "模型未训练";
    }

    state.latestPrediction = latest || null;
    state.latestPredictions = {
      1: data.latest_prediction || null,
      7: data.latest_weekly_prediction || null,
    };
    renderPredictionCard();
  }

  function renderPredictionCard() {
    const latest = state.horizon === 7
      ? state.latestPredictions[7]
      : (state.latestPredictions[1] || state.latestPrediction);
    const weekly = state.horizon === 7;
    const directionEl = $("#pred-direction");
    const statusEl = $("#pred-status");
    $("#pred-eyebrow").textContent = weekly ? "未来 7 天预测" : "下一日预测";
    document.querySelectorAll(".horizon-tab").forEach((tab) => {
      tab.classList.toggle("active", Number(tab.dataset.horizon) === state.horizon);
    });
    if (!latest) {
      directionEl.textContent = "—";
      directionEl.className = "direction-big";
      $("#pred-target").textContent = weekly ? "周预测尚未生成" : "等待首次运行";
      $("#pred-range").textContent = weekly
        ? "每日任务运行后生成 7 天方向预测"
        : "点击右上角“立即更新”生成预测";
      statusEl.textContent = weekly ? "无周预测" : "未初始化";
      statusEl.className = "pill pending";
      ["#pred-prob", "#pred-confidence", "#pred-close", "#pred-predicted-close"].forEach((selector) => {
        $(selector).textContent = "—";
      });
      $("#pred-prob-bar").style.width = "0%";
      $("#pred-return").textContent = "—";
      $("#pred-return").className = "pred-return";
      $("#pred-drivers").innerHTML = '<span class="muted">暂无因子</span>';
      return;
    }

    const isUp = String(latest.direction).toUpperCase() === "UP";
    directionEl.textContent = isUp ? "上涨" : "下跌";
    directionEl.className = `direction-big ${isUp ? "up" : "down"}`;
    $("#pred-target").textContent = `预测 ${shortDate(latest.target_date)} 收盘`;
    $("#pred-range").textContent = `${shortDate(latest.as_of_date)} 收盘 → ${shortDate(latest.target_date)} 收盘 · ${weekly ? "7 天" : "次日"} · UTC`;
    statusEl.textContent = latest.status === "resolved"
      ? `已结算 · ${latest.correct ? "正确" : "错误"}`
      : "待结算";
    statusEl.className = `pill ${latest.status === "resolved" ? (latest.correct ? "win" : "loss") : "pending"}`;

    $("#pred-prob").textContent = pct(latest.prob_up, 1);
    $("#pred-confidence").textContent = pct(latest.confidence, 1);
    $("#pred-prob-bar").style.width = `${Math.max(0, Math.min(100, Number(latest.prob_up) * 100))}%`;
    $("#pred-close").textContent = money(latest.close);
    $("#pred-predicted-close").textContent = money(latest.predicted_close);
    const returnEl = $("#pred-return");
    returnEl.textContent = `预期变动 ${signedPct(latest.predicted_return)} · 模型认为${isUp ? "上行" : "下行"}概率更高`;
    returnEl.className = `pred-return ${isUp ? "up" : "down"}`;

    const drivers = latest.drivers || [];
    if (!drivers.length) {
      $("#pred-drivers").innerHTML = '<span class="muted">暂无因子快照</span>';
    } else {
      $("#pred-drivers").innerHTML = drivers.slice(0, 6).map((driver) => {
        const bullish = Number(driver.contribution) >= 0;
        return `<div class="driver"><span>${escapeHtml(label(driver.feature))}</span><b class="${bullish ? "bull" : "bear"}">${bullish ? "利多" : "利空"}</b></div>`;
      }).join("");
    }
  }

  function renderStats(metrics) {
    const overall = metrics.overall || {};
    const windows = overall.windows || {};
    const setText = (selector, value) => { const el = $(selector); if (el) el.textContent = value; };
    const accuracyClass = (value) => value == null ? "" : value >= 0.52 ? "good" : value < 0.48 ? "bad" : "";

    setText("#stat-all-acc", pct(overall.accuracy));
    $("#stat-all-acc").className = accuracyClass(overall.accuracy);
    setText("#stat-all-n", overall.n ? `${overall.n} 次已结算` : "暂无已结算");
    setText("#stat-30", windows["30"] ? pct(windows["30"].accuracy) : "—");
    setText("#stat-30-n", windows["30"] ? `${windows["30"].n} 次 · ${windows["30"].from} 起` : "样本不足");
    setText("#stat-90", windows["90"] ? pct(windows["90"].accuracy) : "—");
    setText("#stat-90-n", windows["90"] ? `${windows["90"].n} 次 · ${windows["90"].from} 起` : "样本不足");
    setText("#stat-edge", overall.edge == null ? "—" : signedPct(overall.edge, 1));
    $("#stat-edge").className = overall.edge == null ? "" : overall.edge >= 0 ? "good" : "bad";
    setText("#stat-baseline", overall.baseline_accuracy == null ? "基线 —" : `基线 ${pct(overall.baseline_accuracy)}`);
    setText("#stat-brier", overall.brier == null ? "—" : num(overall.brier, 3));
    setText("#stat-strategy", overall.strategy && overall.strategy.cumulative_return != null ? signedPct(overall.strategy.cumulative_return, 1) : "—");
    $("#stat-strategy").className = overall.strategy && overall.strategy.cumulative_return >= 0 ? "good" : overall.strategy ? "bad" : "";
    setText("#stat-sharpe", overall.strategy && overall.strategy.sharpe != null ? `Sharpe ${num(overall.strategy.sharpe, 2)}` : "多空日频策略");

    const weekly = (metrics.weekly && metrics.weekly.overall) || {};
    setText("#stat-weekly-acc", weekly.accuracy == null ? "—" : pct(weekly.accuracy));
    $("#stat-weekly-acc").className = accuracyClass(weekly.accuracy);
    setText("#stat-weekly-n", weekly.n
      ? `${weekly.n} 次 · 基线 ${pct(weekly.baseline_accuracy)}`
      : "样本不足");
  }

  function pText(value) {
    if (value == null) return "—";
    if (Number(value) < 0.001) return "<0.001";
    return Number(value).toFixed(3);
  }

  function renderSignificance(metrics) {
    const overall = metrics.overall || {};
    const sig = overall.significance || {};
    const container = $("#significance");
    const badge = $("#sig-badge");
    if (!container) return;
    if (!sig.n) {
      container.innerHTML = '<div class="empty">暂无已结算样本</div>';
      if (badge) { badge.textContent = "样本不足"; badge.className = "badge muted"; }
      return;
    }
    const ci = sig.ci95 || [];
    const vsCoin = sig.vs_coinflip || {};
    const vsBase = sig.vs_baseline || {};
    const edge = Number(overall.edge || 0);
    const pValue = vsCoin.p_value == null ? null : Number(vsCoin.p_value);
    const significant = edge > 0 && pValue != null && pValue < 0.05;
    const negative = edge < 0 && pValue != null && pValue < 0.05;
    const pClass = (value, positive) => {
      if (value == null) return "";
      if (Number(value) < 0.05) return positive ? "good" : "bad";
      return "";
    };
    const rows = [
      ["准确率（95% Wilson CI）", ci.length === 2 ? `${pct(overall.accuracy)}　[${pct(ci[0])}, ${pct(ci[1])}]` : pct(overall.accuracy)],
      ["标准误 SE", sig.se_pp == null ? "—" : `${num(sig.se_pp, 2)} pp`],
      ["对随机 50%", `p = ${pText(vsCoin.p_value)}`],
      [`对基线 ${pct(overall.baseline_accuracy)}`, `p = ${pText(vsBase.p_value)}`],
      ["样本量", `${sig.n} 次已结算`],
    ];
    const weeklyOverall = (metrics.weekly && metrics.weekly.overall) || {};
    const weeklySig = weeklyOverall.significance || {};
    const weeklyCoin = weeklySig.vs_coinflip || {};
    if (weeklySig.n) {
      rows.push([
        `周预测（n=${weeklySig.n}）`,
        `${pct(weeklyOverall.accuracy)} · p = ${pText(weeklyCoin.p_value)}`,
      ]);
    }
    container.innerHTML = rows.map(([name, value], index) => {
      let cls = "";
      if (index === 2 || index === 3) cls = pClass(vsCoin.p_value, edge > 0);
      if (index === 5 && weeklyCoin.p_value != null) {
        const weeklyEdge = Number(weeklyOverall.edge || 0);
        cls = pClass(weeklyCoin.p_value, weeklyEdge > 0);
      }
      return `<div class="sig-row"><span>${name}</span><strong class="${cls}">${value}</strong></div>`;
    }).join("") + `<p class="sig-note ${significant ? "" : "warn"}">${significant
      ? `准确率显著高于随机（p &lt; 0.05），但仍需持续样本外验证。`
      : negative
        ? `准确率显著低于随机（p = ${pText(vsCoin.p_value)}），模型方向可能系统性错误，请勿直接使用。`
        : `p ≥ 0.05：当前 ${signedPct(edge, 1)} 的超额与零无法区分，不能视为稳定优势。`}</p>`;
    if (badge) {
      badge.textContent = significant ? "显著高于随机" : negative ? "显著低于随机" : "不显著";
      badge.className = significant ? "badge" : "badge warn";
    }
  }

  function renderReliability(metrics) {
    const overall = metrics.overall || {};
    const cal = overall.calibration || {};
    const container = $("#reliability-chart");
    if (!container) return;
    const bins = (cal.bins || []).filter((bin) => bin.n > 0 && bin.mean_prob != null && bin.observed_up != null);
    if (!bins.length) {
      container.innerHTML = '<div class="empty">暂无校准数据</div>';
    } else {
      const W = 460;
      const H = 250;
      const pad = { left: 40, right: 14, top: 12, bottom: 34 };
      const x = (value) => pad.left + value * (W - pad.left - pad.right);
      const y = (value) => H - pad.bottom - value * (H - pad.top - pad.bottom);
      const grid = [0, 0.25, 0.5, 0.75, 1].map((tick) => {
        const lines = [];
        lines.push(`<line x1="${x(tick)}" y1="${y(0)}" x2="${x(tick)}" y2="${y(1)}" stroke="rgba(148,163,184,0.10)" />`);
        lines.push(`<line x1="${x(0)}" y1="${y(tick)}" x2="${x(1)}" y2="${y(tick)}" stroke="rgba(148,163,184,0.10)" />`);
        lines.push(`<text x="${x(tick)}" y="${H - pad.bottom + 14}" fill="#64748b" font-size="9" text-anchor="middle">${Math.round(tick * 100)}%</text>`);
        if (tick > 0) lines.push(`<text x="${pad.left - 8}" y="${y(tick) + 3}" fill="#64748b" font-size="9" text-anchor="end">${Math.round(tick * 100)}%</text>`);
        return lines.join("");
      }).join("");
      const points = bins.map((bin) => {
        const cx = x(Number(bin.mean_prob));
        const cy = y(Number(bin.observed_up));
        const hi = bin.ci95 ? y(Number(bin.ci95[1])) : cy;
        const lo = bin.ci95 ? y(Number(bin.ci95[0])) : cy;
        const radius = Math.min(9, 3.5 + Math.sqrt(Number(bin.n)) * 1.2);
        return `<line x1="${cx}" y1="${lo}" x2="${cx}" y2="${hi}" stroke="rgba(34,211,238,0.45)" stroke-width="1.5" />`
          + `<circle cx="${cx}" cy="${cy}" r="${radius}" fill="rgba(34,211,238,0.22)" stroke="#22d3ee" stroke-width="2">`
          + `<title>预测 ${pct(bin.mean_prob, 1)} · 实际 ${pct(bin.observed_up, 1)} · n=${bin.n}</title></circle>`;
      }).join("");
      container.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="可靠性图">
        ${grid}
        <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="rgba(168,85,247,0.6)" stroke-dasharray="5 4" />
        ${points}
        <text x="${(x(0) + x(1)) / 2}" y="${H - 4}" fill="#8b9bb4" font-size="10" text-anchor="middle">预测上涨概率</text>
        <text x="10" y="${(y(0) + y(1)) / 2}" fill="#8b9bb4" font-size="10" text-anchor="middle" transform="rotate(-90 10 ${(y(0) + y(1)) / 2})">实际上涨频率</text>
      </svg>`;
    }
    const stats = $("#cal-stats");
    if (stats) {
      stats.innerHTML = [
        ["ECE", cal.ece == null ? "—" : num(cal.ece, 3)],
        ["Brier skill", cal.brier_skill_vs_half == null ? "—" : num(cal.brier_skill_vs_half, 3)],
        ["Sharpness", cal.sharpness == null ? "—" : num(cal.sharpness, 3)],
        ["平均概率", cal.mean_prob == null ? "—" : pct(cal.mean_prob, 1)],
      ].map(([name, value]) => `<div><span>${name}</span><b>${value}</b></div>`).join("");
    }
    const badge = $("#cal-badge");
    if (badge) {
      if (cal.ece == null) {
        badge.textContent = "无数据";
        badge.className = "badge muted";
      } else {
        const weak = cal.sharpness != null && Number(cal.sharpness) < 0.03;
        badge.textContent = weak ? "区分度很弱" : `ECE ${num(cal.ece, 3)}`;
        badge.className = weak ? "badge warn" : "badge muted";
      }
    }
  }

  function renderValidation(data) {
    const tbody = $("#validation-tbody");
    const meta = $("#validation-meta");
    if (!tbody) return;
    const results = (data && data.results) || [];
    if (!results.length) {
      tbody.innerHTML = '<tr><td colspan="9" class="empty">运行 python -m app.validate_assets 生成验证结果</td></tr>';
      if (meta) { meta.textContent = "未运行"; meta.className = "badge muted"; }
      return;
    }
    tbody.innerHTML = results.map((item) => {
      const sig = item.significance || {};
      const vsCoin = sig.vs_coinflip || {};
      const cal = item.calibration || {};
      const edge = Number(item.edge || 0);
      const pValue = vsCoin.p_value == null ? "—" : Number(vsCoin.p_value) < 0.001 ? "<0.001" : Number(vsCoin.p_value).toFixed(3);
      return `<tr>
        <td>${escapeHtml(item.label || item.symbol)}</td>
        <td class="num">${item.n ?? "—"}</td>
        <td class="num">${pct(item.accuracy)}</td>
        <td class="num">${pct(item.baseline_accuracy)}</td>
        <td class="num"><span class="${edge >= 0 ? "tag up" : "tag down"}">${signedPct(edge, 1)}</span></td>
        <td class="num">${pValue}</td>
        <td class="num">${num(item.brier, 3)}</td>
        <td class="num">${num(cal.brier_skill_vs_half, 3)}</td>
        <td class="num">${num(cal.ece, 3)}</td>
      </tr>`;
    }).join("");
    if (meta) {
      meta.textContent = `${data.days || "—"} 天走步 · ${shortDate(data.generated_at)}`;
      meta.className = "badge muted";
    }
  }

  function renderPriceChart(data) {
    initPriceChart();
    if (!state.priceChart) return;
    const candles = data.candles || [];
    if (!candles.length) {
      if (state.candleSeries) state.candleSeries.setData([]);
      return;
    }
    state.candleSeries.setData(candles.map((item) => ({
      time: item.time,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    })));
    state.volumeSeries.setData(candles.map((item) => ({
      time: item.time,
      value: item.volume,
      color: item.close >= item.open ? "rgba(34, 197, 94, 0.22)" : "rgba(239, 68, 68, 0.22)",
    })));
    if (data.ema20) state.ema20Series.setData(data.ema20);
    if (data.ema50) state.ema50Series.setData(data.ema50);

    const prediction = data.prediction;
    const weekly = data.weekly_prediction;
    if (prediction && prediction.as_of_date && prediction.predicted_close != null) {
      state.predictionSeries.setData([
        { time: prediction.as_of_date, value: prediction.close },
        { time: prediction.target_date, value: prediction.predicted_close },
      ]);
    } else {
      state.predictionSeries.setData([]);
    }
    if (weekly && weekly.as_of_date && weekly.predicted_close != null) {
      state.weeklyPredictionSeries.setData([
        { time: weekly.as_of_date, value: weekly.close },
        { time: weekly.target_date, value: weekly.predicted_close },
      ]);
    } else {
      state.weeklyPredictionSeries.setData([]);
    }
    if (prediction && prediction.as_of_date && prediction.predicted_close != null) {
      $("#chart-sub").textContent = `次日 ${shortDate(prediction.target_date)} · 周 ${weekly ? shortDate(weekly.target_date) : "—"} · 紫/青虚线`;
    } else {
      $("#chart-sub").textContent = "最近 240 个已收盘交易日";
    }

    const candleTimes = new Set(candles.map((item) => item.time));
    const markers = (data.markers || []).filter((marker) => candleTimes.has(marker.time));
    if (prediction && prediction.as_of_date && candleTimes.has(prediction.as_of_date)) {
      markers.push({
        time: prediction.as_of_date,
        position: "aboveBar",
        color: "#a855f7",
        shape: "arrowDown",
        text: "预测",
      });
    }
    markers.sort((a, b) => String(a.time).localeCompare(String(b.time)));
    state.candleSeries.setMarkers(markers);
    state.priceChart.timeScale().fitContent();
  }

  function renderAccuracyChart(metrics) {
    initAccuracyChart();
    if (!state.accuracyChart) return;
    const series = metrics.accuracy_series || [];
    if (!series.length) {
      state.accuracySeries.setData([]);
      state.cumulativeSeries.setData([]);
      state.baselineSeries.setData([]);
      return;
    }
    state.accuracySeries.setData(series.map((item) => ({ time: item.date, value: item.accuracy })));
    state.cumulativeSeries.setData(series.map((item) => ({ time: item.date, value: item.cumulative })));
    const first = series[0].date;
    const last = series[series.length - 1].date;
    state.baselineSeries.setData([{ time: first, value: 0.5 }, { time: last, value: 0.5 }]);
    state.accuracyChart.timeScale().fitContent();
  }

  function renderFailures(data) {
    const breakdown = data.breakdown || [];
    const total = breakdown.reduce((sum, item) => sum + Number(item.count || 0), 0);
    $("#failure-count").textContent = total ? `${total} 个标签` : "暂无失败";
    const container = $("#failure-list");
    if (!breakdown.length) {
      container.innerHTML = '<div class="empty">暂无失败样本</div>';
      return;
    }
    const max = Math.max(...breakdown.map((item) => Number(item.count || 0)), 1);
    container.innerHTML = breakdown.slice(0, 10).map((item) => `
      <div class="failure-row">
        <span>${escapeHtml(item.category)}</span>
        <div class="failure-track"><div class="failure-fill" style="width:${Math.max(4, (Number(item.count) / max) * 100)}%"></div></div>
        <b>${Number(item.count)}</b>
      </div>
    `).join("");
  }

  function renderPredictions(data) {
    const tbody = $("#pred-tbody");
    const items = data.items || [];
    if (!items.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty">暂无预测记录</td></tr>';
      return;
    }
    tbody.innerHTML = items.slice(0, 40).map((item) => {
      const isUp = String(item.direction).toUpperCase() === "UP";
      const resolved = item.status === "resolved";
      const weekly = Number(item.horizon_days) === 7;
      const resultClass = resolved ? (item.correct ? "win" : "loss") : "pending";
      const resultText = resolved ? (item.correct ? "正确" : "错误") : "待结算";
      const reason = item.failure_reason || (resolved ? "—" : "等待目标日收盘后结算");
      return `
        <tr>
          <td>${shortDate(item.target_date)}<div class="muted" style="font-size:10px">${item.mode === "live" ? "实盘" : "回测"}</div></td>
          <td><span class="tag ${weekly ? "weekly" : ""}">${weekly ? "周" : "次日"}</span></td>
          <td><span class="tag ${isUp ? "up" : "down"}">${isUp ? "涨" : "跌"}</span></td>
          <td class="num">${pct(item.prob_up, 1)}</td>
          <td class="num">${money(item.predicted_close)}</td>
          <td class="num">${item.actual_close != null ? money(item.actual_close) : "—"}</td>
          <td><span class="tag ${resultClass}">${resultText}</span></td>
          <td class="reason" title="${escapeHtml(reason)}">${escapeHtml(reason.length > 90 ? `${reason.slice(0, 90)}…` : reason)}</td>
        </tr>
      `;
    }).join("");
  }

  function renderNews(data) {
    const summary = data.summary || {};
    $("#news-summary").textContent = `情绪 ${Number(summary.score || 0) >= 0 ? "+" : ""}${num(summary.score, 3)} · ${summary.count || 0} 条 · 正 ${summary.positive || 0} / 负 ${summary.negative || 0}`;
    const items = data.items || [];
    const container = $("#news-list");
    if (!items.length) {
      container.innerHTML = '<div class="empty">暂无新闻</div>';
      return;
    }
    container.innerHTML = items.slice(0, 40).map((item) => {
      const score = Number(item.sentiment || 0);
      const sentimentClass = score >= 0.05 ? "pos" : score <= -0.05 ? "neg" : "neu";
      const sentimentText = score >= 0.05 ? "偏多" : score <= -0.05 ? "偏空" : "中性";
      const href = /^https?:\/\//i.test(item.url || "") ? escapeHtml(item.url) : "#";
      return `
        <a class="news-item" href="${href}" target="_blank" rel="noopener noreferrer">
          <div class="news-title">${escapeHtml(item.title)}</div>
          <div class="news-meta">
            <span>${escapeHtml(item.source || "news")}</span>
            <span>·</span>
            <span>${escapeHtml(relativeTime(item.published_at))}</span>
            <span class="sentiment ${sentimentClass}">${sentimentText} ${score >= 0 ? "+" : ""}${score.toFixed(2)}</span>
          </div>
        </a>
      `;
    }).join("");
  }

  function renderReport(data) {
    const content = data.report && data.report.content;
    if (!content) {
      $("#report-content").textContent = "等待首次运行…";
      $("#report-date").textContent = "—";
      return;
    }
    $("#report-content").textContent = content;
    $("#report-date").textContent = data.report.report_date || "";
  }

  async function loadAll(showErrors = true) {
    if (state.loading) return;
    state.loading = true;
    try {
      const results = await Promise.allSettled([
        getJSON("/api/overview"),
        getJSON("/api/candles?days=240"),
        getJSON("/api/metrics"),
        getJSON("/api/failures?limit=80"),
        getJSON("/api/news?limit=50"),
        getJSON("/api/report"),
        getJSON("/api/predictions?mode=all&limit=80"),
        getJSON("/api/validation"),
      ]);
      const [overview, candles, metrics, failures, news, report, predictions, validation] = results;
      if (overview.status === "fulfilled") renderOverview(overview.value);
      if (candles.status === "fulfilled") renderPriceChart(candles.value);
      if (metrics.status === "fulfilled") {
        renderStats(metrics.value);
        renderAccuracyChart(metrics.value);
        renderSignificance(metrics.value);
        renderReliability(metrics.value);
      }
      if (failures.status === "fulfilled") renderFailures(failures.value);
      if (news.status === "fulfilled") renderNews(news.value);
      if (report.status === "fulfilled") renderReport(report.value);
      if (predictions.status === "fulfilled") renderPredictions(predictions.value);
      if (validation.status === "fulfilled") renderValidation(validation.value);
      const errors = results.filter((result) => result.status === "rejected");
      if (errors.length && showErrors) {
        toast(`部分数据加载失败：${errors[0].reason.message}`, "error");
      }
      state.lastLoaded = new Date();
    } finally {
      state.loading = false;
    }
  }

  async function runUpdate() {
    if (state.updating) return;
    state.updating = true;
    const button = $("#refresh-btn");
    button.disabled = true;
    button.textContent = "更新中…";
    try {
      const start = await getJSON("/api/run", { method: "POST" });
      toast("已启动每日流程：拉数据 → 结算 → 重训 → 预测", "success");
      const jobId = start.job_id;
      const startedAt = Date.now();
      while (Date.now() - startedAt < 15 * 60 * 1000) {
        await new Promise((resolve) => setTimeout(resolve, 2500));
        const job = await getJSON(`/api/job/${jobId}`);
        if (job.status === "done") {
          toast("更新完成，正在刷新界面", "success");
          await loadAll(false);
          break;
        }
        if (job.status === "error") {
          throw new Error(job.error || "更新失败");
        }
        button.textContent = job.status === "running" ? "训练中…" : "排队中…";
      }
    } catch (error) {
      console.error(error);
      toast(`更新失败：${error.message}`, "error");
    } finally {
      state.updating = false;
      button.disabled = false;
      button.textContent = "立即更新";
    }
  }

  function init() {
    initPriceChart();
    initAccuracyChart();
    $("#refresh-btn").addEventListener("click", runUpdate);
    document.querySelectorAll(".horizon-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        state.horizon = Number(tab.dataset.horizon) === 7 ? 7 : 1;
        renderPredictionCard();
      });
    });
    loadAll();
    window.setInterval(() => {
      if (!state.updating && document.visibilityState === "visible") loadAll(false);
    }, 60000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

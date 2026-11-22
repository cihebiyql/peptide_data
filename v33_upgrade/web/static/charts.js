/**
 * Theme- & language-aware zero-dependency crisp SVG visualizations for Peptide OmniPanel 1.1:
 *  - 6-axis single-molecule ADMET Radar Chart (Light + Dark mode, ZH + EN)
 *  - Multi-molecule Batch/Task Comparison Radar Overlay (up to 4 series)
 *  - 80% Conformal Prediction Interval Range Bar (regression)
 *  - Calibrated Probability Bar (classification)
 *  - Interactive 17 Clinical Cases Parity Scatter Chart (Predicted vs FDA T½ log10 h)
 */

const SERIES_COLORS_LIGHT = [
  { stroke: "#0369A1", fill: "rgba(3, 105, 161, 0.16)" },
  { stroke: "#0D9488", fill: "rgba(13, 148, 136, 0.15)" },
  { stroke: "#D97706", fill: "rgba(217, 119, 6, 0.14)" },
  { stroke: "#7C3AED", fill: "rgba(124, 58, 237, 0.14)" },
];

const SERIES_COLORS_DARK = [
  { stroke: "#38BDF8", fill: "rgba(56, 189, 248, 0.20)" },
  { stroke: "#2DD4BF", fill: "rgba(45, 212, 191, 0.19)" },
  { stroke: "#FBBF24", fill: "rgba(251, 191, 36, 0.18)" },
  { stroke: "#A78BFA", fill: "rgba(167, 139, 250, 0.18)" },
];

const STATUS_DOT_COLOR = {
  favorable: "#10B981",
  moderate: "#F59E0B",
  caution: "#F43F5E",
  info: "#0EA5E9",
};

const CASE17_ZH_MAP = {
  Semaglutide: "司美格鲁肽",
  Tirzepatide: "替尔泊肽",
  Liraglutide: "利拉鲁肽",
  Exenatide: "艾塞那肽",
  Leuprolide: "亮丙瑞林",
  Bivalirudin: "比伐卢定",
  Glucagon: "胰高血糖素",
  Teriparatide: "特立帕肽",
  Cyclosporine_A: "环孢素 A",
  Oxytocin: "缩宫素",
  Octreotide: "奥曲肽",
  Linaclotide: "利那洛肽",
  Ziconotide: "齐考诺肽",
  Daptomycin: "达托霉素",
  Caspofungin: "卡泊芬净",
  Insulin_detemir: "地特胰岛素",
  Insulin_degludec: "德谷胰岛素",
};

const CYCLIC_DRUGS = new Set([
  "Cyclosporine_A",
  "Daptomycin",
  "Caspofungin",
  "Octreotide",
  "Oxytocin",
  "Linaclotide",
  "Ziconotide",
]);

function isDarkMode() {
  return document.documentElement.getAttribute("data-theme") === "dark";
}

function chartPalette() {
  const dark = isDarkMode();
  return {
    dark,
    outerFill: dark ? "#162033" : "#FAFBFD",
    gridStroke: dark ? "#2B3A54" : "#E2E8F0",
    axisStroke: dark ? "#3A4E6E" : "#CBD5E1",
    bandFill: dark ? "rgba(56, 189, 248, 0.08)" : "rgba(3, 105, 161, 0.055)",
    labelFill: dark ? "#CBD5E1" : "#334155",
    tickFill: dark ? "#94A3B8" : "#64748B",
    accentStroke: dark ? "#38BDF8" : "#0369A1",
    accentFill: dark ? "rgba(56, 189, 248, 0.19)" : "rgba(3, 105, 161, 0.14)",
    dotRing: dark ? "#0F172A" : "#FFFFFF",
    lipColor: dark ? "#FBBF24" : "#D97706",
    cycColor: dark ? "#38BDF8" : "#0369A1",
    linColor: dark ? "#2DD4BF" : "#0D9488",
    series: dark ? SERIES_COLORS_DARK : SERIES_COLORS_LIGHT,
  };
}

function polarPoint(cx, cy, radius, angleRad) {
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad),
  };
}

function axisLabel(ax, lang) {
  if (lang === "zh" && ax.label_zh) return ax.label_zh;
  return ax.label_en === "Lipophilicity Balance" ? "Lipophilicity" : ax.label_en;
}

export function renderAdmetRadar(containerEl, axes, readoutEl, lang = "en") {
  if (!containerEl || !axes || !axes.length) return;
  const pal = chartPalette();
  const W = 396;
  const H = 256;
  const cx = 198;
  const cy = 128;
  const R = 74;
  const n = axes.length;
  const levels = [25, 50, 75, 100];
  const angleAt = (i) => -Math.PI / 2 + (2 * Math.PI * i) / n;

  let gridSvg = "";
  levels.forEach((lvl) => {
    const r = (R * lvl) / 100;
    const pts = axes
      .map((_, i) => {
        const p = polarPoint(cx, cy, r, angleAt(i));
        return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      })
      .join(" ");
    gridSvg += `<polygon points="${pts}" fill="${lvl === 100 ? pal.outerFill : "none"}" stroke="${pal.gridStroke}" stroke-width="${lvl === 100 ? 1.2 : 0.9}" stroke-dasharray="${lvl === 50 ? "3,3" : "none"}" />`;
  });

  let spokesSvg = "";
  let labelsSvg = "";
  axes.forEach((ax, i) => {
    const ang = angleAt(i);
    const outer = polarPoint(cx, cy, R, ang);
    spokesSvg += `<line x1="${cx}" y1="${cy}" x2="${outer.x.toFixed(1)}" y2="${outer.y.toFixed(1)}" stroke="${pal.gridStroke}" stroke-width="1" />`;

    const lp = polarPoint(cx, cy, R + 18, ang);
    let anchor = "middle";
    let dx = 0;
    if (Math.cos(ang) > 0.25) {
      anchor = "start";
      dx = 4;
    } else if (Math.cos(ang) < -0.25) {
      anchor = "end";
      dx = -4;
    }
    let dy = 3;
    if (Math.sin(ang) < -0.7) dy = -4;
    if (Math.sin(ang) > 0.7) dy = 11;

    labelsSvg += `
      <text x="${(lp.x + dx).toFixed(1)}" y="${(lp.y + dy).toFixed(1)}" text-anchor="${anchor}"
            font-family="Inter, PingFang SC, Microsoft YaHei, sans-serif" font-size="11" font-weight="600" fill="${pal.labelFill}">
        ${escapeHtml(axisLabel(ax, lang))}
      </text>`;
  });

  const dataPts = axes.map((ax, i) => polarPoint(cx, cy, (R * ax.score) / 100, angleAt(i)));
  const polyStr = dataPts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

  const dotsSvg = axes
    .map((ax, i) => {
      const p = dataPts[i];
      const dotColor = STATUS_DOT_COLOR[ax.status] || pal.accentStroke;
      return `
        <g class="radar-node" data-idx="${i}" style="cursor:pointer;">
          <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="10" fill="transparent" />
          <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.5" fill="${dotColor}" stroke="${pal.dotRing}" stroke-width="1.8" />
        </g>`;
    })
    .join("");

  containerEl.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="6-axis ADMET Overview Radar Chart">
      ${gridSvg}
      ${spokesSvg}
      <polygon points="${polyStr}" fill="${pal.accentFill}" stroke="${pal.accentStroke}" stroke-width="2" stroke-linejoin="round" />
      ${dotsSvg}
      ${labelsSvg}
      <text x="${cx + 3}" y="${cy - R * 0.5 + 3}" font-family="JetBrains Mono, monospace" font-size="8.5" fill="${pal.tickFill}">50</text>
      <text x="${cx + 3}" y="${cy - R + 10}" font-family="JetBrains Mono, monospace" font-size="8.5" fill="${pal.tickFill}">100</text>
    </svg>`;

  if (readoutEl) {
    const defaultAxis = axes[2] || axes[0];
    const scoreWord = lang === "zh" ? "评分" : "Score";
    const showAxis = (ax) => {
      const title = lang === "zh" ? `${ax.label_zh} (${ax.label_en})` : `${ax.label_en} (${ax.label_zh})`;
      readoutEl.innerHTML = `<strong>${escapeHtml(title)}</strong> · ${scoreWord} <span class="mono">${ax.score}/100</span> · <span class="mono">${escapeHtml(ax.raw_display)}</span>`;
    };
    showAxis(defaultAxis);
    containerEl.querySelectorAll(".radar-node").forEach((node) => {
      node.addEventListener("mouseenter", () => {
        const idx = Number(node.getAttribute("data-idx"));
        if (axes[idx]) showAxis(axes[idx]);
      });
    });
  }
}

export function renderBatchRadarOverlay(containerEl, legendEl, selectedMolecules, lang = "en", emptyText = null) {
  if (!containerEl) return;
  if (!selectedMolecules || !selectedMolecules.length) {
    const msg =
      emptyText ||
      (lang === "zh"
        ? "请在左侧结果表中勾选 1–4 个多肽，即可叠加对比六维 ADMET 雷达画像。"
        : "Select 1–4 molecules in the matrix to overlay their 6-axis ADMET profiles.");
    containerEl.innerHTML = `<div style="color:var(--ink-muted);font-size:12.5px;text-align:center;padding:30px;">${escapeHtml(msg)}</div>`;
    if (legendEl) legendEl.innerHTML = "";
    return;
  }

  const refAxes = selectedMolecules[0].radar || [];
  if (!refAxes.length) return;
  const pal = chartPalette();

  const W = 408;
  const H = 278;
  const cx = 204;
  const cy = 139;
  const R = 82;
  const n = refAxes.length;
  const angleAt = (i) => -Math.PI / 2 + (2 * Math.PI * i) / n;

  let gridSvg = "";
  [25, 50, 75, 100].forEach((lvl) => {
    const r = (R * lvl) / 100;
    const pts = refAxes
      .map((_, i) => {
        const p = polarPoint(cx, cy, r, angleAt(i));
        return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      })
      .join(" ");
    gridSvg += `<polygon points="${pts}" fill="${lvl === 100 ? pal.outerFill : "none"}" stroke="${pal.gridStroke}" stroke-width="1" />`;
  });

  let spokesSvg = "";
  let labelsSvg = "";
  refAxes.forEach((ax, i) => {
    const ang = angleAt(i);
    const outer = polarPoint(cx, cy, R, ang);
    spokesSvg += `<line x1="${cx}" y1="${cy}" x2="${outer.x.toFixed(1)}" y2="${outer.y.toFixed(1)}" stroke="${pal.gridStroke}" stroke-width="1" />`;
    const lp = polarPoint(cx, cy, R + 18, ang);
    let anchor = "middle";
    let dx = 0;
    if (Math.cos(ang) > 0.25) {
      anchor = "start";
      dx = 4;
    } else if (Math.cos(ang) < -0.25) {
      anchor = "end";
      dx = -4;
    }
    let dy = 3;
    if (Math.sin(ang) < -0.7) dy = -4;
    if (Math.sin(ang) > 0.7) dy = 11;
    labelsSvg += `<text x="${(lp.x + dx).toFixed(1)}" y="${(lp.y + dy).toFixed(1)}" text-anchor="${anchor}" font-family="Inter, PingFang SC, Microsoft YaHei, sans-serif" font-size="11" font-weight="600" fill="${pal.labelFill}">${escapeHtml(axisLabel(ax, lang))}</text>`;
  });

  let seriesSvg = "";
  let legendHtml = "";
  selectedMolecules.slice(0, 4).forEach((mol, sIdx) => {
    const col = pal.series[sIdx % pal.series.length];
    const pts = (mol.radar || []).map((ax, i) => polarPoint(cx, cy, (R * ax.score) / 100, angleAt(i)));
    if (!pts.length) return;
    const polyStr = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    seriesSvg += `<polygon points="${polyStr}" fill="${col.fill}" stroke="${col.stroke}" stroke-width="2" stroke-linejoin="round" />`;
    pts.forEach((p) => {
      seriesSvg += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.2" fill="${col.stroke}" stroke="${pal.dotRing}" stroke-width="1.2" />`;
    });
    legendHtml += `
      <span class="radar-series-chip">
        <span class="radar-series-dot" style="background:${col.stroke}"></span>
        ${escapeHtml(mol.name)}
      </span>`;
  });

  containerEl.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Multi-Molecule ADMET Radar Comparison">
      ${gridSvg}
      ${spokesSvg}
      ${seriesSvg}
      ${labelsSvg}
    </svg>`;
  if (legendEl) legendEl.innerHTML = legendHtml;
}

/**
 * Interactive 17 Clinical Cases Benchmark Parity Chart (Predicted vs FDA Clinical Half-Life, log10 h).
 * Uses real audited drugs from `/api/v1/endpoints` (`case17_scorecard.drugs`).
 */
export function renderCase17ParityChart(containerEl, readoutEl, drugs, { mode = "hybrid", lang = "zh", compact = false, onSelectDrug = null } = {}) {
  if (!containerEl) return;
  const rows = (drugs || []).filter((d) => d.fda_half_life_h != null && d.fda_half_life_h > 0);
  if (!rows.length) {
    containerEl.innerHTML = `<div class="empty-inline">${lang === "zh" ? "正在加载临床验证数据…" : "Loading clinical benchmark…"}</div>`;
    return;
  }

  const pal = chartPalette();
  const W = compact ? 540 : 620;
  const H = compact ? 276 : 312;
  const pad = { l: 54, r: 24, t: 20, b: 44 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  const minLog = -1.35; // ~0.045 h
  const maxLog = 2.45;  // ~280 h
  const span = maxLog - minLog;
  const xOf = (lg) => pad.l + ((lg - minLog) / span) * plotW;
  const yOf = (lg) => pad.t + plotH - ((lg - minLog) / span) * plotH;

  const ticks = [
    { lg: -1, lbl: "0.1h" },
    { lg: 0, lbl: "1h" },
    { lg: 1, lbl: "10h" },
    { lg: 2, lbl: "100h" },
  ];

  let gridSvg = "";
  ticks.forEach((tk) => {
    const x = xOf(tk.lg);
    const y = yOf(tk.lg);
    gridSvg += `<line x1="${x.toFixed(1)}" y1="${pad.t}" x2="${x.toFixed(1)}" y2="${pad.t + plotH}" stroke="${pal.gridStroke}" stroke-width="0.9" stroke-dasharray="3 3" />`;
    gridSvg += `<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${pad.l + plotW}" y2="${y.toFixed(1)}" stroke="${pal.gridStroke}" stroke-width="0.9" stroke-dasharray="3 3" />`;
    gridSvg += `<text x="${x.toFixed(1)}" y="${pad.t + plotH + 16}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="10" fill="${pal.tickFill}">${tk.lbl}</text>`;
    gridSvg += `<text x="${pad.l - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-family="JetBrains Mono, monospace" font-size="10" fill="${pal.tickFill}">${tk.lbl}</text>`;
  });

  // ±0.48 log10 (3-fold) accuracy band around y = x
  const foldBand = 0.477;
  const clamp = (v) => Math.max(minLog, Math.min(maxLog, v));
  const bandPts = [
    [xOf(minLog), yOf(clamp(minLog + foldBand))],
    [xOf(maxLog - foldBand), yOf(maxLog)],
    [xOf(maxLog), yOf(maxLog)],
    [xOf(maxLog), yOf(clamp(maxLog - foldBand))],
    [xOf(minLog + foldBand), yOf(minLog)],
    [xOf(minLog), yOf(minLog)],
  ]
    .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
    .join(" ");

  const diagLine = `<line x1="${xOf(minLog).toFixed(1)}" y1="${yOf(minLog).toFixed(1)}" x2="${xOf(maxLog).toFixed(1)}" y2="${yOf(maxLog).toFixed(1)}" stroke="${pal.accentStroke}" stroke-width="1.35" stroke-dasharray="5 4" opacity="0.7" />`;

  // Highlighted landmark labels (carefully spaced so zero overlap)
  const labelOffsets = {
    Semaglutide: { dx: -10, dy: -9, anchor: "end" },
    Tirzepatide: { dx: -10, dy: 14, anchor: "end" },
    Insulin_degludec: { dx: 9, dy: 3, anchor: "start" },
    Cyclosporine_A: { dx: -9, dy: -8, anchor: "end" },
    Caspofungin: { dx: 9, dy: 4, anchor: "start" },
    Leuprolide: { dx: 9, dy: 4, anchor: "start" },
    Glucagon: { dx: 9, dy: 4, anchor: "start" },
    Oxytocin: { dx: 9, dy: -3, anchor: "start" },
  };

  let arrowsSvg = "";
  let pointsSvg = "";
  let labelsSvg = "";

  rows.forEach((d, idx) => {
    const fdaH = Number(d.fda_half_life_h);
    const pureH = Number(d.v33_half_life_route_matched_h ?? d.v33_half_life_iv_h);
    const hybH = Number(d.v33_half_life_hybrid_h ?? pureH);
    const predH = mode === "pure" ? pureH : hybH;
    if (!predH || predH <= 0) return;

    const xLg = Math.log10(fdaH);
    const yLg = Math.log10(predH);
    const yPureLg = Math.log10(pureH);

    const cx = xOf(clamp(xLg));
    const cy = yOf(clamp(yLg));
    const isLip = d.v33_half_life_mechanism_anchor?.anchor_class && d.v33_half_life_mechanism_anchor.anchor_class !== "non_acylated";
    const isCyc = CYCLIC_DRUGS.has(d.drug_name);
    const col = isLip ? pal.lipColor : isCyc ? pal.cycColor : pal.linColor;

    // Show pure-ML -> anchor shift vector for lipidated drugs in hybrid mode
    if (mode === "hybrid" && isLip && Math.abs(hybH - pureH) > 0.5) {
      const cyPure = yOf(clamp(yPureLg));
      arrowsSvg += `
        <line x1="${cx.toFixed(1)}" y1="${cyPure.toFixed(1)}" x2="${cx.toFixed(1)}" y2="${cy.toFixed(1)}"
              stroke="${col}" stroke-width="1.35" stroke-dasharray="2.5 2" opacity="0.65"/>
        <circle cx="${cx.toFixed(1)}" cy="${cyPure.toFixed(1)}" r="2.7" fill="none" stroke="${col}" stroke-width="1.3" opacity="0.65"/>`;
    }

    const dispName = d.drug_name.replace(/_/g, " ");
    pointsSvg += `
      <g class="c17-dot" data-idx="${idx}" data-drug="${escapeHtml(dispName)}" style="cursor:pointer;">
        <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="11" fill="transparent"/>
        <circle class="c17-dot-core" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${isLip ? 5.6 : 4.8}" fill="${col}" stroke="${pal.dotRing}" stroke-width="1.8"/>
      </g>`;

    const lo = labelOffsets[d.drug_name];
    if (lo) {
      const shortLbl =
        lang === "zh"
          ? CASE17_ZH_MAP[d.drug_name] || dispName
          : d.drug_name === "Insulin_degludec"
            ? "Degludec"
            : d.drug_name === "Cyclosporine_A"
              ? "CsA"
              : dispName;
      labelsSvg += `
        <text x="${(cx + lo.dx).toFixed(1)}" y="${(cy + lo.dy).toFixed(1)}" text-anchor="${lo.anchor}"
              font-family="Inter, PingFang SC, Microsoft YaHei, sans-serif" font-size="10.2" font-weight="600" fill="${pal.labelFill}" pointer-events="none">
          ${escapeHtml(shortLbl)}
        </text>`;
    }
  });

  const xAxisTitle = lang === "zh" ? "FDA 临床半衰期 T½ (小时, log₁₀ 刻度)" : "FDA Clinical Half-Life T½ (h, log₁₀ scale)";
  const yAxisTitle = lang === "zh" ? "OmniPanel 预测 T½ (h)" : "Predicted T½ (h)";
  const bandTag = lang === "zh" ? "±3× 误差带" : "±3× fold band";

  containerEl.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" class="c17-parity-svg" role="img" aria-label="17 Clinical Cases Half-Life Parity Plot">
      <rect x="${pad.l}" y="${pad.t}" width="${plotW}" height="${plotH}" rx="6" fill="${pal.outerFill}" stroke="${pal.axisStroke}" stroke-width="1"/>
      <polygon points="${bandPts}" fill="${pal.bandFill}"/>
      ${gridSvg}
      ${diagLine}
      <text x="${(pad.l + 10).toFixed(1)}" y="${(pad.t + 16).toFixed(1)}" text-anchor="start" font-family="JetBrains Mono, monospace" font-size="9.5" fill="${pal.tickFill}">${escapeHtml(bandTag)}</text>
      ${arrowsSvg}
      ${labelsSvg}
      ${pointsSvg}
      <text x="${(pad.l + plotW / 2).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-family="Inter, PingFang SC, sans-serif" font-size="11" font-weight="600" fill="${pal.labelFill}">${escapeHtml(xAxisTitle)}</text>
      <text x="15" y="${(pad.t + plotH / 2).toFixed(1)}" text-anchor="middle" transform="rotate(-90 15 ${(pad.t + plotH / 2).toFixed(1)})" font-family="Inter, PingFang SC, sans-serif" font-size="11" font-weight="600" fill="${pal.labelFill}">${escapeHtml(yAxisTitle)}</text>
    </svg>`;

  const formatReadout = (d) => {
    const dispEn = d.drug_name.replace(/_/g, " ");
    const zh = CASE17_ZH_MAP[d.drug_name] || dispEn;
    const title = lang === "zh" ? `${zh} (${dispEn})` : `${dispEn}`;
    const fdaStr = `${Number(d.fda_half_life_h).toFixed(d.fda_half_life_h < 1 ? 2 : 1)} h`;
    const pureStr = `${Number(d.v33_half_life_route_matched_h).toFixed(1)} h`;
    const hybStr = `${Number(d.v33_half_life_hybrid_h).toFixed(1)} h`;
    const hemPct = `${(Number(d.v33_hemolysis_prob_cal || 0) * 100).toFixed(1)}%`;
    const route = d.clinical_route || "IV";
    if (lang === "zh") {
      return `<strong>${escapeHtml(title)}</strong> [${escapeHtml(route)}] · FDA <span class="mono">${fdaStr}</span> · 纯ML <span class="mono">${pureStr}</span> · 锚定后 <span class="mono" style="color:var(--cobalt)">${hybStr}</span> · 溶血 <span class="mono" style="color:var(--emerald)">${hemPct}</span>`;
    }
    return `<strong>${escapeHtml(title)}</strong> [${escapeHtml(route)}] · FDA <span class="mono">${fdaStr}</span> · Pure-ML <span class="mono">${pureStr}</span> · Anchored <span class="mono" style="color:var(--cobalt)">${hybStr}</span> · Hemolysis <span class="mono" style="color:var(--emerald)">${hemPct}</span>`;
  };

  if (readoutEl) {
    const sema = rows.find((r) => r.drug_name === "Semaglutide") || rows[0];
    if (sema) readoutEl.innerHTML = formatReadout(sema);
  }

  containerEl.querySelectorAll(".c17-dot").forEach((dot) => {
    const idx = Number(dot.getAttribute("data-idx"));
    const d = rows[idx];
    if (!d) return;
    dot.addEventListener("mouseenter", () => {
      if (readoutEl) readoutEl.innerHTML = formatReadout(d);
    });
    dot.addEventListener("click", () => {
      if (readoutEl) readoutEl.innerHTML = formatReadout(d);
      if (typeof onSelectDrug === "function") {
        onSelectDrug(d.drug_name.replace(/_/g, " "));
      }
    });
  });
}

export function buildIntervalBarHtml(pred, lang = "en") {
  if (pred.value === null || pred.value === undefined) {
    return `<span class="ci-text">—</span>`;
  }

  if (pred.kind === "classification") {
    const pct = Math.max(1, Math.min(99, (pred.calibrated_prob ?? pred.value) * 100));
    const fillCol = STATUS_DOT_COLOR[pred.risk_level] || "#0EA5E9";
    const calPrefix = lang === "zh" ? "校准概率" : "Cal. P";
    const calMeta =
      pred.brier_holdout != null && pred.ece_holdout != null
        ? `Holdout Brier ${pred.brier_holdout.toFixed(3)} · ECE ${pred.ece_holdout.toFixed(3)}`
        : "Platt / Isotonic Calibrated Probability";
    return `
      <div class="ci-track" title="${calPrefix}: ${pct.toFixed(1)}% (${escapeHtml(calMeta)})">
        <div class="prob-fill" style="width:${pct.toFixed(1)}%;background:${fillCol};"></div>
        <div class="ci-dot" style="left:${pct.toFixed(1)}%;background:${fillCol};"></div>
      </div>
      <span class="ci-text">${calPrefix} · ${pct.toFixed(1)}%</span>`;
  }

  if (pred.ci80_low === null || pred.ci80_high === null) {
    return `<span class="ci-text">80% CI —</span>`;
  }
  const bMin = pred.ci_bar_min ?? 0;
  const bMax = pred.ci_bar_max ?? 1;
  const span = Math.max(bMax - bMin, 1e-6);

  const project = (v) => {
    let x = v;
    if (pred.ci_scale === "log10") {
      x = Math.log10(Math.max(v, 1e-12));
    }
    return Math.max(2, Math.min(98, ((x - bMin) / span) * 100));
  };

  const pLow = project(pred.ci80_low);
  const pHigh = project(pred.ci80_high);
  const pVal = project(pred.value);
  const left = Math.min(pLow, pHigh);
  const width = Math.max(4, Math.abs(pHigh - pLow));
  const tip = pred.ci90_display
    ? `80% CI: ${pred.ci80_display} | 90% CI: ${pred.ci90_display}`
    : `80% CI: ${pred.ci80_display || ""}`;

  return `
    <div class="ci-track" title="${escapeHtml(tip)}">
      <div class="ci-band" style="left:${left.toFixed(1)}%;width:${width.toFixed(1)}%;"></div>
      <div class="ci-dot" style="left:${pVal.toFixed(1)}%;"></div>
    </div>
    <span class="ci-text">${escapeHtml(pred.ci80_display || "")}</span>`;
}

export function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

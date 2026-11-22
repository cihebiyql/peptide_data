/**
 * Theme- & language-aware zero-dependency crisp SVG visualizations for Peptide OmniPanel:
 *  - 6-axis single-molecule ADMET Radar Chart (Light + Dark mode, ZH + EN)
 *  - Multi-molecule Batch/Task Comparison Radar Overlay (up to 4 series)
 *  - 80% Conformal Prediction Interval Range Bar (regression)
 *  - Calibrated Probability Bar (classification)
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

function isDarkMode() {
  return document.documentElement.getAttribute("data-theme") === "dark";
}

function chartPalette() {
  const dark = isDarkMode();
  return {
    dark,
    outerFill: dark ? "#162033" : "#FAFBFD",
    gridStroke: dark ? "#2B3A54" : "#E2E8F0",
    labelFill: dark ? "#CBD5E1" : "#334155",
    tickFill: dark ? "#64748B" : "#94A3B8",
    accentStroke: dark ? "#38BDF8" : "#0369A1",
    accentFill: dark ? "rgba(56, 189, 248, 0.19)" : "rgba(3, 105, 161, 0.14)",
    dotRing: dark ? "#0F172A" : "#FFFFFF",
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

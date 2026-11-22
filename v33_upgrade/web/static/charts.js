/**
 * Pure zero-dependency crisp SVG visualizations for Peptide OmniPanel V33:
 *  - 6-axis single-molecule ADMET Radar Chart (generous viewBox, zero label clipping)
 *  - Multi-molecule Batch Comparison Radar Overlay (up to 4 series, zero label clipping)
 *  - 80% Conformal Prediction Interval Range Bar (regression)
 *  - Calibrated Probability Bar (classification)
 */

const SERIES_COLORS = [
  { stroke: "#0369A1", fill: "rgba(3, 105, 161, 0.16)", name: "cobalt" },
  { stroke: "#0D9488", fill: "rgba(13, 148, 136, 0.15)", name: "teal" },
  { stroke: "#D97706", fill: "rgba(217, 119, 6, 0.14)", name: "amber" },
  { stroke: "#7C3AED", fill: "rgba(124, 58, 237, 0.14)", name: "violet" },
];

const STATUS_DOT_COLOR = {
  favorable: "#059669",
  moderate: "#D97706",
  caution: "#E11D48",
  info: "#0369A1",
};

function polarPoint(cx, cy, radius, angleRad) {
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad),
  };
}

/**
 * Render crisp 6-axis single-molecule ADMET radar SVG and wire hover callbacks.
 * Widened viewBox (396 x 256) guarantees generous left/right margin for all axis labels.
 */
export function renderAdmetRadar(containerEl, axes, readoutEl) {
  if (!containerEl || !axes || !axes.length) return;
  const W = 396;
  const H = 256;
  const cx = 198;
  const cy = 128;
  const R = 74;
  const n = axes.length;
  const levels = [25, 50, 75, 100];

  const angleAt = (i) => -Math.PI / 2 + (2 * Math.PI * i) / n;

  // Concentric hex grids
  let gridSvg = "";
  levels.forEach((lvl) => {
    const r = (R * lvl) / 100;
    const pts = axes
      .map((_, i) => {
        const p = polarPoint(cx, cy, r, angleAt(i));
        return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      })
      .join(" ");
    gridSvg += `<polygon points="${pts}" fill="${lvl === 100 ? "#FAFBFD" : "none"}" stroke="#E2E8F0" stroke-width="${lvl === 100 ? 1.2 : 0.9}" stroke-dasharray="${lvl === 50 ? "3,3" : "none"}" />`;
  });

  // Spokes + Labels
  let spokesSvg = "";
  let labelsSvg = "";
  axes.forEach((ax, i) => {
    const ang = angleAt(i);
    const outer = polarPoint(cx, cy, R, ang);
    spokesSvg += `<line x1="${cx}" y1="${cy}" x2="${outer.x.toFixed(1)}" y2="${outer.y.toFixed(1)}" stroke="#E2E8F0" stroke-width="1" />`;

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

    const shortLabel = ax.label_en === "Lipophilicity Balance" ? "Lipophilicity" : ax.label_en;
    labelsSvg += `
      <text x="${(lp.x + dx).toFixed(1)}" y="${(lp.y + dy).toFixed(1)}" text-anchor="${anchor}"
            font-family="Inter, PingFang SC, sans-serif" font-size="11" font-weight="600" fill="#334155">
        ${escapeHtml(shortLabel)}
      </text>`;
  });

  // Data polygon
  const dataPts = axes.map((ax, i) => polarPoint(cx, cy, (R * ax.score) / 100, angleAt(i)));
  const polyStr = dataPts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

  // Interactive vertex dots
  const dotsSvg = axes
    .map((ax, i) => {
      const p = dataPts[i];
      const dotColor = STATUS_DOT_COLOR[ax.status] || "#0369A1";
      return `
        <g class="radar-node" data-idx="${i}" style="cursor:pointer;">
          <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="10" fill="transparent" />
          <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.5" fill="${dotColor}" stroke="#FFFFFF" stroke-width="1.8" />
        </g>`;
    })
    .join("");

  containerEl.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="6-axis ADMET Overview Radar Chart">
      ${gridSvg}
      ${spokesSvg}
      <polygon points="${polyStr}" fill="rgba(3, 105, 161, 0.14)" stroke="#0369A1" stroke-width="2" stroke-linejoin="round" />
      ${dotsSvg}
      ${labelsSvg}
      <text x="${cx + 3}" y="${cy - R * 0.5 + 3}" font-family="JetBrains Mono, monospace" font-size="8.5" fill="#94A3B8">50</text>
      <text x="${cx + 3}" y="${cy - R + 10}" font-family="JetBrains Mono, monospace" font-size="8.5" fill="#94A3B8">100</text>
    </svg>`;

  if (readoutEl) {
    const defaultAxis = axes[2] || axes[0];
    const showAxis = (ax) => {
      readoutEl.innerHTML = `<strong>${escapeHtml(ax.label_en)} (${escapeHtml(ax.label_zh)})</strong> · Score <span class="mono">${ax.score}/100</span> · <span class="mono">${escapeHtml(ax.raw_display)}</span>`;
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

/**
 * Render multi-molecule batch comparison radar overlay (up to 4 series).
 * Widened viewBox (408 x 278) ensures zero clipping on right/left labels.
 */
export function renderBatchRadarOverlay(containerEl, legendEl, selectedMolecules) {
  if (!containerEl) return;
  if (!selectedMolecules || !selectedMolecules.length) {
    containerEl.innerHTML = `<div style="color:#64748B;font-size:12.5px;text-align:center;padding:30px;">Select 1–4 molecules in the matrix to overlay their 6-axis ADMET profiles.</div>`;
    if (legendEl) legendEl.innerHTML = "";
    return;
  }

  const refAxes = selectedMolecules[0].radar || [];
  if (!refAxes.length) return;

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
    gridSvg += `<polygon points="${pts}" fill="${lvl === 100 ? "#FAFBFD" : "none"}" stroke="#E2E8F0" stroke-width="1" />`;
  });

  let spokesSvg = "";
  let labelsSvg = "";
  refAxes.forEach((ax, i) => {
    const ang = angleAt(i);
    const outer = polarPoint(cx, cy, R, ang);
    spokesSvg += `<line x1="${cx}" y1="${cy}" x2="${outer.x.toFixed(1)}" y2="${outer.y.toFixed(1)}" stroke="#E2E8F0" stroke-width="1" />`;
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
    const shortLabel = ax.label_en === "Lipophilicity Balance" ? "Lipophilicity" : ax.label_en;
    labelsSvg += `<text x="${(lp.x + dx).toFixed(1)}" y="${(lp.y + dy).toFixed(1)}" text-anchor="${anchor}" font-family="Inter, PingFang SC, sans-serif" font-size="11" font-weight="600" fill="#334155">${escapeHtml(shortLabel)}</text>`;
  });

  let seriesSvg = "";
  let legendHtml = "";
  selectedMolecules.slice(0, 4).forEach((mol, sIdx) => {
    const col = SERIES_COLORS[sIdx % SERIES_COLORS.length];
    const pts = (mol.radar || []).map((ax, i) => polarPoint(cx, cy, (R * ax.score) / 100, angleAt(i)));
    if (!pts.length) return;
    const polyStr = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    seriesSvg += `<polygon points="${polyStr}" fill="${col.fill}" stroke="${col.stroke}" stroke-width="2" stroke-linejoin="round" />`;
    pts.forEach((p) => {
      seriesSvg += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.2" fill="${col.stroke}" stroke="#FFFFFF" stroke-width="1.2" />`;
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
 * Render either:
 *  - Calibrated probability meter bar for classification endpoints
 *  - 80% Conformal prediction interval range bar for regression endpoints
 */
export function buildIntervalBarHtml(pred) {
  if (pred.value === null || pred.value === undefined) {
    return `<span class="ci-text">—</span>`;
  }

  // Classification: calibrated probability bar (0% -> 100%)
  if (pred.kind === "classification") {
    const pct = Math.max(1, Math.min(99, (pred.calibrated_prob ?? pred.value) * 100));
    const fillCol = STATUS_DOT_COLOR[pred.risk_level] || "#0369A1";
    const calMeta =
      pred.brier_holdout != null && pred.ece_holdout != null
        ? `Holdout Brier ${pred.brier_holdout.toFixed(3)} · ECE ${pred.ece_holdout.toFixed(3)}`
        : "Platt / Isotonic Calibrated Probability";
    return `
      <div class="ci-track" title="Calibrated Probability: ${pct.toFixed(1)}% (${escapeHtml(calMeta)})">
        <div class="prob-fill" style="width:${pct.toFixed(1)}%;background:${fillCol};"></div>
        <div class="ci-dot" style="left:${pct.toFixed(1)}%;background:${fillCol};"></div>
      </div>
      <span class="ci-text">Cal. P · ${pct.toFixed(1)}%</span>`;
  }

  // Regression: 80% Conformal interval range bar
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
    ? `80% Conformal CI: ${pred.ci80_display} | 90% CI: ${pred.ci90_display}`
    : `80% Conformal CI: ${pred.ci80_display || ""}`;

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

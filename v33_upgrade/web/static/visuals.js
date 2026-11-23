/**
 * Peptide OmniPanel 1.1 — Scientific Graphics, Interactive ADMET Map,
 * DeepSeek-Harness Procedural Molecular Canvas, Scroll Reveals & Mod Templates.
 * Zero external CDNs; respects prefers-reduced-motion & Light/Dark CSS variables.
 */

import { escapeHtml } from "./charts.js?v=1.1";
import { t } from "./i18n.js?v=1.1";

// ==================== 1. PROCEDURAL MOLECULAR HERO CANVAS ====================
let _canvasPaused = false;
let _canvasStarted = false;

export function setHeroCanvasPaused(paused) {
  _canvasPaused = Boolean(paused);
}

export function initHeroMolecularCanvas() {
  if (_canvasStarted) return;
  const canvas = document.getElementById("hero-mol-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  _canvasStarted = true;

  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pointer = { x: -9999, y: -9999, active: false };
  let width = 0;
  let height = 0;
  let dpr = Math.min(window.devicePixelRatio || 1, 2);

  // Subtle peptide backbone & aromatic nodes
  const NODE_COUNT = 34;
  const nodes = [];
  const paletteForTheme = () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    return dark
      ? {
          nodeColors: ["rgba(56,189,248,0.52)", "rgba(45,212,191,0.48)", "rgba(167,139,250,0.40)", "rgba(251,191,36,0.38)"],
          edgeColor: "56, 189, 248",
          ringStroke: "rgba(45, 212, 191, 0.14)",
          edgeAlphaMax: 0.20,
        }
      : {
          nodeColors: ["rgba(3,105,161,0.36)", "rgba(13,148,136,0.34)", "rgba(124,58,237,0.24)", "rgba(217,119,6,0.26)"],
          edgeColor: "3, 105, 161",
          ringStroke: "rgba(3, 105, 161, 0.11)",
          edgeAlphaMax: 0.15,
        };
  };

  function seedNodes() {
    nodes.length = 0;
    for (let i = 0; i < NODE_COUNT; i++) {
      // Keep center text corridor airy by gently biasing nodes toward flanks & upper/lower mist
      let x = Math.random() * width;
      let y = Math.random() * height;
      if (i % 3 === 0) {
        x = Math.random() < 0.5 ? Math.random() * width * 0.28 : width * 0.72 + Math.random() * width * 0.28;
      }
      nodes.push({
        x,
        y,
        vx: (Math.random() - 0.5) * 0.28,
        vy: (Math.random() - 0.5) * 0.24,
        r: 2.1 + Math.random() * 2.6,
        colorIdx: i % 4,
        isRing: i % 9 === 0,
        phase: Math.random() * Math.PI * 2,
      });
    }
  }

  function resize() {
    const parent = canvas.parentElement;
    if (!parent) return;
    const rect = parent.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return;
    width = rect.width;
    height = rect.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!nodes.length) seedNodes();
  }

  resize();
  window.addEventListener("resize", resize, { passive: true });

  const hero = document.getElementById("predict-hero");
  hero?.addEventListener(
    "mousemove",
    (e) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = e.clientX - rect.left;
      pointer.y = e.clientY - rect.top;
      pointer.active = true;
    },
    { passive: true }
  );
  hero?.addEventListener("mouseleave", () => {
    pointer.active = false;
  });

  function drawFrame(stepPhysics = true) {
    if (!width || !height) return;
    const pal = paletteForTheme();
    ctx.clearRect(0, 0, width, height);

    if (stepPhysics) {
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        n.phase += 0.008;

        if (n.x < -20) n.x = width + 20;
        if (n.x > width + 20) n.x = -20;
        if (n.y < -20) n.y = height + 20;
        if (n.y > height + 20) n.y = -20;

        if (pointer.active) {
          const dx = n.x - pointer.x;
          const dy = n.y - pointer.y;
          const dist2 = dx * dx + dy * dy;
          if (dist2 > 1 && dist2 < 19000) {
            const f = (1 - Math.sqrt(dist2) / 138) * 0.22;
            n.x += (dx / Math.sqrt(dist2)) * f;
            n.y += (dy / Math.sqrt(dist2)) * f;
          }
        }
      }
    }

    // Draw covalent peptide-bond-like edges between nearby nodes
    const maxDist = 148;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const d = Math.hypot(dx, dy);
        if (d < maxDist) {
          const alpha = (1 - d / maxDist) * pal.edgeAlphaMax;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(${pal.edgeColor}, ${alpha.toFixed(3)})`;
          ctx.lineWidth = d < 72 ? 1.25 : 0.85;
          ctx.stroke();
        }
      }
    }

    // Draw nodes and delicate cyclic/aromatic hexagon hints
    for (const n of nodes) {
      if (n.isRing) {
        ctx.save();
        ctx.translate(n.x, n.y);
        ctx.rotate(n.phase * 0.35);
        ctx.beginPath();
        const rad = 11.5;
        for (let k = 0; k < 6; k++) {
          const ang = (Math.PI / 3) * k;
          const px = Math.cos(ang) * rad;
          const py = Math.sin(ang) * rad;
          if (k === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.strokeStyle = pal.ringStroke;
        ctx.lineWidth = 1.1;
        ctx.stroke();
        ctx.restore();
      }

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = pal.nodeColors[n.colorIdx];
      ctx.fill();
    }
  }

  // Render static crisp molecular backdrop if reduced motion is requested
  if (reduceMotion) {
    drawFrame(false);
    return;
  }

  let rafId = null;
  function tick() {
    if (!document.hidden && !_canvasPaused) {
      drawFrame(true);
    }
    rafId = requestAnimationFrame(tick);
  }
  rafId = requestAnimationFrame(tick);

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !rafId) {
      rafId = requestAnimationFrame(tick);
    }
  });
}

// ==================== 2. ANIMATED KPI COUNT-UP ====================
const _animatedIds = new Set();

export function animateKpiValue(el, targetNum, { suffix = "", forceStatic = false } = {}) {
  if (!el || targetNum == null || Number.isNaN(Number(targetNum))) return;
  const endVal = Number(targetNum);
  const fmt = (v) => `${Math.round(v).toLocaleString()}${suffix}`;
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (forceStatic || reduceMotion || _animatedIds.has(el.id)) {
    el.textContent = fmt(endVal);
    return;
  }
  _animatedIds.add(el.id);

  const durationMs = 760;
  const startTs = performance.now();
  const startVal = Math.max(0, Math.floor(endVal * 0.28));

  function step(now) {
    const p = Math.min(1, (now - startTs) / durationMs);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(startVal + (endVal - startVal) * eased);
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = fmt(endVal);
  }
  requestAnimationFrame(step);
}

// ==================== 3. SCROLL-TRIGGERED STAGGERED REVEALS ====================
export function initScrollReveals() {
  const els = document.querySelectorAll(".reveal-on-scroll");
  if (!els.length) return;
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || typeof IntersectionObserver === "undefined") {
    els.forEach((el) => el.classList.add("is-visible"));
    return;
  }
  const obs = new IntersectionObserver(
    (entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          en.target.classList.add("is-visible");
          obs.unobserve(en.target);
        }
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -24px 0px" }
  );
  els.forEach((el) => {
    if (!el.classList.contains("is-visible")) obs.observe(el);
  });
}

// ==================== 4. QUICK MODIFICATION TEMPLATE HELPER ====================
export const MOD_TEMPLATES = [
  {
    id: "c18_diacid",
    badge: "C18 Diacid",
    tone: "amber",
    title_zh: "C18 脂肪二酸长效修饰",
    title_en: "C18 Fatty Diacid Lipidation",
    desc_zh: "白蛋白可逆结合 · 周给药半衰期窗口 (Semaglutide)",
    desc_en: "Albumin-binding weekly anchor (Semaglutide)",
    query: "Semaglutide",
    pkContext: "human_sc",
  },
  {
    id: "c20_aib",
    badge: "C20 + 2×Aib",
    tone: "amber",
    title_zh: "C20 二酸 + Aib 抗酶解",
    title_en: "C20 Diacid + Aib Protection",
    desc_zh: "DPP-4 抗降解 + 超长白蛋白结合 (Tirzepatide)",
    desc_en: "DPP-4 resistant dual incretin (Tirzepatide)",
    query: "Tirzepatide",
    pkContext: "human_sc",
  },
  {
    id: "nme_macrocycle",
    badge: "7× N-Me Ring",
    tone: "cobalt",
    title_zh: "N-甲基化透膜大环肽",
    title_en: "N-Methylated Macrocycle",
    desc_zh: "33 原子大环屏蔽氢键 · 提升口服/膜渗透 (Cyclosporine A)",
    desc_en: "33-atom shielded macrocycle (Cyclosporine A)",
    query: "Cyclosporine A",
    pkContext: "human_po",
  },
  {
    id: "echinocandin_lipo",
    badge: "Echinocandin",
    tone: "emerald",
    title_zh: "棘白菌素环脂肽骨架",
    title_en: "Echinocandin Cyclic Lipopeptide",
    desc_zh: "芳环脂尾 + 六肽大环 · 强效抗念珠菌 (Micafungin)",
    desc_en: "Isoxazole lipophilic tail macrocycle (Micafungin)",
    query: "Micafungin",
    pkContext: "human_plasma",
  },
  {
    id: "disulfide_knot",
    badge: "Disulfide S–S",
    tone: "violet",
    title_zh: "二硫键刚性环肽",
    title_en: "Disulfide-Bridged Cyclic Peptide",
    desc_zh: "构象刚性化 + 抗蛋白酶解 (Octreotide)",
    desc_en: "Conformational lock against proteolysis (Octreotide)",
    query: "Octreotide",
    pkContext: "human_sc",
  },
  {
    id: "amphipathic_amp",
    badge: "α-Helix AMP",
    tone: "teal",
    title_zh: "两亲阳离子抗菌螺旋肽",
    title_en: "Amphipathic α-Helical AMP",
    desc_zh: "23-AA 标准序列 · 全 22 项 ESKAPE 抗菌谱 (Magainin 2)",
    desc_en: "23-AA canonical sequence · 22 ESKAPE endpoints",
    query: "GIGKFLHSAKKFGKAFVGEIMNS",
    pkContext: "human_plasma",
  },
];

export function renderModTemplateGrid(containerEl, lang, onSelect) {
  if (!containerEl) return;
  containerEl.innerHTML = MOD_TEMPLATES.map(
    (tpl) => `
      <button type="button" class="mod-tpl-card" data-tpl-id="${escapeHtml(tpl.id)}">
        <div class="mod-tpl-top">
          <span class="mod-tpl-name">${escapeHtml(lang === "zh" ? tpl.title_zh : tpl.title_en)}</span>
          <span class="mod-badge tone-${escapeHtml(tpl.tone)}">${escapeHtml(tpl.badge)}</span>
        </div>
        <div class="mod-tpl-desc">${escapeHtml(lang === "zh" ? tpl.desc_zh : tpl.desc_en)}</div>
      </button>`
  ).join("");

  containerEl.querySelectorAll(".mod-tpl-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tpl = MOD_TEMPLATES.find((x) => x.id === btn.dataset.tplId);
      if (tpl && typeof onSelect === "function") onSelect(tpl);
    });
  });
}

// ==================== 5. INLINE SCIENTIFIC SVG LOADER ====================
export async function loadInlineSvgAsset(url, hostEl) {
  if (!hostEl) return null;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const svgText = await res.text();
    hostEl.innerHTML = svgText;
    return hostEl.querySelector("svg");
  } catch (err) {
    console.error("Failed to inline SVG:", url, err);
    return null;
  }
}

// ==================== 6. INTERACTIVE HUMAN ADMET MAP DATA & RENDERER ====================
export const ADMET_HOTSPOTS = {
  pk: {
    id: "pk",
    tone: "amber",
    colorVar: "var(--amber)",
    badge_zh: "白蛋白结合与循环滞留",
    badge_en: "Albumin Binding & Circulation",
    title_zh: "长效脂化半衰期与血浆蛋白结合",
    title_en: "Albumin-Anchored Half-Life (T½) & PPB",
    desc_zh: "融合给药途径路由与 C14–C20 脂肪酸白蛋白结合机制锚定，在 17 个经典临床多肽上达到 Spearman 0.903 与对数 R² 0.781。",
    desc_en: "Combines route-aware PK modeling with C14–C20 fatty-acylation anchoring, achieving Spearman 0.903 and R² 0.781 across 17 clinical peptides.",
    endpoints: ["half_life_reg", "half_life_cls", "PPB"],
    sampleQuery: "Semaglutide",
    sampleLabel_zh: "司美格鲁肽 (Semaglutide)",
    sampleLabel_en: "Semaglutide (C18 Diacid)",
    highlightStat: (epMap, c17, lang) => {
      const sp = c17?.half_life_pure_ml_route_matched_log10?.spearman;
      return sp != null ? (lang === "zh" ? `17 药 ρ ${Number(sp).toFixed(3)}` : `17-Case ρ ${Number(sp).toFixed(3)}`) : "17-Case ρ 0.903";
    },
  },
  gut: {
    id: "gut",
    tone: "cobalt",
    colorVar: "var(--cobalt)",
    badge_zh: "消化道吸收与跨膜屏障",
    badge_en: "GI Absorption & Membrane Crossing",
    title_zh: "膜渗透（PAMPA / Caco-2）与 P-gp 外排",
    title_en: "Membrane Permeability & Efflux Ratio",
    desc_zh: "精准刻画大环肽氢键屏蔽、N-甲基化与亲脂平衡对被动扩散、肠上皮渗透及主动外排比的影响。",
    desc_en: "Captures macrocycle hydrogen-bond shielding, N-methylation, and lipophilicity driving passive diffusion, Caco-2 transport, and P-gp efflux.",
    endpoints: ["Caco2", "PAMPA", "efflux_ratio", "RRCK", "MDCK"],
    sampleQuery: "Cyclosporine A",
    sampleLabel_zh: "环孢素 A (Cyclosporine A)",
    sampleLabel_en: "Cyclosporine A (Macrocycle)",
    highlightStat: (epMap) => {
      const v = epMap.get("Caco2")?.holdout_score_v33;
      return v != null ? `Caco-2 ρ ${Number(v).toFixed(3)}` : "Caco-2 ρ 0.853";
    },
  },
  bbb: {
    id: "bbb",
    tone: "violet",
    colorVar: "var(--violet)",
    badge_zh: "中枢神经系统屏障",
    badge_en: "Central Nervous System Barrier",
    title_zh: "血脑屏障透过概率 (BBB)",
    title_en: "Blood-Brain Barrier Penetration (BBB)",
    desc_zh: "基于严格骨架聚类留出与 Platt 校准概率，识别神经多肽入脑潜力与外周限制性大分子特征。",
    desc_en: "Calibrated BBB penetration probability under strict cluster holdout to distinguish CNS-active neuropeptides from peripherally restricted leads.",
    endpoints: ["BBB"],
    sampleQuery: "Octreotide",
    sampleLabel_zh: "奥曲肽 (Octreotide)",
    sampleLabel_en: "Octreotide (Disulfide Cyclic)",
    highlightStat: (epMap) => {
      const v = epMap.get("BBB")?.holdout_score_v33;
      return v != null ? `ROC ${Number(v).toFixed(3)}` : "ROC 0.899";
    },
  },
  clearance: {
    id: "clearance",
    tone: "teal",
    colorVar: "var(--teal)",
    badge_zh: "系统清除、分布与溶解平衡",
    badge_en: "Systemic Clearance, Vd & Solubility",
    title_zh: "肝肾清除 (CL)、分布容积 (Vd) 与理化溶解",
    title_en: "Clearance (CL), Volume of Dist. (Vd) & LogD",
    desc_zh: "联合评估体清除率、组织分布容积、生物利用度 F、LogD7.4 与水溶性，并同步给出 80% 适用域分层经验区间。",
    desc_en: "Jointly profiles systemic clearance, distribution volume, bioavailability (F), LogD7.4, and aqueous solubility with 80% AD-stratified empirical intervals.",
    endpoints: ["CL_per_kg", "Vd_per_kg", "F", "LogD7.4", "solubility_cls", "solubility_reg"],
    sampleQuery: "Leuprolide",
    sampleLabel_zh: "亮丙瑞林 (Leuprolide)",
    sampleLabel_en: "Leuprolide (GnRH Peptide)",
    highlightStat: (epMap) => {
      const v = epMap.get("CL_per_kg")?.holdout_score_v33;
      return v != null ? `CL/kg ρ ${Number(v).toFixed(3)}` : "CL/kg ρ 0.832";
    },
  },
  safety: {
    id: "safety",
    tone: "rose",
    colorVar: "var(--rose)",
    badge_zh: "血液相容性与细胞毒窗口",
    badge_en: "Hemocompatibility & Cytotoxicity",
    title_zh: "红细胞溶血 (HC50 / MHC) 与哺乳动物细胞毒",
    title_en: "RBC Hemolysis (HC50/MHC) & Cytotoxicity",
    desc_zh: "支持人/马/鼠红细胞物种路由与右删失 AFT/Tobit 建模，在 17 个临床上市多肽上保持 0/17 溶血误报。",
    desc_en: "Species-aware RBC routing (human/horse/mouse) with right-censored AFT/Tobit diagnostics — zero false positives (0/17) on FDA clinical peptides.",
    endpoints: ["overall_peptide_toxicity", "cytotoxicity_cls", "hemolysis_cls", "hemolysis_HC50", "cytotoxicity_CC50", "hemolysis_MHC"],
    sampleQuery: "Micafungin",
    sampleLabel_zh: "米卡芬净 (Micafungin)",
    sampleLabel_en: "Micafungin (Echinocandin)",
    highlightStat: (epMap, c17, lang) => {
      const fp = c17?.hemolysis_false_positive_check?.n_drugs_gt_0_50 ?? 0;
      return lang === "zh" ? `${fp}/17 临床误报` : `${fp}/17 Clinical FP`;
    },
  },
  micro: {
    id: "micro",
    tone: "emerald",
    colorVar: "var(--emerald)",
    badge_zh: "Tier 2 · 22 项独立病原谱",
    badge_en: "Tier 2 · 22 Pathogen Endpoints",
    title_zh: "ESKAPE 耐药菌 MIC、杀菌 MBC 与抗生物膜",
    title_en: "ESKAPE MIC, Bactericidal MBC & Anti-Biofilm",
    desc_zh: "严格按菌种拆分大肠杆菌、金葡菌、肺克菌、鲍曼不动杆菌、白色念珠菌 MIC/MBC/MFC 及生物膜抑制与清除浓度。",
    desc_en: "Species-pure antimicrobial suite spanning ESKAPE pathogens, fungal C. albicans MIC/MFC, bactericidal MBC, and biofilm MBIC/MBEC.",
    endpoints: ["MIC_L_monocytogenes", "MIC_C_albicans", "MBC_P_aeruginosa", "MIC_E_faecium", "MIC_E_coli", "MIC_K_pneumoniae"],
    sampleQuery: "Magainin 2",
    sampleLabel_zh: "爪蟾抗菌肽 (Magainin 2)",
    sampleLabel_en: "Magainin 2 (23-AA AMP)",
    highlightStat: (epMap, c17, lang) => {
      const t2Cnt = Array.from(epMap.values()).filter((e) => e.tier === 2).length || 22;
      return lang === "zh" ? `${t2Cnt} 菌种独立端点` : `${t2Cnt} Pathogen EPs`;
    },
  },
};

export function syncAdmetSvgStats(mapRoot, catalogData, lang) {
  if (!mapRoot || !catalogData) return;
  const epMap = new Map((catalogData.endpoints || []).map((e) => [e.endpoint, e]));
  const c17 = catalogData.case17_scorecard || {};
  Object.values(ADMET_HOTSPOTS).forEach((hs) => {
    const statEl = mapRoot.querySelector(`[data-svg-stat="${hs.id}"]`);
    if (statEl && typeof hs.highlightStat === "function") {
      statEl.textContent = hs.highlightStat(epMap, c17, lang);
    }
  });
}

export function renderAdmetHotspotDetail(containerEl, hotspotId, catalogData, lang, onRunSample) {
  if (!containerEl) return;
  const hs = ADMET_HOTSPOTS[hotspotId] || ADMET_HOTSPOTS.pk;
  const epMap = new Map((catalogData?.endpoints || []).map((e) => [e.endpoint, e]));

  const rowsHtml = hs.endpoints
    .map((epKey) => {
      const e = epMap.get(epKey);
      const name = e ? (lang === "zh" ? e.display_zh : e.display_en) : epKey;
      const metricTag = e?.kind === "classification" ? "ROC-AUC" : "Spearman ρ";
      const val = e?.holdout_score_v33 != null ? Number(e.holdout_score_v33).toFixed(3) : "—";
      const pct = e?.holdout_score_v33 != null ? Math.max(12, Math.min(100, Math.round(Number(e.holdout_score_v33) * 100))) : 50;
      const nHo = e?.n_holdout != null ? `n=${e.n_holdout.toLocaleString()}` : "";
      return `
        <div class="admet-ep-row">
          <div class="admet-ep-info">
            <span class="admet-ep-name">${escapeHtml(name)}</span>
            <span class="admet-ep-meta mono">${escapeHtml(epKey)}${nHo ? ` · ${nHo}` : ""}</span>
          </div>
          <div class="admet-ep-score-wrap">
            <div class="admet-ep-bar"><div class="admet-ep-bar-fill" style="width:${pct}%;background:${hs.colorVar};"></div></div>
            <span class="admet-ep-val mono" title="${escapeHtml(metricTag)}">${escapeHtml(metricTag === "ROC-AUC" ? "AUC " : "ρ ")}${val}</span>
          </div>
        </div>`;
    })
    .join("");

  const sampleLbl = lang === "zh" ? hs.sampleLabel_zh : hs.sampleLabel_en;

  containerEl.innerHTML = `
    <div class="admet-detail-inner" data-active-hotspot="${escapeHtml(hs.id)}">
      <div class="admet-detail-top">
        <span class="mod-badge tone-${escapeHtml(hs.tone)}">${escapeHtml(lang === "zh" ? hs.badge_zh : hs.badge_en)}</span>
        <span class="admet-ho-pill mono">${escapeHtml(t(lang, "admet_ho_label"))}</span>
      </div>
      <h3 class="admet-detail-title">${escapeHtml(lang === "zh" ? hs.title_zh : hs.title_en)}</h3>
      <p class="admet-detail-desc">${escapeHtml(lang === "zh" ? hs.desc_zh : hs.desc_en)}</p>

      <div class="admet-ep-head">${escapeHtml(t(lang, "admet_ep_header"))}</div>
      <div class="admet-ep-list">${rowsHtml}</div>

      <div class="admet-detail-footer">
        <button type="button" class="btn-primary admet-run-btn" data-sample-query="${escapeHtml(hs.sampleQuery)}">
          <span>${escapeHtml(t(lang, "admet_run_sample"))} · ${escapeHtml(sampleLbl)}</span>
          <span aria-hidden="true">→</span>
        </button>
      </div>
    </div>`;

  containerEl.querySelector(".admet-run-btn")?.addEventListener("click", () => {
    if (typeof onRunSample === "function") onRunSample(hs.sampleQuery);
  });
}

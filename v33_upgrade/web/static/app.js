/**
 * Peptide OmniPanel 1.2 — Frontend SPA Controller
 * Features:
 *  - Atmospheric DeepSeek-Harness Hero, Procedural Molecular Canvas & Scroll Reveals
 *  - 4 Clinical Modality Showcase Cards, Interactive Human ADMET Barrier Map & 4-FM Animated Pipeline
 *  - Interactive 17 Clinical Cases Parity Chart & Plan B Atom-Attribution Toggle Hook
 *  - Full Bilingual (中文 / English) & Light / Dark Theme switching
 *  - Dedicated Persistent Task Center with real-time chunked progress, column presets & exports
 *  - Seamless Batch/Task <-> Compound Detail navigation (0ms cached drill-down, Back, Prev/Next, Jump, Esc/Arrows/Browser Back)
 */

import { buildIntervalBarHtml, escapeHtml, renderAdmetRadar, renderBatchRadarOverlay, renderCase17ParityChart } from "./charts.js?v=1.2";
import { applyStaticI18n, applySvgI18n, getInitialLang, t, trAd, trRisk } from "./i18n.js?v=1.2";
import {
  animateKpiValue,
  initHeroMolecularCanvas,
  initScrollReveals,
  loadInlineSvgAsset,
  renderAdmetHotspotDetail,
  renderModTemplateGrid,
  setHeroCanvasPaused,
  syncAdmetSvgStats,
} from "./visuals.js?v=1.2";

const API_BASE = (() => {
  const p = window.location.pathname || "/";
  if (p === "/" || p === "") return "";
  return p.endsWith("/") ? p.slice(0, -1) : p.replace(/\/[^/]*$/, "");
})();

function apiUrl(path) {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${clean}`;
}

function getClientId() {
  let id = localStorage.getItem("omnipanel_client_id");
  if (!id || !/^[A-Za-z0-9_\-]{6,64}$/.test(id)) {
    const rand = (crypto?.randomUUID ? crypto.randomUUID().replace(/-/g, "").slice(0, 16) : Math.random().toString(36).slice(2, 14));
    id = `cli_${rand}`;
    localStorage.setItem("omnipanel_client_id", id);
  }
  return id;
}

function apiFetch(path, opts = {}) {
  const headers = { ...(opts.headers || {}), "X-Client-Id": getClientId() };
  return fetch(apiUrl(path), { ...opts, headers });
}

const DOMAIN_META = {
  pk: { en: "Pharmacokinetics & PK", zh: "药代动力学", dot: "dot-pk" },
  permeability: { en: "Membrane Permeability & Efflux", zh: "膜渗透与外排", dot: "dot-permeability" },
  safety: { en: "Safety, Hemolysis & Cytotoxicity", zh: "安全性、溶血与细胞毒", dot: "dot-safety" },
  physicochemical: { en: "Physicochemical & In Vitro", zh: "理化性质与溶解度", dot: "dot-physicochemical" },
  antimicrobial: { en: "Antimicrobial, Bactericidal & Biofilm", zh: "抗菌谱、杀菌与抗生物膜", dot: "dot-antimicrobial" },
};

const FM_DISPLAY_NAMES = {
  ncaa_repdistill_v2: { label: "ncAA-RepDistill v2", modality: "SMILES · 768+2048+1280d", color: "#0369A1" },
  esmc_600m: { label: "ESM-C 600M (4-Region)", modality: "Sequence · 4×1152d", color: "#0D9488" },
  peptideclm2: { label: "PeptideCLM-2", modality: "SMILES · 512d", color: "#7C3AED" },
  chemberta_mtr: { label: "ChemBERTa-MTR", modality: "SMILES · 384d", color: "#D97706" },
};

const COLUMN_PRESETS = {
  overview: [
    { key: "mw", label_en: "MW", label_zh: "分子量", num: true, get: (r) => (r.mw ? r.mw.toFixed(0) : "—"), sortVal: (r) => r.mw ?? -Infinity },
    { key: "half_life_val", label_en: "Half-Life", label_zh: "半衰期", num: true, anchorStyle: true, get: (r) => r.key_metrics?.half_life || "—", sortVal: (r) => r.key_metrics?.half_life_val ?? -Infinity },
    { key: "ppb_val", label_en: "PPB", label_zh: "蛋白结合", num: true, get: (r) => r.key_metrics?.ppb || "—", sortVal: (r) => r.key_metrics?.ppb_val ?? -Infinity },
    { key: "pampa_val", label_en: "PAMPA", label_zh: "PAMPA", num: true, get: (r) => r.key_metrics?.pampa || "—", sortVal: (r) => r.key_metrics?.pampa_val ?? -Infinity },
    { key: "caco2_val", label_en: "Caco-2", label_zh: "Caco-2", num: true, get: (r) => r.key_metrics?.caco2 || "—", sortVal: (r) => r.key_metrics?.caco2_val ?? -Infinity },
    { key: "hemolysis", label_en: "Hemolysis", label_zh: "溶血风险", riskCell: true, get: (r) => r.key_metrics?.hemolysis_risk || "—", sortVal: (r) => r.key_metrics?.hemolysis_risk_val ?? -Infinity },
    { key: "logd", label_en: "LogD7.4", label_zh: "LogD7.4", num: true, get: (r) => r.key_metrics?.logd || "—", sortVal: (r) => parseFloat(r.key_metrics?.logd) || -Infinity },
  ],
  pk: [
    { key: "mw", label_en: "MW", label_zh: "分子量", num: true, get: (r) => (r.mw ? r.mw.toFixed(0) : "—"), sortVal: (r) => r.mw ?? -Infinity },
    { key: "half_life_val", label_en: "Half-Life", label_zh: "半衰期", num: true, anchorStyle: true, get: (r) => r.key_metrics?.half_life || "—", sortVal: (r) => r.key_metrics?.half_life_val ?? -Infinity },
    { key: "ppb_val", label_en: "PPB", label_zh: "蛋白结合", num: true, get: (r) => r.key_metrics?.ppb || "—", sortVal: (r) => r.key_metrics?.ppb_val ?? -Infinity },
    { key: "cl_per_kg", label_en: "CL/kg", label_zh: "清除率", num: true, get: (r) => r.key_metrics?.cl_per_kg || "—", sortVal: (r) => r.key_metrics?.cl_per_kg_val ?? -Infinity },
    { key: "vd_per_kg", label_en: "Vd/kg", label_zh: "分布容积", num: true, get: (r) => r.key_metrics?.vd_per_kg || "—", sortVal: (r) => r.key_metrics?.vd_per_kg_val ?? -Infinity },
    { key: "bioavail_f", label_en: "F", label_zh: "生物利用度", num: true, get: (r) => r.key_metrics?.bioavail_f || "—", sortVal: (r) => r.key_metrics?.bioavail_f_val ?? -Infinity },
    { key: "cmax", label_en: "Cmax", label_zh: "峰浓度", num: true, get: (r) => r.key_metrics?.cmax || "—", sortVal: (r) => r.key_metrics?.cmax_val ?? -Infinity },
  ],
  perm: [
    { key: "pampa_val", label_en: "PAMPA", label_zh: "PAMPA", num: true, get: (r) => r.key_metrics?.pampa || "—", sortVal: (r) => r.key_metrics?.pampa_val ?? -Infinity },
    { key: "caco2_val", label_en: "Caco-2", label_zh: "Caco-2", num: true, get: (r) => r.key_metrics?.caco2 || "—", sortVal: (r) => r.key_metrics?.caco2_val ?? -Infinity },
    { key: "mdck", label_en: "MDCK", label_zh: "MDCK", num: true, get: (r) => r.key_metrics?.mdck || "—", sortVal: (r) => parseFloat(r.key_metrics?.mdck) || -Infinity },
    { key: "efflux_ratio", label_en: "Efflux ER", label_zh: "外排比", num: true, get: (r) => r.key_metrics?.efflux_ratio || "—", sortVal: (r) => r.key_metrics?.efflux_ratio_val ?? -Infinity },
    { key: "bbb", label_en: "BBB", label_zh: "血脑屏障", num: true, get: (r) => r.key_metrics?.bbb || "—", sortVal: (r) => parseFloat(r.key_metrics?.bbb) || -Infinity },
    { key: "logd", label_en: "LogD7.4", label_zh: "LogD7.4", num: true, get: (r) => r.key_metrics?.logd || "—", sortVal: (r) => parseFloat(r.key_metrics?.logd) || -Infinity },
    { key: "solubility", label_en: "Solubility", label_zh: "水溶性 µM", num: true, get: (r) => r.key_metrics?.solubility || "—", sortVal: (r) => parseFloat(r.key_metrics?.solubility) || -Infinity },
  ],
  safety: [
    { key: "hemolysis", label_en: "Hemolysis", label_zh: "溶血概率", riskCell: true, get: (r) => r.key_metrics?.hemolysis_risk || "—", sortVal: (r) => r.key_metrics?.hemolysis_risk_val ?? -Infinity },
    { key: "hc50", label_en: "HC50 (µM)", label_zh: "溶血 HC50", num: true, get: (r) => r.key_metrics?.hc50 || "—", sortVal: (r) => r.key_metrics?.hc50_val ?? -Infinity },
    { key: "mhc", label_en: "MHC/HC10", label_zh: "溶血 MHC", num: true, get: (r) => r.key_metrics?.mhc || "—", sortVal: (r) => parseFloat(r.key_metrics?.mhc) || -Infinity },
    { key: "toxicity", label_en: "Toxicity", label_zh: "综合毒性", num: true, get: (r) => r.key_metrics?.toxicity || "—", sortVal: (r) => parseFloat(r.key_metrics?.toxicity) || -Infinity },
    { key: "cyto_risk", label_en: "Cyto Risk", label_zh: "细胞毒概率", num: true, get: (r) => r.key_metrics?.cyto_risk || "—", sortVal: (r) => parseFloat(r.key_metrics?.cyto_risk) || -Infinity },
    { key: "cyto_cc50", label_en: "CC50 (µM)", label_zh: "细胞毒 CC50", num: true, get: (r) => r.key_metrics?.cytotoxicity_cc50 || "—", sortVal: (r) => r.key_metrics?.cytotoxicity_cc50_val ?? -Infinity },
  ],
  micro: [
    { key: "best_mic", label_en: "Best MIC", label_zh: "最强 MIC", num: true, get: (r) => r.key_metrics?.best_mic || "—", sortVal: (r) => r.key_metrics?.best_mic_val ?? Infinity },
    { key: "mic_ecoli", label_en: "E. coli", label_zh: "大肠杆菌", num: true, get: (r) => r.key_metrics?.mic_ecoli || "—", sortVal: (r) => parseFloat(r.key_metrics?.mic_ecoli) || Infinity },
    { key: "mic_saureus", label_en: "S. aureus", label_zh: "金葡菌", num: true, get: (r) => r.key_metrics?.mic_saureus || "—", sortVal: (r) => parseFloat(r.key_metrics?.mic_saureus) || Infinity },
    { key: "mic_paeruginosa", label_en: "P. aerug.", label_zh: "铜绿假单胞", num: true, get: (r) => r.key_metrics?.mic_paeruginosa || "—", sortVal: (r) => parseFloat(r.key_metrics?.mic_paeruginosa) || Infinity },
    { key: "mic_calbicans", label_en: "C. alb.", label_zh: "白念珠菌", num: true, get: (r) => r.key_metrics?.mic_calbicans || "—", sortVal: (r) => parseFloat(r.key_metrics?.mic_calbicans) || Infinity },
    { key: "mic_kpneumoniae", label_en: "K. pneum.", label_zh: "肺克菌", num: true, get: (r) => r.key_metrics?.mic_kpneumoniae || "—", sortVal: (r) => parseFloat(r.key_metrics?.mic_kpneumoniae) || Infinity },
    { key: "mic_abaumannii", label_en: "A. baum.", label_zh: "鲍曼不动", num: true, get: (r) => r.key_metrics?.mic_abaumannii || "—", sortVal: (r) => parseFloat(r.key_metrics?.mic_abaumannii) || Infinity },
  ],
};

function makeInspectorUiState() {
  return {
    colView: "overview",
    riskFilter: "all",
    search: "",
    sortCol: null,
    sortAsc: true,
    selectedIds: new Set(),
    lastInspectedIdx: null,
  };
}

const state = {
  lang: getInitialLang(),
  theme: "light",
  activeTab: "predict",
  healthData: null,
  examples: [],
  exampleCategories: [],
  drugs: [],
  batchDemoCsv: "",
  batchPresets: {},
  rbcSpecies: "human",
  lastPrediction: null,
  svgCache: new Map(),
  activeTier: 1,
  tier2Group: "all",
  adFilter: "all",
  endpointSearch: "",
  // Seamless Batch/Task -> Compound Detail navigation context
  inspectCtx: null,
  // Batch workspace state
  batchParsed: null,
  batchJobId: null,
  batchJobDetail: null,
  // Tasks Center state
  jobs: [],
  activeJobCount: 0,
  tasksStatusFilter: "all",
  tasksTypeFilter: "all",
  tasksSearch: "",
  selectedTaskId: null,
  selectedTaskDetail: null,
  pollTimer: null,
  // Independent inspector view states for Batch host & Tasks host
  inspectorUi: {
    batch: makeInspectorUiState(),
    tasks: makeInspectorUiState(),
  },
  // Validation state
  catalogData: null,
  valTier: "all",
  valDomain: "all",
  valSearch: "",
  // v1.2 Interactive Visuals & Live GNN Atom Attribution
  activeHotspot: "pk",
  c17Mode: "hybrid",
  svgViewMode: "motifs",
  attrEndpoint: "PAMPA",
  attrSvgCache: new Map(),
  attrLoading: false,
  // API snippets
  codeLang: "curl",
};

const ATTR_ENDPOINT_LABELS = {
  PAMPA: { zh: "PAMPA", en: "PAMPA" },
  Caco2: { zh: "Caco-2", en: "Caco-2" },
  hemolysis_cls: { zh: "溶血风险", en: "Hemolysis" },
  hemolysis_HC50: { zh: "HC50", en: "HC50" },
  overall_peptide_toxicity: { zh: "综合毒性", en: "Toxicity" },
  cytotoxicity_cls: { zh: "细胞毒", en: "Cytotox" },
  cytotoxicity_CC50: { zh: "CC50", en: "CC50" },
  half_life_reg: { zh: "半衰期 T½", en: "Half-Life" },
};

// ==================== BOOTSTRAP ====================
document.addEventListener("DOMContentLoaded", async () => {
  initThemeAndLang();
  wireNavigation();
  wirePredictWorkspace();
  wireLandingVisuals();
  wireBatchWorkspace();
  wireTasksWorkspace();
  wireValidationWorkspace();
  wireApiWorkspace();
  wireSvgModal();
  wireGlobalShortcutsAndHistory();
  initHeroMolecularCanvas();
  initScrollReveals();

  await Promise.all([
    loadHealth(),
    loadExamples(),
    loadEndpointsCatalog(),
    initInlineScientificSvgs(),
    refreshJobsList({ autoSelectFirst: true }),
  ]);
});

// ==================== v1.1 LANDING VISUALS, ADMET MAP & C17 BENCHMARK ====================
function wireLandingVisuals() {
  // 4 Featured Modality Cards -> 1-click live prediction
  document.querySelectorAll(".modality-card").forEach((card) => {
    const launch = () => {
      const q = card.dataset.modalityQuery;
      if (q) runExampleQuery(q);
    };
    card.addEventListener("click", launch);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        launch();
      }
    });
  });

  // Quick Modification Template Helper toggle
  const modBtn = document.getElementById("btn-toggle-modhelper");
  const modPanel = document.getElementById("mod-template-panel");
  modBtn?.addEventListener("click", () => {
    const isHidden = modPanel.classList.toggle("hidden");
    modBtn.setAttribute("aria-expanded", String(!isHidden));
    if (!isHidden) {
      document.getElementById("example-library-panel")?.classList.add("hidden");
      document.getElementById("btn-toggle-exlib")?.setAttribute("aria-expanded", "false");
      document.getElementById("advanced-panel")?.classList.add("hidden");
      document.getElementById("btn-toggle-advanced")?.setAttribute("aria-expanded", "false");
    }
  });
  document.getElementById("btn-close-modhelper")?.addEventListener("click", () => {
    modPanel?.classList.add("hidden");
    modBtn?.setAttribute("aria-expanded", "false");
  });
  renderModTemplateGrid(document.getElementById("mod-template-grid"), state.lang, applyModTemplate);

  // 17 Clinical Cases chart mode toggles (synced between Landing & Validation views)
  document.querySelectorAll("[data-c17mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.c17Mode = btn.dataset.c17mode;
      document.querySelectorAll("[data-c17mode]").forEach((x) => x.classList.toggle("active", x.dataset.c17mode === state.c17Mode));
      renderAllCase17Charts();
    });
  });

  document.getElementById("btn-landing-goto-val")?.addEventListener("click", () => {
    switchTab("validation");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Plan B Structure Card Toggle Hook (Motif Highlights <-> Live Pure-PyTorch E-GAT Atom Attribution Heatmap)
  document.querySelectorAll("#svg-view-toggle .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      const nextMode = btn.dataset.svgMode || "motifs";
      state.svgViewMode = nextMode;
      if (state.lastPrediction) {
        if (nextMode === "attribution" && !getActiveAttributionSvg(state.lastPrediction)) {
          fetchAtomAttributionForCurrent(state.lastPrediction, state.attrEndpoint || "PAMPA");
        } else {
          renderSvgCard(state.lastPrediction);
        }
      }
    });
  });
}

function applyModTemplate(tpl) {
  document.getElementById("mod-template-panel")?.classList.add("hidden");
  document.getElementById("btn-toggle-modhelper")?.setAttribute("aria-expanded", "false");
  const pkSel = document.getElementById("pk-context-select");
  if (pkSel && tpl.pkContext) pkSel.value = tpl.pkContext;
  runExampleQuery(tpl.query);
}

async function initInlineScientificSvgs() {
  const mapHost = document.getElementById("admet-map-host");
  const archHost = document.getElementById("arch-flow-host");

  await Promise.all([
    loadInlineSvgAsset(apiUrl("/static/assets/admet_human_map.svg?v=1.1"), mapHost),
    loadInlineSvgAsset(apiUrl("/static/assets/arch_4fm_flow.svg?v=1.1"), archHost),
  ]);

  if (mapHost) applySvgI18n(state.lang, mapHost);
  if (archHost) applySvgI18n(state.lang, archHost);

  if (mapHost) {
    const activate = (hsId) => {
      state.activeHotspot = hsId;
      mapHost.querySelectorAll(".admet-hotspot").forEach((g) => {
        g.classList.toggle("is-active", g.getAttribute("data-hotspot") === hsId);
      });
      renderAdmetHotspotDetail(
        document.getElementById("admet-detail-card"),
        hsId,
        state.catalogData,
        state.lang,
        (q) => {
          window.scrollTo({ top: 0, behavior: "smooth" });
          runExampleQuery(q);
        }
      );
    };

    mapHost.querySelectorAll(".admet-hotspot").forEach((g) => {
      const hsId = g.getAttribute("data-hotspot");
      if (!hsId) return;
      g.addEventListener("mouseenter", () => activate(hsId));
      g.addEventListener("click", () => activate(hsId));
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activate(hsId);
        }
      });
    });

    if (state.catalogData) syncAdmetSvgStats(mapHost, state.catalogData, state.lang);
    activate(state.activeHotspot);
  }
}

function renderAllCase17Charts() {
  const drugs = state.catalogData?.case17_scorecard?.drugs || [];
  const c17 = state.catalogData?.case17_scorecard || {};
  const hlPure = c17.half_life_pure_ml_route_matched_log10 || {};
  const hlAnch = c17.half_life_with_mechanism_anchor_log10 || {};
  const hem = c17.hemolysis_false_positive_check || {};

  const elRho = document.getElementById("lc17-rho");
  const elR2 = document.getElementById("lc17-r2");
  const elMae = document.getElementById("lc17-mae");
  const elFp = document.getElementById("lc17-fp");
  if (elRho && hlPure.spearman != null) elRho.textContent = Number(hlPure.spearman).toFixed(3);
  if (elR2 && hlAnch.r2 != null) elR2.textContent = Number(hlAnch.r2).toFixed(3);
  if (elMae && hlAnch.mae != null) elMae.textContent = Number(hlAnch.mae).toFixed(3);
  if (elFp) elFp.textContent = `${hem.n_drugs_gt_0_50 ?? 0} / 17`;

  const onSelectDrug = (drugName) => {
    switchTab("predict");
    window.scrollTo({ top: 0, behavior: "smooth" });
    runExampleQuery(drugName);
  };

  renderCase17ParityChart(
    document.getElementById("landing-case17-chart"),
    document.getElementById("landing-case17-readout"),
    drugs,
    { mode: state.c17Mode, lang: state.lang, compact: true, onSelectDrug }
  );
  renderCase17ParityChart(
    document.getElementById("val-case17-chart"),
    document.getElementById("val-case17-readout"),
    drugs,
    { mode: state.c17Mode, lang: state.lang, compact: false, onSelectDrug }
  );
}

// ==================== THEME & LANGUAGE ====================
function initThemeAndLang() {
  const savedTheme = localStorage.getItem("omnipanel_theme");
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(savedTheme === "dark" || savedTheme === "light" ? savedTheme : prefersDark ? "dark" : "light", false);
  applyLanguage(state.lang, false);

  document.querySelectorAll(".lang-btn").forEach((btn) => {
    btn.addEventListener("click", () => applyLanguage(btn.dataset.lang, true));
  });
  document.getElementById("btn-theme-toggle")?.addEventListener("click", () => {
    applyTheme(state.theme === "dark" ? "light" : "dark", true);
  });
}

function applyTheme(theme, reRenderCharts = true) {
  state.theme = theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", state.theme);
  localStorage.setItem("omnipanel_theme", state.theme);
  document.getElementById("icon-sun")?.classList.toggle("hidden", state.theme === "dark");
  document.getElementById("icon-moon")?.classList.toggle("hidden", state.theme !== "dark");

  if (reRenderCharts) {
    if (state.lastPrediction) {
      renderAdmetRadar(
        document.getElementById("res-radar-stage"),
        state.lastPrediction.radar || [],
        document.getElementById("res-radar-readout"),
        state.lang
      );
    }
    if (state.batchJobDetail) renderJobInspector("batch");
    if (state.selectedTaskDetail) renderJobInspector("tasks");
    if (state.catalogData) renderAllCase17Charts();
  }
}

function applyLanguage(lang, reRenderDynamic = true) {
  state.lang = lang === "en" ? "en" : "zh";
  localStorage.setItem("omnipanel_lang", state.lang);
  document.querySelectorAll(".lang-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.lang === state.lang);
  });
  applyStaticI18n(state.lang);

  if (reRenderDynamic) {
    if (state.healthData) hydrateHealthUi(state.healthData, { forceStatic: true });
    renderExamplesPillsAndLibrary();
    renderModTemplateGrid(document.getElementById("mod-template-grid"), state.lang, applyModTemplate);
    if (state.lastPrediction) renderPredictionResult(state.lastPrediction, { keepContext: true });
    if (state.batchJobDetail) renderJobInspector("batch");
    renderTasksQueue();
    if (state.selectedTaskDetail) renderJobInspector("tasks");
    if (state.catalogData) {
      syncAdmetSvgStats(document.getElementById("admet-map-host"), state.catalogData, state.lang);
      renderAdmetHotspotDetail(
        document.getElementById("admet-detail-card"),
        state.activeHotspot,
        state.catalogData,
        state.lang,
        (q) => {
          window.scrollTo({ top: 0, behavior: "smooth" });
          runExampleQuery(q);
        }
      );
      renderAllCase17Charts();
      renderValidationKpis(state.catalogData);
      renderValidationHighlights(state.catalogData);
      renderValidationTable();
    }
    renderCodeSnippet(state.codeLang);
  }
}

// ==================== NAVIGATION & KEYBOARD / HISTORY ====================
function wireNavigation() {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.tab !== "predict") clearInspectContext(false);
      switchTab(btn.dataset.tab);
    });
  });
  document.getElementById("brand-home").addEventListener("click", (e) => {
    e.preventDefault();
    clearInspectContext(false);
    switchTab("predict");
    if (state.lastPrediction) resetToLanding();
  });
}

function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".view-section").forEach((sec) => {
    sec.classList.toggle("active", sec.id === `view-${tab}`);
  });
  setHeroCanvasPaused(tab !== "predict" || Boolean(state.lastPrediction));
  if (tab === "validation" && !state.catalogData) {
    loadEndpointsCatalog();
  } else if (tab === "validation" && state.catalogData) {
    renderAllCase17Charts();
  }
  if (tab === "tasks") {
    refreshJobsList({ autoSelectFirst: !state.selectedTaskId });
  }
}

function resetToLanding() {
  state.lastPrediction = null;
  clearInspectContext(false);
  document.getElementById("predict-hero").classList.add("hero-centered");
  document.getElementById("predict-hero").classList.remove("hero-compact");
  document.getElementById("compact-actions").classList.add("hidden");
  document.getElementById("landing-overview").classList.remove("hidden");
  document.getElementById("predict-results").classList.add("hidden");
  setHeroCanvasPaused(false);
}

function wireGlobalShortcutsAndHistory() {
  document.addEventListener("keydown", (e) => {
    const modal = document.getElementById("svg-modal");
    if (modal && !modal.classList.contains("hidden")) {
      if (e.key === "Escape") modal.classList.add("hidden");
      return;
    }
    if (!state.inspectCtx || state.activeTab !== "predict") return;
    const tag = (e.target?.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

    if (e.key === "ArrowLeft") {
      e.preventDefault();
      stepInspectContext(-1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      stepInspectContext(1);
    } else if (e.key === "Escape") {
      e.preventDefault();
      returnFromCompoundDetail();
    }
  });

  window.addEventListener("popstate", () => {
    if (state.inspectCtx && state.activeTab === "predict") {
      returnFromCompoundDetail({ fromPopState: true });
    }
  });
}

// ==================== HEALTH & EXAMPLES ====================
async function loadHealth() {
  try {
    const res = await apiFetch("/api/v1/health");
    const data = await res.json();
    state.healthData = data;
    hydrateHealthUi(data);
  } catch {
    const pill = document.getElementById("backend-status");
    pill.classList.add("offline");
    pill.querySelector(".status-text").textContent = t(state.lang, "status_offline");
  }
}

function hydrateHealthUi(data, { forceStatic = false } = {}) {
  const pill = document.getElementById("backend-status");
  const txt = pill.querySelector(".status-text");
  pill.classList.remove("offline", "fallback");
  if (data.adapter_mode === "v33_remote" || data.v33_remote_healthy) {
    txt.textContent = t(state.lang, "status_online", { n: data.endpoints_total });
    pill.title = `Peptide OmniPanel 1.2 · RDKit ${data.rdkit_version}`;
  } else {
    pill.classList.add("fallback");
    txt.textContent = t(state.lang, "status_fallback", { n: data.endpoints_total });
  }

  const ds = data.dataset_summary || {};
  const compEl = document.getElementById("kpi-compounds");
  if (compEl && ds.n_unique_compounds) animateKpiValue(compEl, ds.n_unique_compounds, { forceStatic });
  const epEl = document.getElementById("kpi-ep-count");
  if (epEl && data.endpoints_total) animateKpiValue(epEl, data.endpoints_total, { forceStatic });
  const epLbl = document.getElementById("kpi-ep-lbl");
  if (epLbl && data.tier1_endpoints != null) {
    epLbl.textContent = t(state.lang, "kpi_endpoints", { t1: data.tier1_endpoints, t2: data.tier2_endpoints });
  }
  const advT2 = document.getElementById("adv-tier2-label");
  if (advT2 && data.tier2_endpoints != null) {
    advT2.textContent = t(state.lang, "adv_tier2", { n: data.tier2_endpoints });
  }
  const recEl = document.getElementById("kpi-records");
  if (recEl && ds.n_total_rows) animateKpiValue(recEl, ds.n_total_rows, { forceStatic });
  const recLbl = document.getElementById("kpi-records-lbl");
  if (recLbl && ds.n_censored_rows != null) {
    recLbl.textContent = t(state.lang, "kpi_records", { cens: Number(ds.n_censored_rows).toLocaleString() });
  }
  const uqEl = document.getElementById("kpi-uq");
  if (uqEl) animateKpiValue(uqEl, 80, { suffix: "%", forceStatic });
}

async function loadExamples() {
  try {
    const res = await apiFetch("/api/v1/examples");
    const data = await res.json();
    state.examples = data.quick_examples || [];
    state.exampleCategories = data.example_categories || [];
    state.drugs = data.drugs || [];
    state.batchDemoCsv = data.batch_demo_csv || "";
    state.batchPresets = data.batch_presets || {};
    renderExamplesPillsAndLibrary();
  } catch (err) {
    console.error("Failed to load examples:", err);
  }
}

function renderExamplesPillsAndLibrary() {
  const pillsWrap = document.getElementById("example-pills");
  if (pillsWrap) {
    pillsWrap.innerHTML = `<span class="examples-label">${escapeHtml(t(state.lang, "examples_label"))}</span>`;
    state.examples.forEach((ex) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "example-pill";
      btn.title = ex.subtitle || ex.label;
      btn.innerHTML = `${escapeHtml(ex.label)} <span class="pill-tag">${escapeHtml(ex.tag)}</span>`;
      btn.addEventListener("click", () => runExampleQuery(ex.query));
      pillsWrap.appendChild(btn);
    });
  }
  const btnLbl = document.getElementById("exlib-btn-label");
  if (btnLbl && state.drugs.length) {
    btnLbl.textContent = t(state.lang, "btn_all_examples", { n: state.drugs.length });
  }

  const grid = document.getElementById("exlib-grid");
  if (!grid || !state.drugs.length) return;
  const cats = state.exampleCategories.length
    ? state.exampleCategories
    : [{ id: "all", title_en: "Curated Peptides", title_zh: "精选多肽示例" }];

  grid.innerHTML = cats
    .map((cat) => {
      const items = state.drugs.filter((d) => cat.id === "all" || d.category === cat.id);
      if (!items.length) return "";
      const primaryTitle = state.lang === "zh" ? cat.title_zh : cat.title_en;
      const secondaryTitle = state.lang === "zh" ? cat.title_en : cat.title_zh;
      const chipsHtml = items
        .map((d) => {
          const q = d.query || d.display_name;
          const sub = d.fda_half_life ? `T½: ${d.fda_half_life.split("(")[0].trim()}` : d.drug_class;
          const nameMain = state.lang === "zh" ? d.zh_name : d.display_name;
          const nameSub = state.lang === "zh" ? d.display_name : d.zh_name;
          return `
            <button type="button" class="exlib-chip" data-query="${escapeHtml(q)}" title="${escapeHtml(`${d.display_name} (${d.zh_name}) — ${d.drug_class}`)}">
              <div class="exlib-chip-top">
                <span class="exlib-chip-en">${escapeHtml(nameMain)}</span>
                <span class="exlib-chip-zh">${escapeHtml(nameSub)}</span>
                ${d.case17 ? `<span class="exlib-c17-dot" title="17 Clinical Cases">17</span>` : ""}
              </div>
              <div class="exlib-chip-sub">${escapeHtml(sub)}</div>
            </button>`;
        })
        .join("");
      return `
        <div class="exlib-group">
          <div class="exlib-group-head">
            <span class="exlib-group-title">${escapeHtml(primaryTitle)}</span>
            <span class="exlib-group-en">${escapeHtml(secondaryTitle)}</span>
          </div>
          <div class="exlib-chip-list">${chipsHtml}</div>
        </div>`;
    })
    .join("");

  grid.querySelectorAll(".exlib-chip").forEach((chip) => {
    chip.addEventListener("click", () => runExampleQuery(chip.dataset.query));
  });
}

function runExampleQuery(qVal) {
  clearInspectContext(false);
  document.getElementById("smart-query-input").value = qVal;
  document.getElementById("adv-sequence").value = "";
  document.getElementById("adv-smiles").value = "";
  document.getElementById("example-library-panel")?.classList.add("hidden");
  document.getElementById("btn-toggle-exlib")?.setAttribute("aria-expanded", "false");
  updateLiveDetection(qVal);
  runSinglePrediction();
}

// ==================== PREDICT WORKSPACE ====================
function wirePredictWorkspace() {
  const form = document.getElementById("predict-form");
  const input = document.getElementById("smart-query-input");
  const acMenu = document.getElementById("autocomplete-list");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    acMenu.classList.add("hidden");
    clearInspectContext(false);
    runSinglePrediction();
  });

  input.addEventListener("input", () => {
    const v = input.value.trim();
    updateLiveDetection(v);
    renderAutocomplete(v);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") acMenu.classList.add("hidden");
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".query-bar-wrap")) acMenu.classList.add("hidden");
  });

  // Sticky Detail Return & Next/Prev Bar controls
  document.getElementById("btn-ctx-back")?.addEventListener("click", () => returnFromCompoundDetail());
  document.getElementById("btn-ctx-prev")?.addEventListener("click", () => stepInspectContext(-1));
  document.getElementById("btn-ctx-next")?.addEventListener("click", () => stepInspectContext(1));
  document.getElementById("ctx-mol-jump")?.addEventListener("change", (e) => {
    const idx = Number(e.target.value);
    if (!Number.isNaN(idx)) openCompoundAtContextIndex(idx);
  });

  // Curated Example Library Drawer toggle
  const exlibBtn = document.getElementById("btn-toggle-exlib");
  const exlibPanel = document.getElementById("example-library-panel");
  exlibBtn?.addEventListener("click", () => {
    const isHidden = exlibPanel.classList.toggle("hidden");
    exlibBtn.setAttribute("aria-expanded", String(!isHidden));
    if (!isHidden) {
      document.getElementById("advanced-panel")?.classList.add("hidden");
      document.getElementById("btn-toggle-advanced")?.setAttribute("aria-expanded", "false");
      document.getElementById("mod-template-panel")?.classList.add("hidden");
      document.getElementById("btn-toggle-modhelper")?.setAttribute("aria-expanded", "false");
    }
  });
  document.getElementById("btn-close-exlib")?.addEventListener("click", () => {
    exlibPanel?.classList.add("hidden");
    exlibBtn?.setAttribute("aria-expanded", "false");
  });

  const advBtn = document.getElementById("btn-toggle-advanced");
  const advPanel = document.getElementById("advanced-panel");
  advBtn?.addEventListener("click", () => {
    const isHidden = advPanel.classList.toggle("hidden");
    advBtn.setAttribute("aria-expanded", String(!isHidden));
    if (!isHidden) {
      exlibPanel?.classList.add("hidden");
      exlibBtn?.setAttribute("aria-expanded", "false");
      document.getElementById("mod-template-panel")?.classList.add("hidden");
      document.getElementById("btn-toggle-modhelper")?.setAttribute("aria-expanded", "false");
    }
  });

  document.querySelectorAll("#rbc-species-group .seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#rbc-species-group .seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.rbcSpecies = b.dataset.val;
    });
  });

  document.querySelectorAll(".tier-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.activeTier = Number(tab.dataset.tier);
      document.querySelectorAll(".tier-tab").forEach((x) => x.classList.toggle("active", Number(x.dataset.tier) === state.activeTier));
      document.getElementById("tier1-dashboard").classList.toggle("hidden", state.activeTier !== 1);
      document.getElementById("tier2-dashboard").classList.toggle("hidden", state.activeTier !== 2);
    });
  });

  document.querySelectorAll("#ad-filter-group .filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.adFilter = chip.dataset.ad;
      document.querySelectorAll("#ad-filter-group .filter-chip").forEach((c) => c.classList.toggle("active", c.dataset.ad === state.adFilter));
      if (state.lastPrediction) renderEndpointDashboards(state.lastPrediction);
    });
  });

  document.getElementById("endpoint-search")?.addEventListener("input", (e) => {
    state.endpointSearch = e.target.value.trim().toLowerCase();
    if (state.lastPrediction) renderEndpointDashboards(state.lastPrediction);
  });

  document.getElementById("btn-export-csv")?.addEventListener("click", exportSingleCsv);
  document.getElementById("btn-export-json")?.addEventListener("click", exportSingleJson);
  document.getElementById("btn-print-report")?.addEventListener("click", () => window.print());
}

function updateLiveDetection(raw) {
  const chip = document.getElementById("detected-type-chip");
  const s = (raw || "").trim();
  if (!s) {
    chip.classList.add("hidden");
    return;
  }
  chip.classList.remove("hidden");
  const norm = s.toLowerCase().replace(/[\s_\-]+/g, "");
  const matchedDrug = state.drugs.find(
    (d) => d.key.toLowerCase().replace(/[\s_\-]+/g, "") === norm || d.display_name.toLowerCase().replace(/[\s_\-]+/g, "") === norm || d.zh_name === s
  );
  if (matchedDrug) {
    chip.textContent = state.lang === "zh" ? `药物 · ${matchedDrug.zh_name}` : `Drug · ${matchedDrug.display_name}`;
    return;
  }
  const cleanSeq = s.replace(/\s+/g, "").toUpperCase();
  if (/^[ACDEFGHIKLMNPQRSTVWY]{4,1022}$/.test(cleanSeq) && !/[a-z0-9=()[\]#@\\/+.\-]/.test(s)) {
    chip.textContent = `${state.lang === "zh" ? "序列" : "Seq"} · ${cleanSeq.length} AA`;
    return;
  }
  if (/[=()[\]@#1-9a-z]/.test(s)) {
    chip.textContent = `SMILES · ${s.length}`;
    return;
  }
  chip.textContent = state.lang === "zh" ? "查询" : "Query";
}

function renderAutocomplete(raw) {
  const menu = document.getElementById("autocomplete-list");
  if (!raw || raw.length < 2 || raw.length > 36 || /[=()[\]@#]/.test(raw)) {
    menu.classList.add("hidden");
    return;
  }
  const q = raw.toLowerCase();
  const hits = state.drugs.filter(
    (d) => d.display_name.toLowerCase().includes(q) || d.zh_name.includes(raw) || d.drug_class.toLowerCase().includes(q)
  );
  if (!hits.length) {
    menu.classList.add("hidden");
    return;
  }
  menu.innerHTML = hits
    .slice(0, 7)
    .map(
      (d) => `
      <div class="ac-item" data-query="${escapeHtml(d.display_name)}">
        <div>
          <span class="ac-name">${escapeHtml(d.display_name)}</span>
          <span class="ac-zh">${escapeHtml(d.zh_name)}</span>
        </div>
        <span class="ac-class">${escapeHtml(d.drug_class)}</span>
      </div>`
    )
    .join("");
  menu.classList.remove("hidden");

  menu.querySelectorAll(".ac-item").forEach((item) => {
    item.addEventListener("click", () => {
      menu.classList.add("hidden");
      runExampleQuery(item.dataset.query);
    });
  });
}

async function runSinglePrediction() {
  const errEl = document.getElementById("predict-error");
  errEl.classList.add("hidden");

  const query = document.getElementById("smart-query-input").value.trim();
  const advSeq = document.getElementById("adv-sequence").value.trim();
  const advSmi = document.getElementById("adv-smiles").value.trim();
  const includeTier2 = document.getElementById("toggle-tier2-req").checked;
  const pkContext = document.getElementById("pk-context-select").value;

  if (!query && !advSeq && !advSmi) {
    errEl.textContent =
      state.lang === "zh"
        ? "请输入多肽药物名称、20 种天然氨基酸序列或有效 SMILES 结构。"
        : "Please enter a peptide drug name, canonical 20-AA sequence, or SMILES string.";
    errEl.classList.remove("hidden");
    return;
  }

  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = t(state.lang, "btn_running");
  btn.querySelector(".btn-spinner").classList.remove("hidden");

  try {
    const payload = {
      query: !advSeq && !advSmi ? query : null,
      sequence: advSeq || null,
      smiles: advSmi || null,
      include_tier2: includeTier2,
      include_svg: true,
      conditions: {
        rbc_species: state.rbcSpecies,
        pk_context: pkContext,
      },
    };
    const res = await apiFetch("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction request failed.");
    state.lastPrediction = data;
    renderPredictionResult(data, { keepContext: false });
    // Silently refresh task history so this single run appears in Tasks Center
    refreshJobsList({ autoSelectFirst: false });
  } catch (err) {
    errEl.textContent = err.message || String(err);
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.querySelector(".btn-label").textContent = t(state.lang, "btn_predict");
    btn.querySelector(".btn-spinner").classList.add("hidden");
  }
}

// ==================== SEAMLESS BATCH / TASK <-> DETAIL LOOP ====================
function openCompoundFromInspector(viewKey, jobDetail, molIndex) {
  const okItems = (jobDetail.results || []).filter((r) => r.status === "ok");
  const pos = okItems.findIndex((r) => r.index === molIndex);
  if (pos < 0) return;

  state.inspectorUi[viewKey].lastInspectedIdx = molIndex;
  state.inspectCtx = {
    sourceTab: viewKey === "tasks" ? "tasks" : "batch",
    viewKey,
    jobId: jobDetail.job_id,
    jobTitle: jobDetail.title,
    items: okItems,
    currentIndex: pos,
    savedScrollY: window.scrollY || 0,
  };

  try {
    window.history.pushState({ omnipanelDetail: true, jobId: jobDetail.job_id, pos }, "");
  } catch {}

  switchTab("predict");
  openCompoundAtContextIndex(pos);
  window.scrollTo({ top: 0, behavior: "instant" });
}

function openCompoundAtContextIndex(pos) {
  const ctx = state.inspectCtx;
  if (!ctx || pos < 0 || pos >= ctx.items.length) return;
  ctx.currentIndex = pos;
  const item = ctx.items[pos];
  state.inspectorUi[ctx.viewKey].lastInspectedIdx = item.index;

  const predObj = batchItemToPredictResponse(item);
  state.lastPrediction = predObj;

  const qVal = item.resolved_type === "drug_name" ? item.name : item.smiles || item.sequence || item.name;
  document.getElementById("smart-query-input").value = qVal;
  updateLiveDetection(qVal);
  renderPredictionResult(predObj, { keepContext: true });
  renderDetailContextBar();

  // Fetch only the crisp 2D RDKit SVG asynchronously (~30ms, zero ML re-computation!)
  hydrateSvgForDetailItem(item, predObj);
}

function batchItemToPredictResponse(item) {
  const pc = item.physchem || {
    mw: item.mw || 0,
    logp: item.logp || 0,
    tpsa: 0,
    hbd: 0,
    hba: 0,
    rotatable_bonds: 0,
    ring_count: 0,
    macrocycle_rings: [],
    heavy_atoms: 0,
    formal_charge: 0,
    net_charge_ph74: item.net_charge_ph74 || 0,
    formula: "—",
    length_aa: item.sequence ? item.sequence.length : null,
    frac_csp3: 0,
  };
  const svgKey = item.canonical_smiles || item.smiles || item.sequence || item.name;
  const cachedSvg = state.svgCache.get(svgKey);
  return {
    query_name: item.name,
    resolved_type: item.resolved_type || "smiles",
    modality_track: item.modality_track || "dual_modality",
    sequence: item.sequence,
    smiles: item.smiles,
    canonical_smiles: item.canonical_smiles,
    drug_reference: item.drug_reference || null,
    physchem: pc,
    modifications: item.modification_badges || (item.modifications || []).map((m) => ({ key: m, label_en: m, label_zh: m, tone: "slate" })),
    svg: cachedSvg ? cachedSvg.svg : null,
    svg_highlights: cachedSvg ? cachedSvg.highlights : [],
    mechanism_anchor: item.mechanism_anchor || { detected: false },
    radar: item.radar || [],
    predictions: item.predictions || [],
    summary: item.summary || {},
    runtime_ms: 0,
    cache_hit: true,
    adapter_mode: "task_cached",
  };
}

async function hydrateSvgForDetailItem(item, predObj) {
  const svgKey = item.canonical_smiles || item.smiles || item.sequence || item.name;
  if (state.svgCache.has(svgKey)) return;
  const stage = document.getElementById("mol-svg-stage");
  if (stage) {
    stage.innerHTML = `<div class="svg-loading-hint">${state.lang === "zh" ? "正在渲染二维结构…" : "Rendering 2D structure…"}</div>`;
  }
  try {
    const body = {
      query: !item.smiles && !item.sequence ? item.name : null,
      smiles: item.smiles || item.canonical_smiles || null,
      sequence: !item.smiles && !item.canonical_smiles ? item.sequence : null,
      width: 560,
      height: 340,
    };
    const res = await apiFetch("/api/v1/render-svg", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) return;
    const svgData = await res.json();
    state.svgCache.set(svgKey, { svg: svgData.svg, highlights: svgData.highlights || [] });
    if (state.lastPrediction === predObj) {
      predObj.svg = svgData.svg;
      predObj.svg_highlights = svgData.highlights || [];
      renderSvgCard(predObj);
    }
  } catch {}
}

function renderDetailContextBar() {
  const bar = document.getElementById("detail-context-bar");
  const ctx = state.inspectCtx;
  if (!bar) return;
  if (!ctx) {
    bar.classList.add("hidden");
    return;
  }
  bar.classList.remove("hidden");
  document.getElementById("ctx-back-label").textContent =
    ctx.sourceTab === "tasks" ? t(state.lang, "ctx_back_task") : t(state.lang, "ctx_back_batch");
  document.getElementById("ctx-task-title").textContent = `${ctx.jobTitle} (${ctx.jobId})`;
  document.getElementById("ctx-pos-pill").textContent = t(state.lang, "ctx_pos", {
    cur: ctx.currentIndex + 1,
    tot: ctx.items.length,
  });

  const sel = document.getElementById("ctx-mol-jump");
  sel.innerHTML = ctx.items
    .map((it, i) => `<option value="${i}" ${i === ctx.currentIndex ? "selected" : ""}>#${it.index} · ${escapeHtml(it.name)}</option>`)
    .join("");

  const prevBtn = document.getElementById("btn-ctx-prev");
  const nextBtn = document.getElementById("btn-ctx-next");
  prevBtn.disabled = ctx.currentIndex <= 0;
  nextBtn.disabled = ctx.currentIndex >= ctx.items.length - 1;
  prevBtn.title = ctx.currentIndex > 0 ? `← ${ctx.items[ctx.currentIndex - 1].name}` : "";
  nextBtn.title = ctx.currentIndex < ctx.items.length - 1 ? `→ ${ctx.items[ctx.currentIndex + 1].name}` : "";
}

function stepInspectContext(delta) {
  const ctx = state.inspectCtx;
  if (!ctx) return;
  const nextPos = ctx.currentIndex + delta;
  if (nextPos >= 0 && nextPos < ctx.items.length) {
    openCompoundAtContextIndex(nextPos);
  }
}

function returnFromCompoundDetail({ fromPopState = false } = {}) {
  const ctx = state.inspectCtx;
  if (!ctx) return;
  const targetTab = ctx.sourceTab;
  const savedY = ctx.savedScrollY;
  const viewKey = ctx.viewKey;
  clearInspectContext(false);
  switchTab(targetTab);
  // Re-render inspector so the row we just viewed gets highlighted clearly
  renderJobInspector(viewKey);
  requestAnimationFrame(() => window.scrollTo({ top: savedY, behavior: "instant" }));
  if (!fromPopState && window.history.state?.omnipanelDetail) {
    try {
      window.history.back();
    } catch {}
  }
}

function clearInspectContext(hideBar = true) {
  state.inspectCtx = null;
  if (hideBar) {
    document.getElementById("detail-context-bar")?.classList.add("hidden");
  } else {
    document.getElementById("detail-context-bar")?.classList.add("hidden");
  }
}

// ==================== RENDER SINGLE PREDICTION / COMPOUND DETAIL ====================
function getMolAttrKey(data, endpoint) {
  const molKey = data?.canonical_smiles || data?.smiles || data?.sequence || data?.query_name || "";
  return `${molKey}::${endpoint}`;
}

function getActiveAttributionSvg(data) {
  if (!data) return null;
  const ep = state.attrEndpoint || data.atom_attribution_endpoint || "PAMPA";
  const cached = state.attrSvgCache.get(getMolAttrKey(data, ep));
  if (cached) return cached;
  if (data.atom_attribution_svg && (!data.atom_attribution_endpoint || data.atom_attribution_endpoint === ep)) {
    return data.atom_attribution_svg;
  }
  return data.atom_attribution_svg || null;
}

function getActiveStructureSvg(data) {
  if (state.svgViewMode === "attribution") {
    const aSvg = getActiveAttributionSvg(data);
    if (aSvg) return aSvg;
  }
  return data?.svg || null;
}

async function fetchAtomAttributionForCurrent(data, endpoint) {
  if (!data) return;
  const targetEp = endpoint || data.atom_attribution_endpoint || "PAMPA";
  state.attrEndpoint = targetEp;
  const ckey = getMolAttrKey(data, targetEp);
  if (state.attrSvgCache.has(ckey)) {
    data.atom_attribution_svg = state.attrSvgCache.get(ckey);
    data.atom_attribution_endpoint = targetEp;
    renderSvgCard(data);
    return;
  }

  state.attrLoading = true;
  renderSvgCard(data);
  try {
    const body = {
      query: !data.smiles && !data.canonical_smiles && !data.sequence ? data.query_name : null,
      smiles: data.canonical_smiles || data.smiles || null,
      sequence: !data.smiles && !data.canonical_smiles ? data.sequence : null,
      endpoint: targetEp,
      width: 760,
      height: 460,
    };
    const res = await apiFetch("/api/v1/atom-attribution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) return;
    const payload = await res.json();
    if (payload?.svg && String(payload.svg).includes("<svg")) {
      state.attrSvgCache.set(ckey, payload.svg);
      data.atom_attribution_svg = payload.svg;
      data.atom_attribution_endpoint = payload.endpoint || targetEp;
      if (Array.isArray(payload.available_endpoints) && payload.available_endpoints.length) {
        data.atom_attribution_endpoints = payload.available_endpoints;
      }
    }
  } catch {
  } finally {
    state.attrLoading = false;
    if (state.lastPrediction === data) renderSvgCard(data);
  }
}

function renderSvgCard(data) {
  // Seed cache whenever backend returns default atom_attribution_svg on /predict
  if (data?.atom_attribution_svg && String(data.atom_attribution_svg).includes("<svg")) {
    const defEp = data.atom_attribution_endpoint || "PAMPA";
    const ckey = getMolAttrKey(data, defEp);
    if (!state.attrSvgCache.has(ckey)) {
      state.attrSvgCache.set(ckey, data.atom_attribution_svg);
    }
    if (!state.attrEndpoint || !((data.atom_attribution_endpoints || []).includes(state.attrEndpoint))) {
      state.attrEndpoint = defEp;
    }
  }
  const availEps =
    Array.isArray(data?.atom_attribution_endpoints) && data.atom_attribution_endpoints.length
      ? data.atom_attribution_endpoints
      : Object.keys(ATTR_ENDPOINT_LABELS);
  const hasAttrReady =
    Boolean(data?.atom_attribution_svg && String(data.atom_attribution_svg).includes("<svg")) ||
    Boolean(data?.canonical_smiles || data?.smiles || data?.sequence);
  if (!hasAttrReady && state.svgViewMode === "attribution") {
    state.svgViewMode = "motifs";
  }

  const attrBtn = document.getElementById("btn-svg-attr");
  if (attrBtn) {
    attrBtn.disabled = !hasAttrReady;
    attrBtn.classList.toggle("has-attr-ready", hasAttrReady);
    attrBtn.title = hasAttrReady ? t(state.lang, "svg_attr_caption") : t(state.lang, "svg_attr_pending");
  }
  document.querySelectorAll("#svg-view-toggle .seg-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.svgMode === state.svgViewMode);
  });

  // Render 8-endpoint attribution selector bar when in attribution mode
  const attrBar = document.getElementById("svg-attr-bar");
  if (attrBar) {
    if (state.svgViewMode === "attribution" && hasAttrReady) {
      attrBar.classList.remove("hidden");
      const activeEp = state.attrEndpoint || data.atom_attribution_endpoint || availEps[0];
      attrBar.innerHTML = availEps
        .map((ep) => {
          const meta = ATTR_ENDPOINT_LABELS[ep] || { zh: ep, en: ep };
          const lbl = state.lang === "zh" ? meta.zh : meta.en;
          return `<button type="button" class="svg-attr-pill ${ep === activeEp ? "active" : ""}" data-attr-ep="${escapeHtml(ep)}">${escapeHtml(lbl)}</button>`;
        })
        .join("");
      attrBar.querySelectorAll(".svg-attr-pill").forEach((pill) => {
        pill.addEventListener("click", () => {
          const nextEp = pill.dataset.attrEp;
          if (nextEp && state.lastPrediction) {
            fetchAtomAttributionForCurrent(state.lastPrediction, nextEp);
          }
        });
      });
    } else {
      attrBar.classList.add("hidden");
      attrBar.innerHTML = "";
    }
  }

  const activeSvg = getActiveStructureSvg(data);
  const svgStage = document.getElementById("mol-svg-stage");
  if (state.svgViewMode === "attribution" && state.attrLoading) {
    svgStage.innerHTML = `<div class="svg-loading-hint">${escapeHtml(t(state.lang, "svg_attr_loading"))}</div>`;
  } else {
    svgStage.innerHTML =
      activeSvg || `<div class="svg-loading-hint">${state.lang === "zh" ? "暂无二维结构" : "Structure unavailable"}</div>`;
  }
  document.getElementById("svg-meta-caption").textContent = `${data.physchem.formula} · ${data.physchem.heavy_atoms} ${state.lang === "zh" ? "重原子" : "heavy atoms"}`;

  const legendEl = document.getElementById("mol-svg-legend");
  if (state.svgViewMode === "attribution" && hasAttrReady) {
    legendEl.innerHTML = `<span class="legend-item" style="color:var(--teal)">${escapeHtml(t(state.lang, "svg_attr_caption"))}</span>`;
  } else if (data.svg_highlights && data.svg_highlights.length) {
    legendEl.innerHTML = data.svg_highlights
      .map(
        (h) => `
        <span class="legend-item" title="${escapeHtml(`${h.label_zh} / ${h.label_en} (${h.atom_count})`)}">
          <span class="legend-swatch" style="background:${escapeHtml(h.color)}"></span>
          ${escapeHtml(state.lang === "zh" ? h.label_zh : h.label_en)}
        </span>`
      )
      .join("");
  } else {
    legendEl.innerHTML = `<span class="legend-item" style="color:var(--ink-faint)">${escapeHtml(t(state.lang, "svg_canonical_note"))}</span>`;
  }
}

function renderPredictionResult(data, { keepContext = false } = {}) {
  if (!keepContext) clearInspectContext(true);
  renderDetailContextBar();

  const hero = document.getElementById("predict-hero");
  hero.classList.remove("hero-centered");
  hero.classList.add("hero-compact");
  setHeroCanvasPaused(true);
  document.getElementById("compact-actions").classList.remove("hidden");
  document.getElementById("landing-overview").classList.add("hidden");
  document.getElementById("predict-results").classList.remove("hidden");

  if (data.atom_attribution_endpoint) {
    state.attrEndpoint = data.atom_attribution_endpoint;
  }
  renderSvgCard(data);

  const ref = data.drug_reference;
  document.getElementById("res-mol-name").textContent = ref
    ? state.lang === "zh"
      ? `${ref.zh_name} · ${data.query_name}`
      : `${data.query_name} · ${ref.zh_name}`
    : data.query_name;
  const trackTag = data.modality_track === "dual_modality" ? "DUAL TRACK" : (data.resolved_type || "SMILES").replace("_", " ").toUpperCase();
  document.getElementById("res-type-pill").textContent = trackTag;
  document.getElementById("res-mol-subtitle").textContent = ref
    ? ref.drug_class
    : data.sequence
      ? state.lang === "zh"
        ? `标准 20-AA 氨基酸序列（${data.sequence.length} 残基）`
        : `Canonical 20-AA Sequence (${data.sequence.length} residues)`
      : state.lang === "zh"
        ? "共价结构 SMILES"
        : "Covalent SMILES Structure";

  document.getElementById("res-mod-badges").innerHTML = (data.modifications || [])
    .map((b) => {
      const lbl = state.lang === "zh" ? b.label_zh || b.label_en : b.label_en;
      return `<span class="mod-badge tone-${escapeHtml(b.tone || "slate")}" title="${escapeHtml(b.detail || b.label_zh || "")}">${escapeHtml(lbl)}</span>`;
    })
    .join("");

  const fdaBox = document.getElementById("res-fda-ref");
  if (ref && (ref.fda_half_life || ref.fda_ppb)) {
    const parts = [];
    if (ref.fda_half_life) parts.push(`<strong>${state.lang === "zh" ? "临床参考 T½" : "Clinical Ref T½"}:</strong> ${escapeHtml(ref.fda_half_life)}`);
    if (ref.fda_ppb) parts.push(`<strong>PPB:</strong> ${escapeHtml(ref.fda_ppb)}`);
    fdaBox.innerHTML = parts.join(" &nbsp;|&nbsp; ");
    fdaBox.classList.remove("hidden");
  } else {
    fdaBox.classList.add("hidden");
  }

  const pc = data.physchem;
  const chargeStr = pc.net_charge_ph74 >= 0 ? `+${pc.net_charge_ph74.toFixed(1)}` : pc.net_charge_ph74.toFixed(1);
  const stripItems = [
    ["MW", `${pc.mw.toFixed(1)} Da`],
    ["cLogP", pc.logp.toFixed(2)],
    ["TPSA", `${pc.tpsa.toFixed(0)} Å²`],
    ["HBD / HBA", `${pc.hbd} / ${pc.hba}`],
    [state.lang === "zh" ? "可旋转键" : "RotBonds", String(pc.rotatable_bonds)],
    [state.lang === "zh" ? "净电荷 (7.4)" : "Charge (7.4)", chargeStr],
    [state.lang === "zh" ? "环数" : "Rings", pc.macrocycle_rings?.length ? `${pc.ring_count} (M${pc.macrocycle_rings[0]})` : String(pc.ring_count)],
    [state.lang === "zh" ? "长度" : "Length", pc.length_aa ? `~${pc.length_aa} AA` : `${pc.heavy_atoms} HA`],
  ];
  document.getElementById("res-physchem-grid").innerHTML = stripItems
    .map(([lbl, val]) => `<div class="pc-cell"><span class="pc-lbl">${lbl}</span><span class="pc-val">${escapeHtml(val)}</span></div>`)
    .join("");

  const rtLabel =
    data.adapter_mode === "task_cached"
      ? state.lang === "zh"
        ? "任务缓存 · 0 ms"
        : "Task Cache · 0 ms"
      : data.cache_hit
        ? `${data.runtime_ms.toFixed(0)} ms · ${state.lang === "zh" ? "缓存" : "cached"}`
        : data.runtime_ms >= 1000
          ? `${(data.runtime_ms / 1000).toFixed(1)} s`
          : `${data.runtime_ms.toFixed(0)} ms`;
  document.getElementById("res-runtime-chip").textContent = rtLabel;

  renderAdmetRadar(
    document.getElementById("res-radar-stage"),
    data.radar || [],
    document.getElementById("res-radar-readout"),
    state.lang
  );

  // Full-width warning banner whenever offline heuristic fallback is active (N5)
  let offWarn = document.getElementById("offline-fallback-banner");
  const resultsHost = document.getElementById("predict-results");
  if (!offWarn && resultsHost) {
    offWarn = document.createElement("div");
    offWarn.id = "offline-fallback-banner";
    offWarn.className = "mechanism-banner hidden";
    offWarn.style.cssText =
      "border-color:var(--amber);background:rgba(245,158,11,0.12);color:var(--ink);padding:10px 14px;margin-bottom:12px;font-size:12.5px;font-weight:600;";
    resultsHost.insertBefore(offWarn, resultsHost.firstChild);
  }
  if (offWarn) {
    if (data.adapter_mode === "v33_offline_fallback") {
      offWarn.textContent = "⚠ " + t(state.lang, "offline_banner");
      offWarn.classList.remove("hidden");
    } else {
      offWarn.classList.add("hidden");
    }
  }

  // Long-acting fatty-acylation mechanism banner
  const mechEl = document.getElementById("mechanism-banner");
  const anch = data.mechanism_anchor;
  if (anch && anch.detected) {
    const rngStr = anch.range_h ? `${anch.range_h[0]}–${anch.range_h[1]} h` : `${anch.value_h} h`;
    const anchorsText = (anch.anchor_drugs || []).join(" · ");
    const routeTag = anch.ml_baseline_route ? ` (${anch.ml_baseline_route})` : "";
    const mechTitle = state.lang === "zh" ? `${anch.label_zh || anch.label_en} · ${anch.label_en}` : `${anch.label_en} / ${anch.label_zh || ""}`;
    mechEl.innerHTML = `
      <div class="mech-left">
        <div class="mech-kicker">${escapeHtml(t(state.lang, "mech_kicker"))}</div>
        <h3 class="mech-title">${escapeHtml(mechTitle)}</h3>
        <p class="mech-note">${escapeHtml(anch.note || "")}<br/><strong>${escapeHtml(t(state.lang, "mech_anchors_lbl"))}</strong> ${escapeHtml(anchorsText)}</p>
      </div>
      <div class="mech-right">
        <div class="mech-metric">
          <span class="mech-metric-val">${escapeHtml(rngStr)}</span>
          <span class="mech-metric-sub">${escapeHtml(t(state.lang, "mech_window_sub", { v: anch.value_h }))}</span>
        </div>
        <div class="mech-ml-pill" title="${escapeHtml(anch.ml_baseline_note || "")}">
          <span>${escapeHtml(t(state.lang, "mech_ml_lbl"))}${escapeHtml(routeTag)}</span>
          <strong>${anch.ml_baseline_h != null ? anch.ml_baseline_h.toFixed(1) + " h" : "—"}</strong>
        </div>
      </div>`;
    mechEl.classList.remove("hidden");
  } else {
    mechEl.classList.add("hidden");
  }

  renderEndpointDashboards(data);
}

function renderEndpointDashboards(data) {
  const allPreds = data.predictions || [];
  const filtered = allPreds.filter((p) => {
    if (state.adFilter !== "all" && p.ad_status !== state.adFilter) return false;
    if (state.endpointSearch) {
      const hay = `${p.endpoint} ${p.display_en} ${p.display_zh} ${p.domain} ${p.model_family}`.toLowerCase();
      if (!hay.includes(state.endpointSearch)) return false;
    }
    return true;
  });

  const tier1 = filtered.filter((p) => p.tier === 1);
  const tier2 = filtered.filter((p) => p.tier === 2);
  document.getElementById("count-tier-1").textContent = String(allPreds.filter((p) => p.tier === 1).length);
  document.getElementById("count-tier-2").textContent = String(allPreds.filter((p) => p.tier === 2).length);

  renderDomainGrid(document.getElementById("tier1-dashboard"), ["pk", "permeability", "safety", "physicochemical"], tier1);
  renderTier2Dashboard(document.getElementById("tier2-dashboard"), tier2, allPreds);
}

function renderDomainGrid(container, domainKeys, preds) {
  container.innerHTML = "";
  domainKeys.forEach((dKey) => {
    const meta = DOMAIN_META[dKey];
    const items = preds.filter((p) => p.domain === dKey);
    const card = document.createElement("div");
    card.className = "domain-card";
    const titleMain = state.lang === "zh" ? meta.zh : meta.en;
    const titleSub = state.lang === "zh" ? meta.en : meta.zh;
    const rowsHtml = items.length
      ? items.map((p) => buildEndpointItemHtml(p)).join("")
      : `<div class="empty-inline">${escapeHtml(t(state.lang, "no_ep_match"))}</div>`;

    card.innerHTML = `
      <div class="domain-header">
        <div class="domain-title-wrap">
          <span class="domain-dot ${meta.dot}"></span>
          <h3 class="domain-title">${escapeHtml(titleMain)} <span class="domain-zh">${escapeHtml(titleSub)}</span></h3>
        </div>
        <span class="domain-count">${escapeHtml(t(state.lang, "ep_count_unit", { n: items.length }))}</span>
      </div>
      <div class="ep-list">${rowsHtml}</div>`;

    wireExpandableDrawers(card);
    container.appendChild(card);
  });
}

function _tier2Subgroup(ep) {
  if (ep.startsWith("MIC_")) return "mic";
  if (ep.startsWith("MBC_") || ep.startsWith("LC_")) return "mbc";
  return "biofilm";
}

function renderTier2Dashboard(container, tier2Preds, allPreds) {
  container.innerHTML = "";
  const nMic = tier2Preds.filter((p) => _tier2Subgroup(p.endpoint) === "mic").length;
  const nMbc = tier2Preds.filter((p) => _tier2Subgroup(p.endpoint) === "mbc").length;
  const nBio = tier2Preds.filter((p) => _tier2Subgroup(p.endpoint) === "biofilm").length;
  const hasMultipleGroups = [nMic > 0, nMbc > 0, nBio > 0].filter(Boolean).length > 1;
  const visibleT2 =
    hasMultipleGroups && state.tier2Group !== "all"
      ? tier2Preds.filter((p) => _tier2Subgroup(p.endpoint) === state.tier2Group)
      : tier2Preds;

  const meta = DOMAIN_META.antimicrobial;
  const titleMain = state.lang === "zh" ? meta.zh : meta.en;
  const titleSub = state.lang === "zh" ? meta.en : meta.zh;

  const micCard = document.createElement("div");
  micCard.className = "domain-card";
  const subFilterHtml = hasMultipleGroups
    ? `<div class="tier2-subfilters">
         <button type="button" class="filter-chip ${state.tier2Group === "all" ? "active" : ""}" data-t2g="all">${escapeHtml(t(state.lang, "t2_all", { n: tier2Preds.length }))}</button>
         ${nMic ? `<button type="button" class="filter-chip ${state.tier2Group === "mic" ? "active" : ""}" data-t2g="mic">${escapeHtml(t(state.lang, "t2_mic", { n: nMic }))}</button>` : ""}
         ${nMbc ? `<button type="button" class="filter-chip ${state.tier2Group === "mbc" ? "active" : ""}" data-t2g="mbc">${escapeHtml(t(state.lang, "t2_mbc", { n: nMbc }))}</button>` : ""}
         ${nBio ? `<button type="button" class="filter-chip ${state.tier2Group === "biofilm" ? "active" : ""}" data-t2g="biofilm">${escapeHtml(t(state.lang, "t2_biofilm", { n: nBio }))}</button>` : ""}
       </div>`
    : "";
  const rowsHtml = visibleT2.length
    ? visibleT2.map((p) => buildEndpointItemHtml(p)).join("")
    : `<div class="empty-inline">${escapeHtml(t(state.lang, "no_ep_match"))}</div>`;

  micCard.innerHTML = `
    <div class="domain-header">
      <div class="domain-title-wrap">
        <span class="domain-dot ${meta.dot}"></span>
        <h3 class="domain-title">${escapeHtml(titleMain)} <span class="domain-zh">${escapeHtml(titleSub)}</span></h3>
      </div>
      <span class="domain-count">${escapeHtml(t(state.lang, "ep_count_unit", { n: visibleT2.length }))}</span>
    </div>
    ${subFilterHtml}
    <div class="ep-list">${rowsHtml}</div>`;
  micCard.querySelectorAll("[data-t2g]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.tier2Group = btn.dataset.t2g;
      renderTier2Dashboard(container, tier2Preds, allPreds);
    });
  });
  wireExpandableDrawers(micCard);
  container.appendChild(micCard);

  const hc50Ep = allPreds.find((p) => p.endpoint === "hemolysis_HC50");
  const hemClsEp = allPreds.find((p) => p.endpoint === "hemolysis_cls");
  const validMics = allPreds.filter((p) => p.tier === 2 && p.endpoint.startsWith("MIC_") && p.value != null && p.value > 0);
  const bestMic = validMics.length ? validMics.reduce((a, b) => (a.value < b.value ? a : b)) : null;
  const tiVal = hc50Ep && hc50Ep.value && bestMic ? hc50Ep.value / bestMic.value : null;

  const guideCard = document.createElement("div");
  guideCard.className = "domain-card tier2-guide-card";
  guideCard.innerHTML = `
    <div class="domain-header">
      <div class="domain-title-wrap">
        <span class="domain-dot dot-safety"></span>
        <h3 class="domain-title">${escapeHtml(t(state.lang, "t2_guide_title"))}</h3>
      </div>
      <span class="domain-count">${escapeHtml(t(state.lang, "t2_guide_badge"))}</span>
    </div>
    <div class="tier2-guide-body">
      <div class="ti-kpi-row">
        <div class="ti-kpi">
          <span class="dm-lbl">${escapeHtml(t(state.lang, "t2_best_mic"))}</span>
          <span class="ti-val">${bestMic ? `${escapeHtml(bestMic.value_display)} µM` : "—"}</span>
          <span class="ti-sub">${bestMic ? escapeHtml(state.lang === "zh" ? bestMic.display_zh : bestMic.display_en) : "—"}</span>
        </div>
        <div class="ti-kpi">
          <span class="dm-lbl">${escapeHtml(t(state.lang, "t2_hc50_margin"))}</span>
          <span class="ti-val">${hc50Ep ? `${escapeHtml(hc50Ep.value_display)} µM` : "—"}</span>
          <span class="ti-sub">${state.lang === "zh" ? "溶血风险" : "Risk"}: ${hemClsEp ? escapeHtml(hemClsEp.value_display) : "—"}</span>
        </div>
        <div class="ti-kpi">
          <span class="dm-lbl">${escapeHtml(t(state.lang, "t2_ti_label"))}</span>
          <span class="ti-val" style="color:${tiVal != null && tiVal >= 10 ? "var(--emerald)" : "var(--amber)"}">${tiVal != null ? `${tiVal >= 100 ? tiVal.toFixed(0) : tiVal.toFixed(1)}×` : "—"}</span>
          <span class="ti-sub">${tiVal != null ? escapeHtml(t(state.lang, tiVal >= 10 ? "t2_ti_good" : "t2_ti_narrow")) : "N/A"}</span>
        </div>
      </div>
      <p class="drawer-note" style="margin-top:12px;">${escapeHtml(t(state.lang, "t2_guide_note"))}</p>
    </div>`;
  container.appendChild(guideCard);
}

function wireExpandableDrawers(card) {
  card.querySelectorAll(".ep-row").forEach((row) => {
    const toggle = () => {
      const parent = row.closest(".ep-item");
      const drawer = parent.querySelector(".ep-drawer");
      const expanded = parent.classList.toggle("expanded");
      row.setAttribute("aria-expanded", String(expanded));
      drawer.classList.toggle("hidden", !expanded);
    };
    row.addEventListener("click", toggle);
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });
  });
}

function buildEndpointItemHtml(p) {
  const adTxt = trAd(state.lang, p.ad_status);
  const tcTxt = p.max_tanimoto !== null && p.max_tanimoto !== undefined ? ` · ${p.max_tanimoto.toFixed(2)}` : "";
  const unitClean = p.unit === "probability" || p.unit === "fraction_bound" || p.unit === "fraction" ? "" : p.unit;
  const epPrimary = state.lang === "zh" ? p.display_zh : p.display_en;
  const epSecondary = state.lang === "zh" ? p.display_en : p.display_zh;

  const analogsHtml = (p.nearest_analogs || [])
    .map((a) => {
      const cBadge = a.censor_type && a.censor_type !== "exact" ? `<span class="censor-mini">${escapeHtml(a.censor_type)}</span>` : "";
      return `
      <div class="analog-card">
        <div class="analog-top">
          <span class="analog-name mono" title="${escapeHtml(a.sequence || a.compound_id)}">${escapeHtml(a.compound_id)}</span>
          <span class="analog-tc">Tc ${a.tanimoto.toFixed(2)}</span>
        </div>
        <div class="analog-obs">${escapeHtml(a.observed_display)} ${escapeHtml(a.unit === "probability" || a.unit === "fraction_bound" ? "" : a.unit)} ${cBadge}</div>
        <span class="analog-src" title="${escapeHtml(a.sequence || a.name)}">${escapeHtml(a.name !== a.compound_id ? a.name : a.source)}</span>
      </div>`;
    })
    .join("");

  const cvDisplay = p.cv_score != null ? `${p.cv_metric_name} ${p.cv_score.toFixed(3)}` : "—";
  const holdDisplay = p.holdout_score != null ? `${p.holdout_score.toFixed(3)}` : "—";
  const calOrCovLabel = t(state.lang, p.kind === "classification" ? "dm_brier_ece" : "dm_cov80");
  const uqLowPill =
    p.kind !== "classification" && p.uq_reliability === "low"
      ? ` <span class="mod-badge tone-amber" style="font-size:10.5px;padding:1px 6px;" title="${escapeHtml(t(state.lang, "uq_low_tip"))}">⚠ ${escapeHtml(t(state.lang, "uq_low_badge"))}</span>`
      : "";
  const calOrCovVal =
    p.kind === "classification"
      ? p.brier_holdout != null && p.ece_holdout != null
        ? `${p.brier_holdout.toFixed(3)} / ${p.ece_holdout.toFixed(3)}`
        : "Platt / Iso"
      : p.conformal_coverage_80 != null
        ? `${(p.conformal_coverage_80 * 100).toFixed(1)}%${uqLowPill}`
        : "—";
  const trainDisplay = p.n_train
    ? `${p.n_train.toLocaleString()} CV / ${(p.n_holdout || 0).toLocaleString()} HO${p.n_censored ? ` · ${p.n_censored} cens.` : ""}`
    : "—";
  const censoredHtml = p.censored_diagnostic_text
    ? `<div class="censored-diag-box">
         <strong>${escapeHtml(t(state.lang, "dm_censored"))}</strong> <span class="mono">${escapeHtml(p.censored_diagnostic_text)}</span>
         <span class="censored-diag-note">— ${escapeHtml(p.censored_note || "")}</span>
       </div>`
    : "";
  const extraCiHtml = p.ci90_display ? ` · <strong>${escapeHtml(t(state.lang, "dm_ci90"))}</strong> <span class="mono">${escapeHtml(p.ci90_display)}</span>` : "";

  return `
    <div class="ep-item" data-endpoint="${escapeHtml(p.endpoint)}">
      <div class="ep-row" role="button" tabindex="0" aria-expanded="false">
        <div class="ep-name-col">
          <span class="ep-en">${escapeHtml(epPrimary)}</span>
          <span class="ep-zh">${escapeHtml(epSecondary)}</span>
        </div>
        <div class="ep-val-col">
          <div class="ep-val-main">${escapeHtml(p.value_display)}<span class="ep-unit">${escapeHtml(unitClean)}</span></div>
          <span class="ep-risk risk-${escapeHtml(p.risk_level)}">
            <span class="risk-dot"></span>${escapeHtml(trRisk(state.lang, p.risk_label))}
          </span>
        </div>
        <div class="ep-ci-col">${buildIntervalBarHtml(p, state.lang)}</div>
        <span class="ad-pill ad-${escapeHtml(p.ad_status)}">${escapeHtml(adTxt)}${tcTxt}</span>
        <span class="ep-chevron">▾</span>
      </div>
      <div class="ep-drawer hidden">
        <div class="drawer-meta-grid">
          <div class="dm-box"><span class="dm-lbl">${escapeHtml(t(state.lang, "dm_champion"))}</span><span class="dm-val">${escapeHtml(p.model_family)}</span></div>
          <div class="dm-box"><span class="dm-lbl">${escapeHtml(t(state.lang, "dm_cv_ho"))}</span><span class="dm-val">${cvDisplay} / ${holdDisplay}</span></div>
          <div class="dm-box"><span class="dm-lbl">${escapeHtml(calOrCovLabel)}</span><span class="dm-val">${calOrCovVal}</span></div>
          <div class="dm-box"><span class="dm-lbl">${escapeHtml(t(state.lang, "dm_corpus"))}</span><span class="dm-val">${trainDisplay}</span></div>
        </div>
        ${censoredHtml}
        <p class="drawer-note">${escapeHtml(p.note || "")}${extraCiHtml}${p.assay_context ? ` <strong>[${escapeHtml(p.assay_context)}]</strong>` : ""}</p>
        <div class="analogs-title">${escapeHtml(t(state.lang, "dm_analogs"))}</div>
        <div class="analog-list">${analogsHtml || `<span style="font-size:12px;color:var(--ink-faint)">${escapeHtml(t(state.lang, "dm_no_analogs"))}</span>`}</div>
      </div>
    </div>`;
}

// ==================== SINGLE EXPORT & SVG MODAL ====================
function exportSingleCsv() {
  const d = state.lastPrediction;
  if (!d) return;
  const cols = [
    "query", "endpoint", "display_en", "display_zh", "tier", "domain", "prediction_source", "value", "value_display", "unit",
    "ci80_low", "ci80_high", "conformal_coverage_80", "uq_reliability", "ad_status", "max_tanimoto", "risk_label", "winning_family", "cv_score", "holdout_score",
  ];
  const lines = [cols.join(",")];
  (d.predictions || []).forEach((p) => {
    const row = [
      d.query_name, p.endpoint, p.display_en, p.display_zh, p.tier, p.domain, p.prediction_source ?? d.adapter_mode ?? "",
      p.value ?? "", p.value_display, p.unit, p.ci80_low ?? "", p.ci80_high ?? "",
      p.conformal_coverage_80 ?? "", p.uq_reliability ?? "",
      p.ad_status, p.max_tanimoto ?? "", p.risk_label, p.winning_family ?? "",
      p.cv_score ?? "", p.holdout_score ?? "",
    ].map((x) => `"${String(x).replace(/"/g, '""')}"`);
    lines.push(row.join(","));
  });
  downloadBlob("\uFEFF" + lines.join("\n"), `${slugify(d.query_name)}_omnipanel.csv`, "text/csv;charset=utf-8");
}

function exportSingleJson() {
  const d = state.lastPrediction;
  if (!d) return;
  const copy = { ...d };
  delete copy.svg;
  downloadBlob(JSON.stringify(copy, null, 2), `${slugify(d.query_name)}_omnipanel.json`, "application/json");
}

function wireSvgModal() {
  const modal = document.getElementById("svg-modal");
  const openModal = () => {
    const d = state.lastPrediction;
    const svg = getActiveStructureSvg(d);
    if (!d || !svg) return;
    document.getElementById("modal-mol-title").textContent = `${d.query_name} — ${t(state.lang, "modal_title")}`;
    document.getElementById("modal-mol-sub").textContent = `${d.physchem.formula} · MW ${d.physchem.mw.toFixed(1)} Da · ${d.physchem.heavy_atoms} HA`;
    document.getElementById("modal-svg-stage").innerHTML = svg;
    document.getElementById("modal-svg-legend").innerHTML = document.getElementById("mol-svg-legend").innerHTML;
    modal.classList.remove("hidden");
  };
  const closeModal = () => modal.classList.add("hidden");
  document.getElementById("btn-zoom-svg")?.addEventListener("click", openModal);
  document.getElementById("mol-svg-stage")?.addEventListener("click", openModal);
  document.getElementById("modal-close")?.addEventListener("click", closeModal);
  modal?.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });
  const dlSvg = () => {
    const d = state.lastPrediction;
    const svg = getActiveStructureSvg(d);
    if (!d || !svg) return;
    downloadBlob(svg, `${slugify(d.query_name)}_2d_structure.svg`, "image/svg+xml;charset=utf-8");
  };
  document.getElementById("btn-download-svg")?.addEventListener("click", dlSvg);
  document.getElementById("modal-download-svg")?.addEventListener("click", dlSvg);
}

// ==================== BATCH SCREENING (SUBMITS ASYNC TASK!) ====================
function wireBatchWorkspace() {
  const dropzone = document.getElementById("batch-dropzone");
  const fileInput = document.getElementById("batch-file-input");
  const textarea = document.getElementById("batch-paste-area");
  const titleInput = document.getElementById("batch-task-title");

  document.getElementById("btn-browse-file")?.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  dropzone?.addEventListener("click", () => fileInput.click());
  dropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone?.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) await handleBatchFile(file);
  });
  fileInput?.addEventListener("change", async () => {
    if (fileInput.files[0]) await handleBatchFile(fileInput.files[0]);
  });

  let parseTimer = null;
  textarea?.addEventListener("input", () => {
    clearTimeout(parseTimer);
    parseTimer = setTimeout(() => previewBatchText(textarea.value), 300);
  });

  const loadBatchPreset = (key, fallbackCsv, defaultTitleZh, defaultTitleEn) => {
    const preset = state.batchPresets?.[key];
    textarea.value = preset?.csv || fallbackCsv;
    titleInput.value = state.lang === "zh" ? preset?.label_zh || defaultTitleZh : preset?.label || defaultTitleEn;
    document.getElementById("batch-file-label").textContent =
      state.lang === "zh" ? preset?.description_zh || defaultTitleZh : preset?.description || defaultTitleEn;
    previewBatchText(textarea.value, `${key}.csv`);
  };

  document.getElementById("btn-batch-demo")?.addEventListener("click", () => {
    loadBatchPreset(
      "demo_7",
      state.batchDemoCsv || "name,query\nSemaglutide,Semaglutide\nTirzepatide,Tirzepatide\nCyclosporine A,Cyclosporine A\nCaspofungin,Caspofungin\nDaptomycin,Daptomycin\nClassic AMP,HISEAVKIFGCKWKENWKWNSNC\nLiraglutide,Liraglutide",
      "综合示例 (7)",
      "Mixed Benchmark (7)"
    );
  });
  document.getElementById("btn-batch-case17")?.addEventListener("click", () => {
    loadBatchPreset(
      "case17_17",
      "name,query\nSemaglutide,Semaglutide\nTirzepatide,Tirzepatide\nLiraglutide,Liraglutide\nExenatide,Exenatide\nLeuprolide,Leuprolide\nBivalirudin,Bivalirudin\nGlucagon,Glucagon\nTeriparatide,Teriparatide\nCyclosporine A,Cyclosporine A\nOxytocin,Oxytocin\nOctreotide,Octreotide\nLinaclotide,Linaclotide\nZiconotide,Ziconotide\nDaptomycin,Daptomycin\nCaspofungin,Caspofungin\nInsulin detemir,Insulin detemir\nInsulin degludec,Insulin degludec",
      "17 个案例",
      "17 Clinical Cases"
    );
  });
  document.getElementById("btn-batch-echino")?.addEventListener("click", () => {
    loadBatchPreset(
      "echinocandin_6",
      "name,query\nCaspofungin,Caspofungin\nMicafungin,Micafungin\nAnidulafungin,Anidulafungin\nRezafungin,Rezafungin\nCyclosporine A,Cyclosporine A\nDaptomycin,Daptomycin",
      "棘白菌素与大环肽 (6)",
      "Echinocandins & Macrocycles (6)"
    );
  });

  document.getElementById("btn-batch-clear")?.addEventListener("click", () => {
    textarea.value = "";
    titleInput.value = "";
    fileInput.value = "";
    state.batchParsed = null;
    document.getElementById("batch-file-label").textContent = t(state.lang, "drop_hint");
    document.getElementById("batch-preview-meta").textContent = t(state.lang, "batch_preview_wait");
    document.getElementById("batch-preview-badges").innerHTML = "";
    document.getElementById("batch-preview-tbody").innerHTML = `<tr><td colspan="5" class="empty-cell">${escapeHtml(t(state.lang, "batch_preview_empty"))}</td></tr>`;
    document.getElementById("batch-error").classList.add("hidden");
  });

  document.getElementById("btn-run-batch")?.addEventListener("click", submitAsyncBatchTask);
  document.getElementById("btn-goto-tasks-from-batch")?.addEventListener("click", () => {
    if (state.batchJobId) selectTaskInCenter(state.batchJobId);
    switchTab("tasks");
  });
}

async function handleBatchFile(file) {
  const text = await file.text();
  document.getElementById("batch-paste-area").value = text;
  document.getElementById("batch-task-title").value = file.name.replace(/\.[^.]+$/, "");
  document.getElementById("batch-file-label").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  await previewBatchText(text, file.name);
}

async function previewBatchText(rawText, filename = null) {
  const tbody = document.getElementById("batch-preview-tbody");
  const metaEl = document.getElementById("batch-preview-meta");
  const badgesEl = document.getElementById("batch-preview-badges");

  if (!rawText.trim()) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-cell">${escapeHtml(t(state.lang, "batch_preview_empty"))}</td></tr>`;
    metaEl.textContent = t(state.lang, "batch_preview_wait");
    badgesEl.innerHTML = "";
    return;
  }

  try {
    const res = await apiFetch("/api/v1/batch/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_text: rawText, filename }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Parse error");
    state.batchParsed = data;

    metaEl.textContent =
      state.lang === "zh" ? `已解析 ${data.total} 个分子（单次上限 ${data.max_batch}）` : `Parsed ${data.total} molecules (max ${data.max_batch})`;
    badgesEl.innerHTML = `
      <span class="ad-pill ad-in_domain">${data.valid_count} ${state.lang === "zh" ? "有效" : "Valid"}</span>
      ${data.invalid_count ? `<span class="ad-pill ad-ood">${data.invalid_count} ${state.lang === "zh" ? "无效" : "Invalid"}</span>` : ""}`;

    tbody.innerHTML = data.rows
      .slice(0, 60)
      .map((r) => {
        const structStr = r.sequence || r.smiles || "—";
        const preview = structStr.length > 34 ? structStr.slice(0, 34) + "…" : structStr;
        return `
          <tr>
            <td class="mono">${r.index}</td>
            <td><strong>${escapeHtml(r.name)}</strong></td>
            <td><span class="type-pill">${escapeHtml(r.detected_type)}</span></td>
            <td class="mono" style="font-size:11.2px;">${escapeHtml(preview)}</td>
            <td>${r.valid ? `<span class="ad-pill ad-in_domain">OK</span>` : `<span class="ad-pill ad-ood" title="${escapeHtml(r.error || "")}">Err</span>`}</td>
          </tr>`;
      })
      .join("");
  } catch (err) {
    metaEl.textContent = err.message;
  }
}

async function submitAsyncBatchTask() {
  const rawText = document.getElementById("batch-paste-area").value.trim();
  const title = document.getElementById("batch-task-title").value.trim();
  const errEl = document.getElementById("batch-error");
  errEl.classList.add("hidden");

  if (!rawText) {
    errEl.textContent =
      state.lang === "zh"
        ? "请先上传文件、粘贴分子列表，或点击上方“17 个案例 (17)”等预设按钮。"
        : 'Please upload a file, paste molecules, or click "17 Clinical Cases (17)" above.';
    errEl.classList.remove("hidden");
    return;
  }

  const btn = document.getElementById("btn-run-batch");
  btn.disabled = true;
  btn.textContent = t(state.lang, "btn_submitting");

  try {
    const res = await apiFetch("/api/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: title || null,
        raw_text: rawText,
        include_tier2: document.getElementById("batch-tier2").checked,
        conditions: {
          rbc_species: document.getElementById("batch-rbc").value,
          pk_context: "human_plasma",
        },
      }),
    });
    const job = await res.json();
    if (!res.ok) throw new Error(job.detail || "Failed to submit task.");

    state.batchJobId = job.job_id;
    state.selectedTaskId = job.job_id;
    state.inspectorUi.batch = makeInspectorUiState();
    state.inspectorUi.tasks = makeInspectorUiState();

    updateBatchActiveBanner(job);
    await refreshJobsList({ autoSelectFirst: false });
    await fetchAndRenderJobDetail(job.job_id, ["batch", "tasks"]);
    ensurePollingActive();
  } catch (err) {
    errEl.textContent = err.message || String(err);
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = t(state.lang, "btn_submit_batch");
  }
}

function updateBatchActiveBanner(job) {
  const banner = document.getElementById("batch-active-banner");
  if (!banner || !job) return;
  banner.classList.remove("hidden");
  document.getElementById("bab-title").textContent = `${job.title} (${job.job_id}) · ${statusLabel(job.status)}`;
  document.getElementById("batch-progress-fill").style.width = `${job.progress_pct || 0}%`;
  if (job.status === "queued" || job.status === "running") {
    const etaStr = job.eta_seconds != null ? t(state.lang, "task_eta", { s: Math.ceil(job.eta_seconds) }) : "";
    document.getElementById("batch-progress-label").textContent = t(state.lang, "task_progress_running", {
      item: job.current_item || "…",
      done: job.completed,
      tot: job.total,
      pct: job.progress_pct,
      eta: etaStr,
    });
  } else {
    const rt = job.runtime_ms >= 1000 ? `${(job.runtime_ms / 1000).toFixed(1)}s` : `${Math.round(job.runtime_ms || 0)}ms`;
    document.getElementById("batch-progress-label").textContent =
      state.lang === "zh"
        ? `${statusLabel(job.status)}：成功 ${job.succeeded}/${job.total} · 耗时 ${rt}`
        : `${statusLabel(job.status)}: ${job.succeeded}/${job.total} succeeded in ${rt}`;
  }
}

// ==================== DEDICATED TASK CENTER ====================
function wireTasksWorkspace() {
  document.getElementById("btn-tasks-new-batch")?.addEventListener("click", () => switchTab("batch"));
  document.getElementById("btn-tasks-refresh")?.addEventListener("click", () => refreshJobsList({ autoSelectFirst: false }));

  document.querySelectorAll("#tasks-status-filter .seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      state.tasksStatusFilter = b.dataset.st;
      document.querySelectorAll("#tasks-status-filter .seg-btn").forEach((x) => x.classList.toggle("active", x.dataset.st === state.tasksStatusFilter));
      renderTasksQueue();
    });
  });

  document.getElementById("tasks-type-filter")?.addEventListener("change", (e) => {
    state.tasksTypeFilter = e.target.value;
    renderTasksQueue();
  });

  document.getElementById("tasks-search")?.addEventListener("input", (e) => {
    state.tasksSearch = e.target.value.trim().toLowerCase();
    renderTasksQueue();
  });
}

function statusLabel(st) {
  return t(state.lang, `task_status_${st}`) || st;
}

function statusTone(st) {
  if (st === "completed") return "st-completed";
  if (st === "running" || st === "queued") return "st-running";
  if (st === "partial") return "st-partial";
  return "st-failed";
}

function formatTimeShort(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString(state.lang === "zh" ? "zh-CN" : "en-US", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

async function refreshJobsList({ autoSelectFirst = false } = {}) {
  try {
    const res = await apiFetch("/api/v1/jobs?limit=80");
    if (!res.ok) return;
    const data = await res.json();
    state.jobs = data.jobs || [];
    state.activeJobCount = data.active_count || 0;

    // Update topbar Tasks nav badge
    const navBadge = document.getElementById("nav-tasks-badge");
    if (navBadge) {
      if (state.activeJobCount > 0) {
        navBadge.textContent = String(state.activeJobCount);
        navBadge.classList.remove("hidden");
        navBadge.classList.add("pulse");
      } else if (state.jobs.length > 0) {
        navBadge.textContent = String(state.jobs.length);
        navBadge.classList.remove("hidden", "pulse");
      } else {
        navBadge.classList.add("hidden");
      }
    }

    renderTasksQueue();

    if (autoSelectFirst && !state.selectedTaskId && state.jobs.length > 0) {
      await selectTaskInCenter(state.jobs[0].job_id);
    }
    if (state.activeJobCount > 0) {
      ensurePollingActive();
    } else {
      stopPolling();
    }
  } catch (err) {
    console.error("Failed to fetch jobs:", err);
  }
}

function ensurePollingActive() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(async () => {
    await refreshJobsList({ autoSelectFirst: false });
    const idsToRefresh = new Set();
    if (state.batchJobId) {
      const bj = state.jobs.find((j) => j.job_id === state.batchJobId);
      if (bj) {
        updateBatchActiveBanner(bj);
        if (bj.status === "running" || bj.status === "queued" || state.batchJobDetail?.status !== bj.status || state.batchJobDetail?.completed !== bj.completed) {
          idsToRefresh.add(state.batchJobId);
        }
      }
    }
    if (state.selectedTaskId) {
      const tj = state.jobs.find((j) => j.job_id === state.selectedTaskId);
      if (tj && (tj.status === "running" || tj.status === "queued" || state.selectedTaskDetail?.status !== tj.status || state.selectedTaskDetail?.completed !== tj.completed)) {
        idsToRefresh.add(state.selectedTaskId);
      }
    }
    for (const jid of idsToRefresh) {
      const targets = [];
      if (jid === state.batchJobId) targets.push("batch");
      if (jid === state.selectedTaskId) targets.push("tasks");
      if (targets.length) await fetchAndRenderJobDetail(jid, targets);
    }
    if (state.activeJobCount === 0) stopPolling();
  }, 1600);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function selectTaskInCenter(jobId) {
  state.selectedTaskId = jobId;
  state.inspectorUi.tasks = makeInspectorUiState();
  renderTasksQueue();
  await fetchAndRenderJobDetail(jobId, ["tasks"]);
}

async function fetchAndRenderJobDetail(jobId, targetViews = ["tasks"]) {
  try {
    const res = await apiFetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
    if (!res.ok) return;
    const detail = await res.json();

    targetViews.forEach((vk) => {
      if (vk === "batch") {
        state.batchJobDetail = detail;
        updateBatchActiveBanner(detail);
      } else {
        state.selectedTaskDetail = detail;
      }
      const ui = state.inspectorUi[vk];
      if (ui.selectedIds.size === 0 && (detail.results || []).length > 0) {
        detail.results
          .filter((r) => r.status === "ok")
          .slice(0, 3)
          .forEach((r) => ui.selectedIds.add(r.index));
      }
      renderJobInspector(vk);
    });
  } catch (err) {
    console.error("Failed to fetch job detail:", err);
  }
}

function renderTasksQueue() {
  const runningN = state.jobs.filter((j) => j.status === "running" || j.status === "queued").length;
  const doneN = state.jobs.filter((j) => j.status === "completed" || j.status === "partial").length;
  const elR = document.getElementById("ts-running");
  const elC = document.getElementById("ts-completed");
  const elT = document.getElementById("ts-total");
  if (elR) elR.textContent = String(runningN);
  if (elC) elC.textContent = String(doneN);
  if (elT) elT.textContent = String(state.jobs.length);

  const listEl = document.getElementById("tasks-queue-list");
  if (!listEl) return;

  const filtered = state.jobs.filter((j) => {
    if (state.tasksStatusFilter === "running" && j.status !== "running" && j.status !== "queued") return false;
    if (state.tasksStatusFilter === "completed" && j.status !== "completed" && j.status !== "partial") return false;
    if (state.tasksStatusFilter === "failed" && j.status !== "failed" && j.status !== "cancelled") return false;
    if (state.tasksTypeFilter !== "all" && j.job_type !== state.tasksTypeFilter) return false;
    if (state.tasksSearch) {
      const hay = `${j.title} ${j.job_id}`.toLowerCase();
      if (!hay.includes(state.tasksSearch)) return false;
    }
    return true;
  });

  if (!filtered.length) {
    listEl.innerHTML = `<div class="tasks-queue-empty">${escapeHtml(t(state.lang, "tasks_empty_title"))}</div>`;
    return;
  }

  listEl.innerHTML = filtered
    .map((j) => {
      const activeCls = j.job_id === state.selectedTaskId ? "active" : "";
      const isLive = j.status === "running" || j.status === "queued";
      const typeLbl = t(state.lang, j.job_type === "batch" ? "task_type_batch" : "task_type_single");
      const rtStr = j.runtime_ms >= 1000 ? `${(j.runtime_ms / 1000).toFixed(1)}s` : `${Math.round(j.runtime_ms || 0)}ms`;
      return `
        <div class="task-card ${activeCls}" data-job-id="${escapeHtml(j.job_id)}">
          <div class="tc-top">
            <span class="tc-title" title="${escapeHtml(j.title)}">${escapeHtml(j.title)}</span>
            <span class="task-st-pill ${statusTone(j.status)}">${isLive ? `<span class="mini-spin"></span>` : ""}${escapeHtml(statusLabel(j.status))}</span>
          </div>
          <div class="tc-meta">
            <span class="type-pill">${escapeHtml(typeLbl)} · ${j.total}</span>
            <span class="mono tc-id">${escapeHtml(j.job_id)}</span>
            <span class="tc-time">${escapeHtml(formatTimeShort(j.created_at))}</span>
          </div>
          ${
            isLive
              ? `<div class="tc-prog">
                   <div class="progress-bar-track"><div class="progress-bar-fill" style="width:${j.progress_pct}%"></div></div>
                   <span class="mono">${j.completed}/${j.total} (${j.progress_pct}%)</span>
                 </div>`
              : `<div class="tc-foot">
                   <span class="mono">${j.succeeded}/${j.total} OK · ${rtStr}</span>
                   <button type="button" class="tc-del-btn" data-del-job="${escapeHtml(j.job_id)}" title="${escapeHtml(t(state.lang, "task_btn_delete"))}">✕</button>
                 </div>`
          }
        </div>`;
    })
    .join("");

  listEl.querySelectorAll(".task-card").forEach((card) => {
    card.addEventListener("click", (e) => {
      if (e.target.closest("[data-del-job]")) return;
      selectTaskInCenter(card.dataset.jobId);
    });
  });

  listEl.querySelectorAll("[data-del-job]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const jid = btn.dataset.delJob;
      await apiFetch(`/api/v1/jobs/${encodeURIComponent(jid)}`, { method: "DELETE" });
      if (state.selectedTaskId === jid) {
        state.selectedTaskId = null;
        state.selectedTaskDetail = null;
        document.getElementById("tasks-detail-host").innerHTML = `<div class="card tasks-empty-panel"><h3>${escapeHtml(t(state.lang, "tasks_empty_title"))}</h3><p>${escapeHtml(t(state.lang, "tasks_select_hint"))}</p></div>`;
      }
      if (state.batchJobId === jid) {
        state.batchJobId = null;
        state.batchJobDetail = null;
        document.getElementById("batch-results-host")?.classList.add("hidden");
      }
      await refreshJobsList({ autoSelectFirst: true });
    });
  });
}

// ==================== UNIFIED RICH TASK / BATCH RESULT INSPECTOR ====================
function renderJobInspector(viewKey) {
  const hostEl = document.getElementById(viewKey === "tasks" ? "tasks-detail-host" : "batch-results-host");
  const detail = viewKey === "tasks" ? state.selectedTaskDetail : state.batchJobDetail;
  if (!hostEl) return;
  if (!detail) {
    if (viewKey === "batch") hostEl.classList.add("hidden");
    return;
  }
  hostEl.classList.remove("hidden");

  const ui = state.inspectorUi[viewKey];
  const sum = detail.summary || {};
  const isLive = detail.status === "running" || detail.status === "queued";
  const rtText = detail.runtime_ms >= 1000 ? `${(detail.runtime_ms / 1000).toFixed(1)} s` : `${Math.round(detail.runtime_ms || 0)} ms`;
  const hasTier2 = detail.include_tier2 || (detail.results || []).some((r) => (r.predictions || []).some((p) => p.tier === 2));
  if (ui.colView === "micro" && !hasTier2) ui.colView = "overview";

  const exportBase = apiUrl(`/api/v1/jobs/${encodeURIComponent(detail.job_id)}/export?client_id=${encodeURIComponent(getClientId())}`);

  const cols = COLUMN_PRESETS[ui.colView] || COLUMN_PRESETS.overview;
  const allRows = detail.results || [];
  let rows = [...allRows];
  if (ui.search) {
    rows = rows.filter(
      (r) =>
        (r.name || "").toLowerCase().includes(ui.search) ||
        (r.modifications || []).join(" ").toLowerCase().includes(ui.search)
    );
  }
  if (ui.riskFilter === "low_hem") {
    rows = rows.filter((r) => r.status === "ok" && r.key_metrics?.hemolysis_risk_val != null && r.key_metrics.hemolysis_risk_val < 0.25);
  } else if (ui.riskFilter === "long_hl") {
    rows = rows.filter((r) => r.status === "ok" && r.key_metrics?.half_life_val != null && r.key_metrics.half_life_val >= 12.0);
  } else if (ui.riskFilter === "modified") {
    rows = rows.filter(
      (r) =>
        r.status === "ok" &&
        ((r.mechanism_anchor && r.mechanism_anchor.detected) ||
          (r.modifications || []).some((m) => /cyclic|macrocycl|disulfide|fatty/i.test(m)))
    );
  }

  if (ui.sortCol) {
    const dir = ui.sortAsc ? 1 : -1;
    const colDef = cols.find((c) => c.key === ui.sortCol);
    rows.sort((a, b) => {
      if (ui.sortCol === "name") return (a.name || "").localeCompare(b.name || "") * dir;
      if (colDef && colDef.sortVal) return (colDef.sortVal(a) - colDef.sortVal(b)) * dir;
      return 0;
    });
  }

  const thHtml = cols
    .map((c) => {
      const arrow = ui.sortCol === c.key ? (ui.sortAsc ? " ↑" : " ↓") : "";
      return `<th data-sort="${escapeHtml(c.key)}" class="${c.num ? "num" : ""}">${escapeHtml(state.lang === "zh" ? c.label_zh : c.label_en)}${arrow}</th>`;
    })
    .join("");

  const tbodyHtml = rows.length
    ? rows
        .map((r) => {
          if (r.status !== "ok") {
            return `<tr><td></td><td><strong>${escapeHtml(r.name)}</strong></td><td colspan="${cols.length + 1}" style="color:var(--rose)">${escapeHtml(r.error || "Error")}</td></tr>`;
          }
          const checked = ui.selectedIds.has(r.index) ? "checked" : "";
          const lastCls = ui.lastInspectedIdx === r.index ? "row-last-inspected" : "";
          const modChip = (r.modifications || [])[0] || "";
          const cells = cols
            .map((c) => {
              const v = c.get(r);
              if (c.riskCell) {
                const lv = r.key_metrics?.hemolysis_risk_level || "info";
                return `<td class="nowrap"><span class="ep-risk risk-${escapeHtml(lv)}"><span class="risk-dot"></span>${escapeHtml(v)}</span></td>`;
              }
              const st = c.anchorStyle && r.mechanism_anchor?.detected ? 'style="color:var(--amber);font-weight:700"' : "";
              return `<td class="${c.num ? "num" : "nowrap"}" ${st}>${escapeHtml(v)}</td>`;
            })
            .join("");

          return `
            <tr class="clickable-mol-row ${lastCls}" data-mol-idx="${r.index}">
              <td><input type="checkbox" class="ins-plot-chk" data-idx="${r.index}" ${checked} aria-label="Plot ${escapeHtml(r.name)}" /></td>
              <td>
                <strong class="mol-cell-name">${escapeHtml(r.name)}</strong>
                ${modChip ? `<div class="mol-cell-mod">${escapeHtml(modChip)}</div>` : ""}
              </td>
              ${cells}
              <td class="nowrap"><button type="button" class="btn-inspect" data-inspect-idx="${r.index}">${escapeHtml(t(state.lang, "btn_detail"))}</button></td>
            </tr>`;
        })
        .join("")
    : `<tr><td colspan="${cols.length + 3}" class="empty-cell">${escapeHtml(t(state.lang, isLive ? "btn_running" : "no_ep_match"))}</td></tr>`;

  const etaStr = detail.eta_seconds != null ? t(state.lang, "task_eta", { s: Math.ceil(detail.eta_seconds) }) : "";
  const liveProgressHtml = isLive
    ? `<div class="ins-live-progress">
         <div class="ins-live-top">
           <span>${escapeHtml(t(state.lang, "task_progress_running", { item: detail.current_item || "…", done: detail.completed, tot: detail.total, pct: detail.progress_pct, eta: etaStr }))}</span>
           <button type="button" class="btn-ghost btn-sm btn-danger-soft" data-cancel-job="${escapeHtml(detail.job_id)}">${escapeHtml(t(state.lang, "task_btn_cancel"))}</button>
         </div>
         <div class="progress-bar-track"><div class="progress-bar-fill" style="width:${detail.progress_pct}%"></div></div>
       </div>`
    : "";

  hostEl.innerHTML = `
    <div class="task-inspector">
      <div class="card ins-header-card">
        <div class="ins-header-top">
          <div class="ins-title-group">
            <h2 class="ins-title">${escapeHtml(detail.title)}</h2>
            <span class="task-st-pill ${statusTone(detail.status)}">${isLive ? `<span class="mini-spin"></span>` : ""}${escapeHtml(statusLabel(detail.status))}</span>
            <span class="mono ins-meta-chip">${escapeHtml(detail.job_id)}</span>
            <span class="ins-meta-time">${escapeHtml(formatTimeShort(detail.created_at))} · ${rtText}</span>
          </div>
          <div class="ins-actions">
            ${viewKey === "batch" ? `<button type="button" class="btn-ghost btn-sm" data-open-in-tasks="${escapeHtml(detail.job_id)}">${escapeHtml(t(state.lang, "batch_view_in_tasks"))}</button>` : ""}
            <a class="btn-ghost btn-sm" href="${exportBase}&format=summary_csv" download>${escapeHtml(t(state.lang, "task_export_summary"))}</a>
            <a class="btn-ghost btn-sm" href="${exportBase}&format=full_csv" download>${escapeHtml(t(state.lang, "task_export_full"))}</a>
            <a class="btn-ghost btn-sm" href="${exportBase}&format=json" download>${escapeHtml(t(state.lang, "task_export_json"))}</a>
          </div>
        </div>
        ${liveProgressHtml}
        <div class="ins-kpi-strip">
          <div class="bkpi"><span>${escapeHtml(t(state.lang, "ins_success"))}</span><strong style="color:var(--emerald)">${detail.succeeded} / ${detail.total}</strong></div>
          <div class="bkpi"><span>${escapeHtml(t(state.lang, "ins_anchors"))}</span><strong style="color:var(--amber)">${sum.n_acylation_anchors ?? 0}</strong></div>
          <div class="bkpi"><span>${escapeHtml(t(state.lang, "ins_cyclic"))}</span><strong style="color:var(--cobalt)">${sum.n_cyclic_or_disulfide ?? 0}</strong></div>
          <div class="bkpi"><span>${escapeHtml(t(state.lang, "ins_low_hem"))}</span><strong style="color:var(--teal)">${sum.n_low_hemolysis ?? 0}</strong></div>
          <div class="bkpi"><span>${escapeHtml(t(state.lang, "ins_high_perm"))}</span><strong>${(sum.n_high_permeability ?? 0) + " / " + (sum.n_long_half_life ?? 0)}</strong></div>
        </div>
      </div>

      <div class="batch-analytics-grid">
        <div class="card batch-matrix-card">
          <div class="ins-matrix-toolbar">
            <div class="seg-group ins-col-tabs">
              <button type="button" class="seg-btn ${ui.colView === "overview" ? "active" : ""}" data-colv="overview">${escapeHtml(t(state.lang, "col_view_overview"))}</button>
              <button type="button" class="seg-btn ${ui.colView === "pk" ? "active" : ""}" data-colv="pk">${escapeHtml(t(state.lang, "col_view_pk"))}</button>
              <button type="button" class="seg-btn ${ui.colView === "perm" ? "active" : ""}" data-colv="perm">${escapeHtml(t(state.lang, "col_view_perm"))}</button>
              <button type="button" class="seg-btn ${ui.colView === "safety" ? "active" : ""}" data-colv="safety">${escapeHtml(t(state.lang, "col_view_safety"))}</button>
              ${hasTier2 ? `<button type="button" class="seg-btn ${ui.colView === "micro" ? "active" : ""}" data-colv="micro">${escapeHtml(t(state.lang, "col_view_micro"))}</button>` : ""}
            </div>
            <div class="ins-filter-bar">
              <button type="button" class="filter-chip ${ui.riskFilter === "all" ? "active" : ""}" data-rf="all">${escapeHtml(t(state.lang, "rf_all"))}</button>
              <button type="button" class="filter-chip ${ui.riskFilter === "low_hem" ? "active" : ""}" data-rf="low_hem">${escapeHtml(t(state.lang, "rf_low_hem"))}</button>
              <button type="button" class="filter-chip ${ui.riskFilter === "long_hl" ? "active" : ""}" data-rf="long_hl">${escapeHtml(t(state.lang, "rf_long_hl"))}</button>
              <button type="button" class="filter-chip ${ui.riskFilter === "modified" ? "active" : ""}" data-rf="modified">${escapeHtml(t(state.lang, "rf_modified"))}</button>
              <input type="search" class="search-sm ins-mol-search" value="${escapeHtml(ui.search)}" placeholder="${escapeHtml(t(state.lang, "filter_mol_ph"))}" />
            </div>
          </div>

          <div class="ins-matrix-hint">${escapeHtml(t(state.lang, "matrix_sub"))}</div>

          <div class="table-scroll">
            <table class="data-table sortable-table">
              <thead>
                <tr>
                  <th class="col-check">Plot</th>
                  <th data-sort="name">${escapeHtml(t(state.lang, "th_name"))}${ui.sortCol === "name" ? (ui.sortAsc ? " ↑" : " ↓") : ""}</th>
                  ${thHtml}
                  <th></th>
                </tr>
              </thead>
              <tbody>${tbodyHtml}</tbody>
            </table>
          </div>
        </div>

        <div class="card batch-radar-card">
          <div class="card-head">
            <div>
              <h2 class="card-title">${escapeHtml(t(state.lang, "radar_cmp_title"))}</h2>
              <span class="card-sub">${escapeHtml(t(state.lang, "radar_cmp_sub"))}</span>
            </div>
          </div>
          <div class="batch-radar-stage" id="radar-stage-${viewKey}"></div>
          <div class="batch-radar-legend" id="radar-legend-${viewKey}"></div>
        </div>
      </div>
    </div>`;

  // Wire inspector events
  hostEl.querySelectorAll("[data-colv]").forEach((b) => {
    b.addEventListener("click", () => {
      ui.colView = b.dataset.colv;
      ui.sortCol = null;
      renderJobInspector(viewKey);
    });
  });
  hostEl.querySelectorAll("[data-rf]").forEach((b) => {
    b.addEventListener("click", () => {
      ui.riskFilter = b.dataset.rf;
      renderJobInspector(viewKey);
    });
  });
  const searchEl = hostEl.querySelector(".ins-mol-search");
  searchEl?.addEventListener("input", (e) => {
    ui.search = e.target.value.trim().toLowerCase();
    renderJobInspector(viewKey);
    const nextInput = hostEl.querySelector(".ins-mol-search");
    if (nextInput) {
      nextInput.focus();
      nextInput.setSelectionRange(nextInput.value.length, nextInput.value.length);
    }
  });
  hostEl.querySelectorAll("th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (ui.sortCol === col) ui.sortAsc = !ui.sortAsc;
      else {
        ui.sortCol = col;
        ui.sortAsc = true;
      }
      renderJobInspector(viewKey);
    });
  });

  hostEl.querySelectorAll(".ins-plot-chk").forEach((chk) => {
    chk.addEventListener("click", (e) => e.stopPropagation());
    chk.addEventListener("change", () => {
      const idx = Number(chk.dataset.idx);
      if (chk.checked) {
        if (ui.selectedIds.size >= 4) {
          const oldest = ui.selectedIds.values().next().value;
          ui.selectedIds.delete(oldest);
        }
        ui.selectedIds.add(idx);
      } else {
        ui.selectedIds.delete(idx);
      }
      renderJobInspector(viewKey);
    });
  });

  // Row click OR button click -> Instant 0ms Compound Detail with Return Bar!
  hostEl.querySelectorAll(".clickable-mol-row").forEach((tr) => {
    tr.addEventListener("click", (e) => {
      if (e.target.closest("input[type='checkbox']")) return;
      openCompoundFromInspector(viewKey, detail, Number(tr.dataset.molIdx));
    });
  });
  hostEl.querySelectorAll("[data-inspect-idx]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      openCompoundFromInspector(viewKey, detail, Number(btn.dataset.inspectIdx));
    });
  });

  hostEl.querySelector("[data-cancel-job]")?.addEventListener("click", async (e) => {
    const jid = e.currentTarget.dataset.cancelJob;
    await apiFetch(`/api/v1/jobs/${encodeURIComponent(jid)}/cancel`, { method: "POST" });
    await refreshJobsList({ autoSelectFirst: false });
    await fetchAndRenderJobDetail(jid, [viewKey]);
  });

  hostEl.querySelector("[data-open-in-tasks]")?.addEventListener("click", async (e) => {
    const jid = e.currentTarget.dataset.openInTasks;
    await selectTaskInCenter(jid);
    switchTab("tasks");
  });

  const selectedMols = allRows.filter((r) => r.status === "ok" && ui.selectedIds.has(r.index));
  renderBatchRadarOverlay(
    document.getElementById(`radar-stage-${viewKey}`),
    document.getElementById(`radar-legend-${viewKey}`),
    selectedMols,
    state.lang,
    t(state.lang, "radar_cmp_empty")
  );
}

// ==================== ENDPOINTS & VALIDATION WORKSPACE ====================
function wireValidationWorkspace() {
  document.querySelectorAll("#val-tier-filter .seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      state.valTier = b.dataset.tier;
      document.querySelectorAll("#val-tier-filter .seg-btn").forEach((x) => x.classList.toggle("active", x.dataset.tier === state.valTier));
      renderValidationTable();
    });
  });
  document.getElementById("val-domain-filter")?.addEventListener("change", (e) => {
    state.valDomain = e.target.value;
    renderValidationTable();
  });
  document.getElementById("val-search")?.addEventListener("input", (e) => {
    state.valSearch = e.target.value.trim().toLowerCase();
    renderValidationTable();
  });
}

async function loadEndpointsCatalog() {
  try {
    const res = await apiFetch("/api/v1/endpoints");
    const data = await res.json();
    state.catalogData = data;
    syncAdmetSvgStats(document.getElementById("admet-map-host"), data, state.lang);
    renderAdmetHotspotDetail(
      document.getElementById("admet-detail-card"),
      state.activeHotspot,
      data,
      state.lang,
      (q) => {
        window.scrollTo({ top: 0, behavior: "smooth" });
        runExampleQuery(q);
      }
    );
    renderAllCase17Charts();
    renderValidationKpis(data);
    renderValidationHighlights(data);
    renderValidationTable();
  } catch (err) {
    console.error("Failed to load endpoints catalog:", err);
  }
}

function renderValidationKpis(data) {
  const ds = data.dataset_summary || {};
  const os = data.overall_summary || {};
  const eps = data.endpoints || [];

  const clsEps = eps.filter((e) => e.kind === "classification");
  const regEps = eps.filter((e) => e.kind === "regression");
  const clsM = os.classification_holdout_dual_means || os.classification_6_holdout_dual_means || {};
  const regM = os.regression_holdout_dual_means || os.regression_18_holdout_dual_means || {};

  const nComp = ds.n_unique_compounds ?? 0;
  const nRows = ds.n_total_rows ?? 0;
  const nCens = ds.n_censored_rows ?? 0;
  const nT1 = ds.n_tier1_endpoints ?? eps.filter((e) => e.tier === 1).length;
  const nT2 = ds.n_tier2_endpoints ?? eps.filter((e) => e.tier === 2).length;

  const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
  const meanAuc = avg(clsEps.map((e) => e.holdout_score_v33).filter((v) => v != null)) ?? clsM.v33_champion_roc_auc;
  const meanRho = avg(regEps.map((e) => e.holdout_score_v33).filter((v) => v != null)) ?? regM.v33_champion_spearman;
  const meanCov80 = avg(regEps.map((e) => e.conformal_coverage_80).filter((v) => v != null));

  const bAll = document.getElementById("val-btn-all");
  const bT1 = document.getElementById("val-btn-t1");
  const bT2 = document.getElementById("val-btn-t2");
  if (bAll) bAll.textContent = t(state.lang, "val_all_eps", { n: eps.length });
  if (bT1) bT1.textContent = t(state.lang, "val_t1_eps", { n: nT1 });
  if (bT2) bT2.textContent = t(state.lang, "val_t2_eps", { n: nT2 });

  document.getElementById("val-kpi-grid").innerHTML = `
    <div class="vkpi">
      <span class="vkpi-lbl">${escapeHtml(t(state.lang, "vkpi_peptides"))}</span>
      <span class="vkpi-val">${Number(nComp).toLocaleString()}</span>
      <span class="vkpi-sub">${escapeHtml(t(state.lang, "vkpi_peptides_sub"))}</span>
    </div>
    <div class="vkpi">
      <span class="vkpi-lbl">${escapeHtml(t(state.lang, "vkpi_eps"))}</span>
      <span class="vkpi-val">${eps.length}</span>
      <span class="vkpi-sub">Tier 1: ${nT1} · Tier 2: ${nT2}</span>
    </div>
    <div class="vkpi">
      <span class="vkpi-lbl">${escapeHtml(t(state.lang, "vkpi_records"))}</span>
      <span class="vkpi-val">${Number(nRows).toLocaleString()}</span>
      <span class="vkpi-sub">${escapeHtml(t(state.lang, "vkpi_records_sub", { n: Number(nCens).toLocaleString() }))}</span>
    </div>
    <div class="vkpi">
      <span class="vkpi-lbl">${escapeHtml(t(state.lang, "vkpi_auc", { n: clsEps.length }))}</span>
      <span class="vkpi-val" style="color:var(--cobalt)">${meanAuc != null ? Number(meanAuc).toFixed(3) : "—"}</span>
      <span class="vkpi-sub">Brier ${clsM.v33_champion_brier != null ? Number(clsM.v33_champion_brier).toFixed(3) : "—"} · ECE ${clsM.v33_champion_ece != null ? Number(clsM.v33_champion_ece).toFixed(3) : "—"}</span>
    </div>
    <div class="vkpi">
      <span class="vkpi-lbl">${escapeHtml(t(state.lang, "vkpi_rho", { n: regEps.length }))}</span>
      <span class="vkpi-val" style="color:var(--emerald)">${meanRho != null ? Number(meanRho).toFixed(3) : "—"}</span>
      <span class="vkpi-sub">${escapeHtml(t(state.lang, "vkpi_cov_sub", { v: meanCov80 != null ? (Number(meanCov80) * 100).toFixed(1) + "%" : "—" }))}</span>
    </div>`;
}

function renderValidationHighlights(data) {
  const c17 = data.case17_scorecard || {};
  const hlPure = c17.half_life_pure_ml_route_matched_log10 || {};
  const hlAnch = c17.half_life_with_mechanism_anchor_log10 || {};
  const ppb = c17.ppb_dual || {};
  const hem = c17.hemolysis_false_positive_check || {};

  document.getElementById("val-case17-card").innerHTML = `
    <div class="card-head" style="margin-bottom:10px;">
      <div>
        <h2 class="card-title">${escapeHtml(t(state.lang, "val_c17_title"))}</h2>
        <span class="card-sub">${escapeHtml(t(state.lang, "val_c17_sub"))}</span>
      </div>
      <span class="ad-pill ad-in_domain">${escapeHtml(t(state.lang, "val_c17_fp_badge"))}</span>
    </div>
    <div class="c17-metrics-grid">
      <div class="c17-stat">
        <span class="dm-lbl">${escapeHtml(t(state.lang, "val_c17_hl_pure"))}</span>
        <span class="c17-val">ρ = ${hlPure.spearman != null ? Number(hlPure.spearman).toFixed(3) : "—"}</span>
        <span class="c17-sub">R² = ${hlPure.r2 != null ? Number(hlPure.r2).toFixed(3) : "—"} · MAE ${hlPure.mae != null ? Number(hlPure.mae).toFixed(3) : "—"} log₁₀ h</span>
      </div>
      <div class="c17-stat">
        <span class="dm-lbl">${escapeHtml(t(state.lang, "val_c17_hl_anch"))}</span>
        <span class="c17-val" style="color:var(--emerald)">MAE ${hlAnch.mae != null ? Number(hlAnch.mae).toFixed(3) : "—"}</span>
        <span class="c17-sub">R² = ${hlAnch.r2 != null ? Number(hlAnch.r2).toFixed(3) : "—"} · ρ = ${hlAnch.spearman != null ? Number(hlAnch.spearman).toFixed(3) : "—"} (n=${hlAnch.n ?? 16})</span>
      </div>
      <div class="c17-stat">
        <span class="dm-lbl">${escapeHtml(t(state.lang, "val_c17_ppb"))}</span>
        <span class="c17-val">ρ = ${ppb.spearman != null ? Number(ppb.spearman).toFixed(3) : "—"}</span>
        <span class="c17-sub">Pearson r = ${ppb.pearson != null ? Number(ppb.pearson).toFixed(3) : "—"} (n=${ppb.n ?? 14})</span>
      </div>
      <div class="c17-stat">
        <span class="dm-lbl">${escapeHtml(t(state.lang, "val_c17_hem"))}</span>
        <span class="c17-val" style="color:var(--emerald)">${hem.n_drugs_gt_0_50 ?? 0} / 17 FP</span>
        <span class="c17-sub">Max P = ${hem.max_prob_across_17_fda_drugs != null ? (Number(hem.max_prob_across_17_fda_drugs) * 100).toFixed(1) + "%" : "—"}</span>
      </div>
    </div>`;

  const rep = data.representation_ablation_summary || {};
  const wins = rep.single_foundation_model_holdout_wins || rep.single_foundation_model_holdout_wins_out_of_24 || {};
  const famCounts = (data.overall_summary || {}).dual_modality_winning_family_counts || {};
  const totalEp = (data.endpoints || []).length || 46;
  const fmDenom = Object.values(wins).reduce((a, b) => a + Number(b || 0), 0) || totalEp;

  const order = ["ncaa_repdistill_v2", "esmc_600m", "peptideclm2", "chemberta_mtr"];
  const barsHtml = order
    .map((k) => {
      const meta = FM_DISPLAY_NAMES[k] || { label: k, modality: "", color: "#0369A1" };
      const cnt = Number(wins[k] ?? 0);
      const pct = Math.round((cnt / fmDenom) * 100);
      return `
        <div class="fm-row">
          <div class="fm-meta">
            <span class="fm-name">${escapeHtml(meta.label)}</span>
            <span class="fm-mod">${escapeHtml(meta.modality)}</span>
          </div>
          <div class="fm-bar-track"><div class="fm-bar-fill" style="width:${Math.max(8, pct)}%;background:${meta.color};"></div></div>
          <span class="fm-score mono"><strong>${cnt}</strong>/${fmDenom}</span>
        </div>`;
    })
    .join("");

  const famSummary = Object.entries(famCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<span class="mod-badge tone-slate mono">${escapeHtml(k)}: ${v}/${totalEp}</span>`)
    .join("");

  document.getElementById("val-fm-strip").innerHTML = `
    <div class="card-head" style="margin-bottom:10px;">
      <div>
        <h2 class="card-title">${escapeHtml(t(state.lang, "val_fm_title"))}</h2>
        <span class="card-sub">${escapeHtml(t(state.lang, "val_fm_sub", { n: totalEp }))}</span>
      </div>
    </div>
    <div class="fm-bars-list">${barsHtml}</div>
    <div class="fm-fam-row">${famSummary}</div>`;
}

function renderValidationTable() {
  if (!state.catalogData) return;
  const eps = (state.catalogData.endpoints || []).filter((e) => {
    if (state.valTier !== "all" && String(e.tier) !== state.valTier) return false;
    if (state.valDomain !== "all" && e.domain !== state.valDomain) return false;
    if (state.valSearch) {
      const hay = `${e.endpoint} ${e.display_en} ${e.display_zh} ${e.model_family} ${e.winning_family} ${e.scope_note}`.toLowerCase();
      if (!hay.includes(state.valSearch)) return false;
    }
    return true;
  });

  const tbody = document.getElementById("val-tbody");
  tbody.innerHTML = eps
    .map((e) => {
      const dm = DOMAIN_META[e.domain] || { en: e.domain, zh: e.domain };
      const famTone =
        e.winning_family === "super_ensemble"
          ? "tone-cobalt"
          : e.winning_family === "fusion"
            ? "tone-teal"
            : e.winning_family === "deep"
              ? "tone-violet"
              : "tone-slate";

      let covOrCal = "—";
      if (e.kind === "classification") {
        if (e.brier_holdout != null && e.ece_holdout != null) {
          covOrCal = `<span title="Brier / ECE">B ${e.brier_holdout.toFixed(3)} · E ${e.ece_holdout.toFixed(3)}</span>`;
        }
      } else if (e.conformal_coverage_80 != null) {
        const covPct = (e.conformal_coverage_80 * 100).toFixed(1) + "%";
        const lowBadge =
          e.uq_reliability === "low"
            ? `<div style="margin-top:2px;"><span class="mod-badge tone-amber" style="font-size:10px;padding:1px 5px;" title="${escapeHtml(t(state.lang, "uq_low_tip"))}">⚠ ${escapeHtml(t(state.lang, "uq_low_badge"))}</span></div>`
            : "";
        covOrCal = `<span title="80% Empirical Cov (90%: ${e.conformal_coverage_90 != null ? (e.conformal_coverage_90 * 100).toFixed(1) + "%" : "—"})">${covPct}</span>${lowBadge}`;
      }

      const namePrimary = state.lang === "zh" ? e.display_zh : e.display_en;
      const nameSub = state.lang === "zh" ? e.display_en : e.display_zh;

      return `
        <tr>
          <td>
            <strong>${escapeHtml(namePrimary)}</strong>
            <div class="val-sub-row">${escapeHtml(nameSub)} · <code>${escapeHtml(e.endpoint)}</code></div>
          </td>
          <td class="nowrap">
            <span class="mod-badge ${e.tier === 1 ? "tone-cobalt" : "tone-amber"}">Tier ${e.tier}</span>
            <div class="val-sub-row">${escapeHtml(state.lang === "zh" ? dm.zh : dm.en.split(" ")[0])}</div>
          </td>
          <td class="nowrap">
            <span class="mono" style="font-size:11.5px">${escapeHtml(e.kind === "classification" ? "cls · AUC" : "reg · ρ")}</span>
            <div class="val-sub-row">${escapeHtml(e.unit)}</div>
          </td>
          <td class="num">${e.n_train.toLocaleString()}</td>
          <td class="num">${(e.n_holdout || 0).toLocaleString()}</td>
          <td class="num">${e.n_censored ? e.n_censored.toLocaleString() : "0"}</td>
          <td>
            <span class="mod-badge ${famTone}" style="font-size:11px;">${escapeHtml(e.winning_family)}</span>
            <div class="val-sub-row">${escapeHtml(e.model_family)}</div>
          </td>
          <td class="num"><strong>${e.cv_score_v33 != null ? e.cv_score_v33.toFixed(3) : "—"}</strong></td>
          <td class="num"><strong style="color:var(--cobalt)">${e.holdout_score_v33 != null ? e.holdout_score_v33.toFixed(3) : "—"}</strong></td>
          <td class="num">${covOrCal}</td>
        </tr>`;
    })
    .join("");
}

// ==================== API DOCS WORKSPACE ====================
const API_PRESETS = {
  predict_sema: {
    method: "POST",
    url: "/api/v1/predict",
    body: { query: "Semaglutide", include_tier2: true, include_svg: false, conditions: { rbc_species: "human", pk_context: "human_sc" } },
  },
  predict_mica: {
    method: "POST",
    url: "/api/v1/predict",
    body: { query: "Micafungin", include_tier2: true, include_svg: false },
  },
  atom_attr_csa: {
    method: "POST",
    url: "/api/v1/atom-attribution",
    body: { query: "Cyclosporine A", endpoint: "PAMPA", width: 760, height: 460 },
  },
  render_csa: {
    method: "POST",
    url: "/api/v1/render-svg",
    body: { query: "Cyclosporine A", width: 560, height: 340 },
  },
  jobs_list: { method: "GET", url: "/api/v1/jobs", body: null },
  health: { method: "GET", url: "/api/v1/health", body: null },
  endpoints: { method: "GET", url: "/api/v1/endpoints", body: null },
};

function wireApiWorkspace() {
  renderCodeSnippet("curl");
  document.querySelectorAll(".code-tab").forEach((tBtn) => {
    tBtn.addEventListener("click", () => {
      state.codeLang = tBtn.dataset.lang;
      document.querySelectorAll(".code-tab").forEach((x) => x.classList.toggle("active", x.dataset.lang === state.codeLang));
      renderCodeSnippet(state.codeLang);
    });
  });

  document.getElementById("btn-copy-snippet")?.addEventListener("click", async () => {
    const txt = document.getElementById("api-code-snippet").textContent;
    await navigator.clipboard?.writeText(txt);
    const btn = document.getElementById("btn-copy-snippet");
    btn.textContent = t(state.lang, "api_copied");
    setTimeout(() => (btn.textContent = t(state.lang, "api_copy")), 1400);
  });

  const sel = document.getElementById("api-preset-select");
  const bodyEl = document.getElementById("api-play-body");
  const syncPreset = () => {
    const p = API_PRESETS[sel.value];
    bodyEl.value = p.body ? JSON.stringify(p.body, null, 2) : "// GET request — no JSON body required";
    bodyEl.disabled = !p.body;
  };
  sel?.addEventListener("change", syncPreset);
  syncPreset();

  document.getElementById("btn-api-try")?.addEventListener("click", async () => {
    const p = API_PRESETS[sel.value];
    const statusEl = document.getElementById("api-play-status");
    const outEl = document.getElementById("api-play-output");
    statusEl.textContent = t(state.lang, "api_sending");
    const t0 = performance.now();
    try {
      const opts = { method: p.method, headers: {} };
      if (p.method === "POST") {
        opts.headers["Content-Type"] = "application/json";
        opts.body = bodyEl.value;
      }
      const res = await apiFetch(p.url, opts);
      const data = await res.json();
      const dt = (performance.now() - t0).toFixed(0);
      statusEl.textContent = `HTTP ${res.status} · ${dt} ms`;
      if (data.svg && data.svg.length > 200) {
        data.svg = data.svg.slice(0, 140) + `… [${data.svg.length} chars SVG truncated]`;
      }
      outEl.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      statusEl.textContent = "Error";
      outEl.textContent = String(err);
    }
  });
}

function renderCodeSnippet(lang) {
  const baseFull = `${window.location.origin}${API_BASE}`;
  const el = document.getElementById("api-code-snippet");
  if (!el) return;
  if (lang === "curl") {
    el.textContent = `# 1) Single-peptide prediction (46 endpoints)
curl -sS -X POST "${baseFull}/api/v1/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"query":"Semaglutide","include_tier2":true,"conditions":{"rbc_species":"human","pk_context":"human_sc"}}'

# 2) Submit async batch task & poll progress
curl -sS -X POST "${baseFull}/api/v1/jobs" \\
  -H "Content-Type: application/json" \\
  -H "X-Client-Id: my_lab_session" \\
  -d '{"title":"Lead Series","raw_text":"Semaglutide\\nMicafungin\\nCyclosporine A","include_tier2":true}'`;
  } else {
    el.textContent = `import httpx

headers = {"X-Client-Id": "my_lab_session"}
job = httpx.post(
    "${baseFull}/api/v1/jobs",
    headers=headers,
    json={"title": "Lead Series", "raw_text": "Semaglutide\\nMicafungin\\nCyclosporine A", "include_tier2": True},
).json()

detail = httpx.get(f"${baseFull}/api/v1/jobs/{job['job_id']}", headers=headers).json()
print("Status:", detail["status"], "Completed:", detail["completed"], "/", detail["total"])`;
  }
}

// ==================== UTILITIES ====================
function slugify(s) {
  return String(s || "peptide").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 40) || "peptide";
}

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

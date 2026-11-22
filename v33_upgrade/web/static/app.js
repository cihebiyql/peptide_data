/**
 * Peptide OmniPanel V33 — Frontend SPA Controller (Clean ES Modules, Subpath-Safe)
 */

import { buildIntervalBarHtml, escapeHtml, renderAdmetRadar, renderBatchRadarOverlay } from "./charts.js";

// Subpath-safe API prefix: works cleanly at `/` AND behind Nginx e.g. `/peptide-next/`
const API_BASE = (() => {
  const p = window.location.pathname || "/";
  if (p === "/" || p === "") return "";
  return p.endsWith("/") ? p.slice(0, -1) : p.replace(/\/[^/]*$/, "");
})();

function apiUrl(path) {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${clean}`;
}

const DOMAIN_META = {
  pk: { title: "Pharmacokinetics & PK", zh: "药代动力学", dot: "dot-pk" },
  permeability: { title: "Membrane Permeability & Efflux", zh: "膜渗透与外排", dot: "dot-permeability" },
  safety: { title: "Safety, Hemolysis & Cytotoxicity", zh: "安全性、溶血与细胞毒", dot: "dot-safety" },
  physicochemical: { title: "Physicochemical & In Vitro", zh: "理化性质与溶解度", dot: "dot-physicochemical" },
  antimicrobial: { title: "Antimicrobial, Bactericidal & Biofilm", zh: "抗菌谱、杀菌与抗生物膜", dot: "dot-antimicrobial" },
};

const AD_LABELS = {
  in_domain: "In-Domain",
  borderline: "Borderline",
  ood: "OOD",
  mechanism: "Anchor",
};

const FM_DISPLAY_NAMES = {
  ncaa_repdistill_v2: { label: "ncAA-RepDistill v2", modality: "SMILES · 768+2048+1280d", color: "#0369A1" },
  esmc_600m: { label: "ESM-C 600M (4-Region)", modality: "Sequence · 4×1152d", color: "#0D9488" },
  peptideclm2: { label: "PeptideCLM-2", modality: "SMILES · 512d", color: "#7C3AED" },
  chemberta_mtr: { label: "ChemBERTa-MTR", modality: "SMILES · 384d", color: "#D97706" },
};

const state = {
  activeTab: "predict",
  examples: [],
  exampleCategories: [],
  drugs: [],
  batchDemoCsv: "",
  batchPresets: {},
  rbcSpecies: "human",
  lastPrediction: null,
  activeTier: 1,
  tier2Group: "all",
  adFilter: "all",
  endpointSearch: "",
  // Batch state
  batchParsed: null,
  batchResults: null,
  batchSort: { col: null, asc: true },
  batchSelectedIds: new Set(),
  // Validation state
  catalogData: null,
  valTier: "all",
  valDomain: "all",
  valSearch: "",
  // API snippets
  codeLang: "curl",
};

// ==================== BOOTSTRAP ====================
document.addEventListener("DOMContentLoaded", async () => {
  wireNavigation();
  wirePredictWorkspace();
  wireBatchWorkspace();
  wireValidationWorkspace();
  wireApiWorkspace();
  wireSvgModal();

  await Promise.all([loadHealth(), loadExamples()]);
});

// ==================== NAVIGATION ====================
function wireNavigation() {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  document.getElementById("brand-home").addEventListener("click", (e) => {
    e.preventDefault();
    switchTab("predict");
    if (state.lastPrediction) {
      resetToLanding();
    }
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
  if (tab === "validation" && !state.catalogData) {
    loadEndpointsCatalog();
  }
}

function resetToLanding() {
  state.lastPrediction = null;
  document.getElementById("predict-hero").classList.add("hero-centered");
  document.getElementById("predict-hero").classList.remove("hero-compact");
  document.getElementById("compact-actions").classList.add("hidden");
  document.getElementById("landing-overview").classList.remove("hidden");
  document.getElementById("predict-results").classList.add("hidden");
}

// ==================== HEALTH & EXAMPLES ====================
async function loadHealth() {
  const pill = document.getElementById("backend-status");
  const txt = pill.querySelector(".status-text");
  try {
    const res = await fetch(apiUrl("/api/v1/health"));
    const data = await res.json();
    pill.classList.remove("offline", "fallback");
    if (data.adapter_mode === "v33_remote") {
      txt.textContent = `v33_remote · ${data.endpoints_total} EPs`;
      pill.title = `Live V33 Super-Ensemble Engine (${data.v33_remote_url || "connected"}) · RDKit ${data.rdkit_version}`;
    } else {
      pill.classList.add("fallback");
      txt.textContent = `${data.adapter_mode} · RDKit ${data.rdkit_version.split(".")[0]}`;
      pill.title = "Offline manifest fallback mode (connect V33_REMOTE_URL for live inference)";
    }

    // Hydrate landing KPI ribbon strictly from audited v33_manifest.json dataset_summary
    const ds = data.dataset_summary || {};
    const compEl = document.getElementById("kpi-compounds");
    if (compEl && ds.n_unique_compounds) {
      compEl.textContent = Number(ds.n_unique_compounds).toLocaleString();
    }
    const epEl = document.getElementById("kpi-ep-count");
    const epLbl = document.getElementById("kpi-ep-lbl");
    if (epEl && data.endpoints_total) {
      epEl.textContent = String(data.endpoints_total);
    }
    if (epLbl && data.tier1_endpoints != null && data.tier2_endpoints != null) {
      epLbl.textContent = `Tiered ADMET & Antimicrobial Endpoints (${data.tier1_endpoints} + ${data.tier2_endpoints})`;
    }
    const advT2 = document.getElementById("adv-tier2-label");
    if (advT2 && data.tier2_endpoints != null) {
      advT2.textContent = `Include Tier-2 Antimicrobial, Bactericidal & Biofilm Endpoints (${data.tier2_endpoints})`;
    }
    const recEl = document.getElementById("kpi-records");
    const recLbl = document.getElementById("kpi-records-lbl");
    if (recEl && ds.n_total_rows) {
      recEl.textContent = Number(ds.n_total_rows).toLocaleString();
    }
    if (recLbl && ds.n_censored_rows != null) {
      recLbl.textContent = `Harmonized Records (${Number(ds.n_censored_rows).toLocaleString()} Censored)`;
    }
  } catch {
    pill.classList.add("offline");
    txt.textContent = "Backend Offline";
  }
}

async function loadExamples() {
  try {
    const res = await fetch(apiUrl("/api/v1/examples"));
    const data = await res.json();
    state.examples = data.quick_examples || [];
    state.exampleCategories = data.example_categories || [];
    state.drugs = data.drugs || [];
    state.batchDemoCsv = data.batch_demo_csv || "";
    state.batchPresets = data.batch_presets || {};

    const pillsWrap = document.getElementById("example-pills");
    pillsWrap.innerHTML = `<span class="examples-label">Examples:</span>`;
    state.examples.forEach((ex) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "example-pill";
      btn.title = ex.subtitle || ex.label;
      btn.innerHTML = `${escapeHtml(ex.label)} <span class="pill-tag">${escapeHtml(ex.tag)}</span>`;
      btn.addEventListener("click", () => runExampleQuery(ex.query));
      pillsWrap.appendChild(btn);
    });

    const btnLbl = document.getElementById("exlib-btn-label");
    if (btnLbl && state.drugs.length) {
      btnLbl.textContent = `All ${state.drugs.length} Examples`;
    }
    renderExampleLibrary();
  } catch (err) {
    console.error("Failed to load examples:", err);
  }
}

function runExampleQuery(qVal) {
  document.getElementById("smart-query-input").value = qVal;
  document.getElementById("adv-sequence").value = "";
  document.getElementById("adv-smiles").value = "";
  const libPanel = document.getElementById("example-library-panel");
  if (libPanel) libPanel.classList.add("hidden");
  const libBtn = document.getElementById("btn-toggle-exlib");
  if (libBtn) libBtn.setAttribute("aria-expanded", "false");
  updateLiveDetection(qVal);
  runSinglePrediction();
}

function renderExampleLibrary() {
  const grid = document.getElementById("exlib-grid");
  if (!grid || !state.drugs.length) return;

  const cats = state.exampleCategories.length
    ? state.exampleCategories
    : [{ id: "all", title_en: "Curated Peptides", title_zh: "精选多肽示例", badge: "" }];

  grid.innerHTML = cats
    .map((cat) => {
      const items = state.drugs.filter((d) => cat.id === "all" || d.category === cat.id);
      if (!items.length) return "";
      const chipsHtml = items
        .map((d) => {
          const q = d.query || d.display_name;
          const metaParts = [];
          if (d.fda_half_life) metaParts.push(`T½: ${d.fda_half_life.split("(")[0].trim()}`);
          else metaParts.push(d.drug_class);
          return `
            <button type="button" class="exlib-chip" data-query="${escapeHtml(q)}" title="${escapeHtml(`${d.display_name} (${d.zh_name}) — ${d.drug_class}`)}">
              <div class="exlib-chip-top">
                <span class="exlib-chip-en">${escapeHtml(d.display_name)}</span>
                <span class="exlib-chip-zh">${escapeHtml(d.zh_name)}</span>
                ${d.case17 ? `<span class="exlib-c17-dot" title="Case17 Clinical Benchmark">C17</span>` : ""}
              </div>
              <div class="exlib-chip-sub">${escapeHtml(metaParts.join(" · "))}</div>
            </button>`;
        })
        .join("");

      return `
        <div class="exlib-group">
          <div class="exlib-group-head">
            <span class="exlib-group-title">${escapeHtml(cat.title_zh)}</span>
            <span class="exlib-group-en">${escapeHtml(cat.title_en)}</span>
          </div>
          <div class="exlib-chip-list">${chipsHtml}</div>
        </div>`;
    })
    .join("");

  grid.querySelectorAll(".exlib-chip").forEach((chip) => {
    chip.addEventListener("click", () => runExampleQuery(chip.dataset.query));
  });
}

// ==================== PREDICT WORKSPACE ====================
function wirePredictWorkspace() {
  const form = document.getElementById("predict-form");
  const input = document.getElementById("smart-query-input");
  const acMenu = document.getElementById("autocomplete-list");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    acMenu.classList.add("hidden");
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

  // Curated Example Library Drawer toggle
  const exlibBtn = document.getElementById("btn-toggle-exlib");
  const exlibPanel = document.getElementById("example-library-panel");
  const exlibClose = document.getElementById("btn-close-exlib");
  if (exlibBtn && exlibPanel) {
    exlibBtn.addEventListener("click", () => {
      const isHidden = exlibPanel.classList.toggle("hidden");
      exlibBtn.setAttribute("aria-expanded", String(!isHidden));
      if (!isHidden) {
        document.getElementById("advanced-panel")?.classList.add("hidden");
        document.getElementById("btn-toggle-advanced")?.setAttribute("aria-expanded", "false");
      }
    });
  }
  if (exlibClose && exlibPanel) {
    exlibClose.addEventListener("click", () => {
      exlibPanel.classList.add("hidden");
      exlibBtn?.setAttribute("aria-expanded", "false");
    });
  }

  // Advanced Options toggle
  const advBtn = document.getElementById("btn-toggle-advanced");
  const advPanel = document.getElementById("advanced-panel");
  advBtn.addEventListener("click", () => {
    const isHidden = advPanel.classList.toggle("hidden");
    advBtn.setAttribute("aria-expanded", String(!isHidden));
    if (!isHidden && exlibPanel) {
      exlibPanel.classList.add("hidden");
      exlibBtn?.setAttribute("aria-expanded", "false");
    }
  });

  // RBC Species Segmented Control
  document.querySelectorAll("#rbc-species-group .seg-btn").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#rbc-species-group .seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.rbcSpecies = b.dataset.val;
    });
  });

  // Tier switcher
  document.querySelectorAll(".tier-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.activeTier = Number(tab.dataset.tier);
      document.querySelectorAll(".tier-tab").forEach((t) => t.classList.toggle("active", Number(t.dataset.tier) === state.activeTier));
      document.getElementById("tier1-dashboard").classList.toggle("hidden", state.activeTier !== 1);
      document.getElementById("tier2-dashboard").classList.toggle("hidden", state.activeTier !== 2);
    });
  });

  // AD filter chips
  document.querySelectorAll("#ad-filter-group .filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.adFilter = chip.dataset.ad;
      document.querySelectorAll("#ad-filter-group .filter-chip").forEach((c) => c.classList.toggle("active", c.dataset.ad === state.adFilter));
      if (state.lastPrediction) renderEndpointDashboards(state.lastPrediction);
    });
  });

  document.getElementById("endpoint-search").addEventListener("input", (e) => {
    state.endpointSearch = e.target.value.trim().toLowerCase();
    if (state.lastPrediction) renderEndpointDashboards(state.lastPrediction);
  });

  // Export CSV / JSON / Print
  document.getElementById("btn-export-csv").addEventListener("click", exportSingleCsv);
  document.getElementById("btn-export-json").addEventListener("click", exportSingleJson);
  document.getElementById("btn-print-report").addEventListener("click", () => window.print());
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
    chip.textContent = `Drug · ${matchedDrug.display_name}`;
    return;
  }
  const cleanSeq = s.replace(/\s+/g, "").toUpperCase();
  if (/^[ACDEFGHIKLMNPQRSTVWY]{4,1022}$/.test(cleanSeq) && !/[a-z0-9=()[\]#@\\/+.\-]/.test(s)) {
    chip.textContent = `Sequence · ${cleanSeq.length} AA`;
    return;
  }
  if (/[=()[\]@#1-9a-z]/.test(s)) {
    chip.textContent = `SMILES · ${s.length} chars`;
    return;
  }
  chip.textContent = "Query";
}

function renderAutocomplete(raw) {
  const menu = document.getElementById("autocomplete-list");
  if (!raw || raw.length < 2 || raw.length > 36 || /[=()[\]@#]/.test(raw)) {
    menu.classList.add("hidden");
    return;
  }
  const q = raw.toLowerCase();
  const hits = state.drugs.filter(
    (d) =>
      d.display_name.toLowerCase().includes(q) ||
      d.zh_name.includes(raw) ||
      d.drug_class.toLowerCase().includes(q)
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
      const qVal = item.dataset.query;
      document.getElementById("smart-query-input").value = qVal;
      menu.classList.add("hidden");
      updateLiveDetection(qVal);
      runSinglePrediction();
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
    errEl.textContent = "Please enter a peptide drug name, canonical 20-AA sequence, or SMILES string.";
    errEl.classList.remove("hidden");
    return;
  }

  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = "Running…";
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
    const res = await fetch(apiUrl("/api/v1/predict"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Prediction request failed.");
    }
    state.lastPrediction = data;
    renderPredictionResult(data);
  } catch (err) {
    errEl.textContent = err.message || String(err);
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.querySelector(".btn-label").textContent = "Predict";
    btn.querySelector(".btn-spinner").classList.add("hidden");
  }
}

function renderPredictionResult(data) {
  const hero = document.getElementById("predict-hero");
  hero.classList.remove("hero-centered");
  hero.classList.add("hero-compact");
  document.getElementById("compact-actions").classList.remove("hidden");
  document.getElementById("landing-overview").classList.add("hidden");
  document.getElementById("predict-results").classList.remove("hidden");

  // 1. 2D Structure SVG
  const svgStage = document.getElementById("mol-svg-stage");
  svgStage.innerHTML = data.svg || `<div style="color:#64748B;font-size:12.5px;">Structure unavailable</div>`;
  document.getElementById("svg-meta-caption").textContent = `${data.physchem.formula} · ${data.physchem.heavy_atoms} heavy atoms`;

  const legendEl = document.getElementById("mol-svg-legend");
  if (data.svg_highlights && data.svg_highlights.length) {
    legendEl.innerHTML = data.svg_highlights
      .map(
        (h) => `
        <span class="legend-item" title="${escapeHtml(h.label_zh)} (${h.atom_count} atoms)">
          <span class="legend-swatch" style="background:${escapeHtml(h.color)}"></span>
          ${escapeHtml(h.label_en)}
        </span>`
      )
      .join("");
  } else {
    legendEl.innerHTML = `<span class="legend-item" style="color:#94A3B8">Canonical backbone rendered via RDKit</span>`;
  }

  // 2. Identity, Modification Badges & Physicochemical Strip
  const ref = data.drug_reference;
  document.getElementById("res-mol-name").textContent = ref ? `${data.query_name} · ${ref.zh_name}` : data.query_name;
  const trackTag = data.modality_track === "dual_modality" ? "DUAL TRACK" : data.resolved_type.replace("_", " ").toUpperCase();
  document.getElementById("res-type-pill").textContent = trackTag;
  document.getElementById("res-mol-subtitle").textContent = ref
    ? ref.drug_class
    : data.sequence
      ? `Canonical 20-AA Sequence (${data.sequence.length} residues)`
      : "Covalent SMILES Structure";

  document.getElementById("res-mod-badges").innerHTML = (data.modifications || [])
    .map(
      (b) => `<span class="mod-badge tone-${escapeHtml(b.tone)}" title="${escapeHtml(b.detail || b.label_zh)}">${escapeHtml(b.label_en)}</span>`
    )
    .join("");

  const fdaBox = document.getElementById("res-fda-ref");
  if (ref && (ref.fda_half_life || ref.fda_ppb)) {
    const parts = [];
    if (ref.fda_half_life) parts.push(`<strong>FDA T½:</strong> ${escapeHtml(ref.fda_half_life)}`);
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
    ["RotBonds", String(pc.rotatable_bonds)],
    ["Charge (7.4)", chargeStr],
    ["Rings", pc.macrocycle_rings.length ? `${pc.ring_count} (M${pc.macrocycle_rings[0]})` : String(pc.ring_count)],
    ["Length", pc.length_aa ? `~${pc.length_aa} AA` : `${pc.heavy_atoms} HA`],
  ];
  document.getElementById("res-physchem-grid").innerHTML = stripItems
    .map(([lbl, val]) => `<div class="pc-cell"><span class="pc-lbl">${lbl}</span><span class="pc-val">${escapeHtml(val)}</span></div>`)
    .join("");

  // 3. 6-Axis ADMET Radar Chart
  const rtLabel = data.cache_hit
    ? `${data.runtime_ms.toFixed(0)} ms · cached`
    : data.runtime_ms >= 1000
      ? `${(data.runtime_ms / 1000).toFixed(1)} s · live`
      : `${data.runtime_ms.toFixed(0)} ms`;
  document.getElementById("res-runtime-chip").textContent = rtLabel;
  renderAdmetRadar(
    document.getElementById("res-radar-stage"),
    data.radar || [],
    document.getElementById("res-radar-readout")
  );

  // 4. Long-Acting Half-Life Mechanism Banner
  const mechEl = document.getElementById("mechanism-banner");
  const anch = data.mechanism_anchor;
  if (anch && anch.detected) {
    const rngStr = anch.range_h ? `${anch.range_h[0]}–${anch.range_h[1]} h` : `${anch.value_h} h`;
    const anchorsText = (anch.anchor_drugs || []).join(" · ");
    const routeTag = anch.ml_baseline_route ? ` (${anch.ml_baseline_route})` : "";
    mechEl.innerHTML = `
      <div class="mech-left">
        <div class="mech-kicker">★ Long-Acting Fatty-Acylation Mechanism Anchor · 长效脂化半衰期机制识别</div>
        <h3 class="mech-title">${escapeHtml(anch.label_en)} <span style="font-weight:400;color:#64748B;">/ ${escapeHtml(anch.label_zh)}</span></h3>
        <p class="mech-note">${escapeHtml(anch.note)}<br/><strong>Reference Class Anchors:</strong> ${escapeHtml(anchorsText)}</p>
      </div>
      <div class="mech-right">
        <div class="mech-metric">
          <span class="mech-metric-val">${escapeHtml(rngStr)}</span>
          <span class="mech-metric-sub">Anchored Clinical Window (Pt: ${anch.value_h} h)</span>
        </div>
        <div class="mech-ml-pill" title="${escapeHtml(anch.ml_baseline_note || "")}">
          <span>V33 Pure-ML${escapeHtml(routeTag)}</span>
          <strong>${anch.ml_baseline_h != null ? anch.ml_baseline_h.toFixed(1) + " h" : "—"}</strong>
        </div>
      </div>`;
    mechEl.classList.remove("hidden");
  } else {
    mechEl.classList.add("hidden");
  }

  // 5. Tier 1 & Tier 2 Endpoint Dashboards
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

    const rowsHtml = items.length
      ? items.map((p) => buildEndpointItemHtml(p)).join("")
      : `<div style="padding:18px;text-align:center;color:#94A3B8;font-size:12.5px;">No endpoints match current filter.</div>`;

    card.innerHTML = `
      <div class="domain-header">
        <div class="domain-title-wrap">
          <span class="domain-dot ${meta.dot}"></span>
          <h3 class="domain-title">${meta.title} <span class="domain-zh">${meta.zh}</span></h3>
        </div>
        <span class="domain-count">${items.length} endpoints</span>
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

  // Left card: Tier-2 Antimicrobial, Bactericidal, Antifungal & Anti-Biofilm endpoints
  const meta = DOMAIN_META.antimicrobial;
  const micCard = document.createElement("div");
  micCard.className = "domain-card";
  const subFilterHtml = hasMultipleGroups
    ? `<div class="tier2-subfilters">
         <button type="button" class="filter-chip ${state.tier2Group === "all" ? "active" : ""}" data-t2g="all">All (${tier2Preds.length})</button>
         ${nMic ? `<button type="button" class="filter-chip ${state.tier2Group === "mic" ? "active" : ""}" data-t2g="mic">MIC · ESKAPE (${nMic})</button>` : ""}
         ${nMbc ? `<button type="button" class="filter-chip ${state.tier2Group === "mbc" ? "active" : ""}" data-t2g="mbc">MBC &amp; Lethal (${nMbc})</button>` : ""}
         ${nBio ? `<button type="button" class="filter-chip ${state.tier2Group === "biofilm" ? "active" : ""}" data-t2g="biofilm">Fungicidal &amp; Biofilm (${nBio})</button>` : ""}
       </div>`
    : "";
  const rowsHtml = visibleT2.length
    ? visibleT2.map((p) => buildEndpointItemHtml(p)).join("")
    : `<div style="padding:20px;text-align:center;color:#94A3B8;font-size:12.5px;">No Tier-2 antimicrobial endpoints enabled or matching current filter.</div>`;
  micCard.innerHTML = `
    <div class="domain-header">
      <div class="domain-title-wrap">
        <span class="domain-dot ${meta.dot}"></span>
        <h3 class="domain-title">${meta.title} <span class="domain-zh">${meta.zh}</span></h3>
      </div>
      <span class="domain-count">${visibleT2.length} endpoints</span>
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

  // Right card: Antimicrobial Potency & Hemolytic Selectivity Index (TI = HC50 / MIC)
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
        <h3 class="domain-title">Selectivity &amp; Strain Benchmark <span class="domain-zh">抗菌选择性指数</span></h3>
      </div>
      <span class="domain-count">Tier-2 Reference</span>
    </div>
    <div class="tier2-guide-body">
      <div class="ti-kpi-row">
        <div class="ti-kpi">
          <span class="dm-lbl">Best Predicted Strain MIC</span>
          <span class="ti-val">${bestMic ? `${escapeHtml(bestMic.value_display)} µM` : "—"}</span>
          <span class="ti-sub">${bestMic ? escapeHtml(bestMic.display_en) : "Enable Tier-2"}</span>
        </div>
        <div class="ti-kpi">
          <span class="dm-lbl">Hemolysis HC50 Margin</span>
          <span class="ti-val">${hc50Ep ? `${escapeHtml(hc50Ep.value_display)} µM` : "—"}</span>
          <span class="ti-sub">Binary Risk: ${hemClsEp ? escapeHtml(hemClsEp.value_display) : "—"}</span>
        </div>
        <div class="ti-kpi">
          <span class="dm-lbl">Est. Selectivity (HC50 / MIC)</span>
          <span class="ti-val" style="color:${tiVal != null && tiVal >= 10 ? "var(--emerald)" : "var(--amber)"}">${tiVal != null ? `${tiVal >= 100 ? tiVal.toFixed(0) : tiVal.toFixed(1)}×` : "—"}</span>
          <span class="ti-sub">${tiVal != null ? (tiVal >= 10 ? "Favorable Window (≥10×)" : "Narrow Margin (<10×)") : "N/A"}</span>
        </div>
      </div>
      <p class="drawer-note" style="margin-top:12px;">
        <strong>Reference-Strain Harmonization:</strong> Tier 2 separates species-pure ESKAPE &amp; clinical reference strains (MIC), bactericidal killing (MBC / LC), fungicidal activity (MFC), and anti-biofilm inhibition/eradication (MBIC / MBEC) via the order-independent DBAASP multiset bridge + ChEMBL organism mapping, with censored-aware diagnostic heads for inactive right-censored records (&gt;64 / &gt;128 µM). Click any row on the left to inspect its 90% CI and top-3 training analogs.
      </p>
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
  const adTxt = AD_LABELS[p.ad_status] || p.ad_status;
  const tcTxt = p.max_tanimoto !== null && p.max_tanimoto !== undefined ? ` · ${p.max_tanimoto.toFixed(2)}` : "";
  const unitClean = p.unit === "probability" || p.unit === "fraction_bound" || p.unit === "fraction" ? "" : p.unit;

  const analogsHtml = (p.nearest_analogs || [])
    .map((a) => {
      const cBadge =
        a.censor_type && a.censor_type !== "exact"
          ? `<span class="censor-mini">${escapeHtml(a.censor_type)}</span>`
          : "";
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
  const calOrCovLabel = p.kind === "classification" ? "Holdout Brier / ECE" : "80% Conformal Cov.";
  const calOrCovVal =
    p.kind === "classification"
      ? p.brier_holdout != null && p.ece_holdout != null
        ? `${p.brier_holdout.toFixed(3)} / ${p.ece_holdout.toFixed(3)}`
        : "Platt / Iso"
      : p.conformal_coverage_80 != null
        ? `${(p.conformal_coverage_80 * 100).toFixed(1)}%`
        : "—";

  const trainDisplay = p.n_train
    ? `${p.n_train.toLocaleString()} CV / ${(p.n_holdout || 0).toLocaleString()} HO${p.n_censored ? ` · ${p.n_censored} cens.` : ""}`
    : "—";

  const censoredHtml = p.censored_diagnostic_text
    ? `<div class="censored-diag-box">
         <strong>Censored AFT/Tobit Diagnostic:</strong> <span class="mono">${escapeHtml(p.censored_diagnostic_text)}</span>
         <span class="censored-diag-note">— ${escapeHtml(p.censored_note || "")}</span>
       </div>`
    : "";

  const extraCiHtml = p.ci90_display ? ` · <strong>90% CI:</strong> <span class="mono">${escapeHtml(p.ci90_display)}</span>` : "";

  return `
    <div class="ep-item" data-endpoint="${escapeHtml(p.endpoint)}">
      <div class="ep-row" role="button" tabindex="0" aria-expanded="false">
        <div class="ep-name-col">
          <span class="ep-en">${escapeHtml(p.display_en)}</span>
          <span class="ep-zh">${escapeHtml(p.display_zh)}</span>
        </div>
        <div class="ep-val-col">
          <div class="ep-val-main">${escapeHtml(p.value_display)}<span class="ep-unit">${escapeHtml(unitClean)}</span></div>
          <span class="ep-risk risk-${escapeHtml(p.risk_level)}">
            <span class="risk-dot"></span>${escapeHtml(p.risk_label)}
          </span>
        </div>
        <div class="ep-ci-col">
          ${buildIntervalBarHtml(p)}
        </div>
        <span class="ad-pill ad-${escapeHtml(p.ad_status)}" title="Applicability Domain (Exact Morgan r=2 Tanimoto)">${adTxt}${tcTxt}</span>
        <span class="ep-chevron">▾</span>
      </div>
      <div class="ep-drawer hidden">
        <div class="drawer-meta-grid">
          <div class="dm-box"><span class="dm-lbl">Winning Champion &amp; Weights</span><span class="dm-val">${escapeHtml(p.model_family)}</span></div>
          <div class="dm-box"><span class="dm-lbl">Grouped CV / 20% Holdout</span><span class="dm-val">${cvDisplay} / ${holdDisplay}</span></div>
          <div class="dm-box"><span class="dm-lbl">${calOrCovLabel}</span><span class="dm-val">${calOrCovVal}</span></div>
          <div class="dm-box"><span class="dm-lbl">Audited Corpus</span><span class="dm-val">${trainDisplay}</span></div>
        </div>
        ${censoredHtml}
        <p class="drawer-note">${escapeHtml(p.note || "")}${extraCiHtml}${p.assay_context ? ` <strong>[${escapeHtml(p.assay_context)}]</strong>` : ""}</p>
        <div class="analogs-title">Top-3 Nearest Training Analogs (Exact Tanimoto)</div>
        <div class="analog-list">${analogsHtml || `<span style="font-size:12px;color:#94A3B8">No analogs indexed</span>`}</div>
      </div>
    </div>`;
}

// ==================== SINGLE EXPORT HELPERS ====================
function exportSingleCsv() {
  const d = state.lastPrediction;
  if (!d) return;
  const cols = [
    "query", "endpoint", "display_en", "display_zh", "tier", "domain", "value", "value_display", "unit",
    "ci80_low", "ci80_high", "ad_status", "max_tanimoto", "risk_label", "winning_family", "cv_score", "holdout_score", "censored_diagnostic_text",
  ];
  const lines = [cols.join(",")];
  (d.predictions || []).forEach((p) => {
    const row = [
      d.query_name, p.endpoint, p.display_en, p.display_zh, p.tier, p.domain,
      p.value ?? "", p.value_display, p.unit, p.ci80_low ?? "", p.ci80_high ?? "",
      p.ad_status, p.max_tanimoto ?? "", p.risk_label, p.winning_family ?? "",
      p.cv_score ?? "", p.holdout_score ?? "", p.censored_diagnostic_text ?? "",
    ].map((x) => `"${String(x).replace(/"/g, '""')}"`);
    lines.push(row.join(","));
  });
  downloadBlob(lines.join("\n"), `${slugify(d.query_name)}_omnipanel_v33.csv`, "text/csv;charset=utf-8");
}

function exportSingleJson() {
  const d = state.lastPrediction;
  if (!d) return;
  const copy = { ...d };
  delete copy.svg;
  downloadBlob(JSON.stringify(copy, null, 2), `${slugify(d.query_name)}_omnipanel_v33.json`, "application/json");
}

// ==================== SVG ZOOM MODAL ====================
function wireSvgModal() {
  const modal = document.getElementById("svg-modal");
  const openModal = () => {
    const d = state.lastPrediction;
    if (!d || !d.svg) return;
    document.getElementById("modal-mol-title").textContent = `${d.query_name} — 2D Molecular Structure`;
    document.getElementById("modal-mol-sub").textContent = `${d.physchem.formula} · MW ${d.physchem.mw.toFixed(1)} Da · ${d.physchem.heavy_atoms} heavy atoms`;
    document.getElementById("modal-svg-stage").innerHTML = d.svg;
    document.getElementById("modal-svg-legend").innerHTML = document.getElementById("mol-svg-legend").innerHTML;
    modal.classList.remove("hidden");
  };
  const closeModal = () => modal.classList.add("hidden");

  document.getElementById("btn-zoom-svg").addEventListener("click", openModal);
  document.getElementById("mol-svg-stage").addEventListener("click", openModal);
  document.getElementById("modal-close").addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  const dlSvg = () => {
    const d = state.lastPrediction;
    if (!d || !d.svg) return;
    downloadBlob(d.svg, `${slugify(d.query_name)}_2d_structure.svg`, "image/svg+xml;charset=utf-8");
  };
  document.getElementById("btn-download-svg").addEventListener("click", dlSvg);
  document.getElementById("modal-download-svg").addEventListener("click", dlSvg);
}

// ==================== BATCH WORKSPACE ====================
function wireBatchWorkspace() {
  const dropzone = document.getElementById("batch-dropzone");
  const fileInput = document.getElementById("batch-file-input");
  const textarea = document.getElementById("batch-paste-area");

  document.getElementById("btn-browse-file").addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) await handleBatchFile(file);
  });

  fileInput.addEventListener("change", async () => {
    if (fileInput.files[0]) await handleBatchFile(fileInput.files[0]);
  });

  let parseTimer = null;
  textarea.addEventListener("input", () => {
    clearTimeout(parseTimer);
    parseTimer = setTimeout(() => previewBatchText(textarea.value), 320);
  });

  const loadBatchPreset = (key, fallbackCsv, fallbackLabel) => {
    const preset = state.batchPresets?.[key];
    textarea.value = preset?.csv || fallbackCsv;
    document.getElementById("batch-file-label").textContent = preset?.description || fallbackLabel;
    previewBatchText(textarea.value, `${key}.csv`);
  };

  document.getElementById("btn-batch-demo").addEventListener("click", () => {
    loadBatchPreset(
      "demo_7",
      state.batchDemoCsv ||
        "name,query\nSemaglutide,Semaglutide\nTirzepatide,Tirzepatide\nCyclosporine A,Cyclosporine A\nCaspofungin,Caspofungin\nDaptomycin,Daptomycin\nClassic AMP,HISEAVKIFGCKWKENWKWNSNC\nLiraglutide,Liraglutide",
      "Loaded 7-peptide clinical, echinocandin & AMP benchmark"
    );
  });

  document.getElementById("btn-batch-case17")?.addEventListener("click", () => {
    loadBatchPreset(
      "case17_17",
      "name,query\nSemaglutide,Semaglutide\nTirzepatide,Tirzepatide\nLiraglutide,Liraglutide\nExenatide,Exenatide\nLeuprolide,Leuprolide\nBivalirudin,Bivalirudin\nGlucagon,Glucagon\nTeriparatide,Teriparatide\nCyclosporine A,Cyclosporine A\nOxytocin,Oxytocin\nOctreotide,Octreotide\nLinaclotide,Linaclotide\nZiconotide,Ziconotide\nDaptomycin,Daptomycin\nCaspofungin,Caspofungin\nInsulin detemir,Insulin detemir\nInsulin degludec,Insulin degludec",
      "Loaded all 17 Case17 clinical peptide drugs (NIFDC / Prof. Ye benchmark)"
    );
  });

  document.getElementById("btn-batch-echino")?.addEventListener("click", () => {
    loadBatchPreset(
      "echinocandin_6",
      "name,query\nCaspofungin,Caspofungin\nMicafungin,Micafungin\nAnidulafungin,Anidulafungin\nRezafungin,Rezafungin\nCyclosporine A,Cyclosporine A\nDaptomycin,Daptomycin",
      "Loaded complete FDA Echinocandin class (4/4) + Cyclosporine A + Daptomycin"
    );
  });

  document.getElementById("btn-batch-clear").addEventListener("click", () => {
    textarea.value = "";
    fileInput.value = "";
    state.batchParsed = null;
    state.batchResults = null;
    state.batchSelectedIds.clear();
    document.getElementById("batch-file-label").textContent = "Auto-detects FASTA headers, columns, drug names, sequences & SMILES";
    document.getElementById("batch-preview-meta").textContent = "Waiting for file or pasted molecules";
    document.getElementById("batch-preview-badges").innerHTML = "";
    document.getElementById("batch-preview-tbody").innerHTML = `<tr><td colspan="5" class="empty-cell">Click “Load Clinical Benchmark (7)” or paste peptides to preview.</td></tr>`;
    document.getElementById("batch-results-section").classList.add("hidden");
  });

  document.getElementById("btn-run-batch").addEventListener("click", runBatchPrediction);
  document.getElementById("batch-result-search").addEventListener("input", renderBatchResultsTable);
  document.getElementById("btn-batch-csv").addEventListener("click", exportBatchCsv);
  document.getElementById("btn-batch-json").addEventListener("click", exportBatchJson);

  document.querySelectorAll("#batch-results-table th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (state.batchSort.col === col) {
        state.batchSort.asc = !state.batchSort.asc;
      } else {
        state.batchSort.col = col;
        state.batchSort.asc = true;
      }
      renderBatchResultsTable();
    });
  });
}

async function handleBatchFile(file) {
  const text = await file.text();
  document.getElementById("batch-paste-area").value = text;
  document.getElementById("batch-file-label").textContent = `Loaded file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  await previewBatchText(text, file.name);
}

async function previewBatchText(rawText, filename = null) {
  const tbody = document.getElementById("batch-preview-tbody");
  const metaEl = document.getElementById("batch-preview-meta");
  const badgesEl = document.getElementById("batch-preview-badges");

  if (!rawText.trim()) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-cell">Paste molecules or upload a file to preview.</td></tr>`;
    metaEl.textContent = "Waiting for input";
    badgesEl.innerHTML = "";
    return;
  }

  try {
    const res = await fetch(apiUrl("/api/v1/batch/parse"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_text: rawText, filename }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Parse error");
    state.batchParsed = data;

    metaEl.textContent = `Parsed ${data.total} molecules (max ${data.max_batch})`;
    badgesEl.innerHTML = `
      <span class="ad-pill ad-in_domain">${data.valid_count} Valid</span>
      ${data.invalid_count ? `<span class="ad-pill ad-ood">${data.invalid_count} Invalid</span>` : ""}`;

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
            <td>${r.valid ? `<span class="ad-pill ad-in_domain">OK</span>` : `<span class="ad-pill ad-ood" title="${escapeHtml(r.error || "")}">Invalid</span>`}</td>
          </tr>`;
      })
      .join("");
  } catch (err) {
    metaEl.textContent = err.message;
  }
}

async function runBatchPrediction() {
  const rawText = document.getElementById("batch-paste-area").value.trim();
  const errEl = document.getElementById("batch-error");
  errEl.classList.add("hidden");

  if (!rawText) {
    errEl.textContent = "Please upload a file, paste molecules, or click 'Load Clinical Benchmark (7)'.";
    errEl.classList.remove("hidden");
    return;
  }

  const progWrap = document.getElementById("batch-progress-wrap");
  const progFill = document.getElementById("batch-progress-fill");
  const progLabel = document.getElementById("batch-progress-label");
  const btn = document.getElementById("btn-run-batch");

  progWrap.classList.remove("hidden");
  progFill.style.width = "18%";
  progLabel.textContent = "Encoding batch & running V33 4-FM super-ensemble…";
  btn.disabled = true;

  let fakePct = 18;
  const timer = setInterval(() => {
    fakePct = Math.min(92, fakePct + 6);
    progFill.style.width = `${fakePct}%`;
  }, 450);

  try {
    const res = await fetch(apiUrl("/api/v1/batch"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        raw_text: rawText,
        include_tier2: document.getElementById("batch-tier2").checked,
        conditions: {
          rbc_species: document.getElementById("batch-rbc").value,
          pk_context: "human_plasma",
        },
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Batch prediction failed.");

    clearInterval(timer);
    progFill.style.width = "100%";
    const rtText = data.runtime_ms >= 1000 ? `${(data.runtime_ms / 1000).toFixed(1)} s` : `${data.runtime_ms.toFixed(0)} ms`;
    progLabel.textContent = `Completed ${data.succeeded}/${data.total} molecules in ${rtText} (${data.adapter_mode}).`;

    state.batchResults = data;
    state.batchSelectedIds.clear();
    data.results
      .filter((r) => r.status === "ok")
      .slice(0, 3)
      .forEach((r) => state.batchSelectedIds.add(r.index));

    renderBatchResults();
  } catch (err) {
    clearInterval(timer);
    progWrap.classList.add("hidden");
    errEl.textContent = err.message || String(err);
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
}

function renderBatchResults() {
  const data = state.batchResults;
  if (!data) return;
  document.getElementById("batch-results-section").classList.remove("hidden");

  const anchoredCount = data.results.filter((r) => r.mechanism_anchor && r.mechanism_anchor.detected).length;
  const rtText = data.runtime_ms >= 1000 ? `${(data.runtime_ms / 1000).toFixed(1)} s` : `${data.runtime_ms.toFixed(0)} ms`;
  document.getElementById("batch-kpis").innerHTML = `
    <div class="bkpi"><span>Total</span><strong>${data.total}</strong></div>
    <div class="bkpi"><span>Succeeded</span><strong style="color:var(--emerald)">${data.succeeded}</strong></div>
    <div class="bkpi"><span>Acylation Anchors</span><strong style="color:var(--amber)">${anchoredCount}</strong></div>
    <div class="bkpi"><span>Runtime</span><strong>${rtText}</strong></div>`;

  renderBatchResultsTable();
  updateBatchRadar();
}

function renderBatchResultsTable() {
  const data = state.batchResults;
  if (!data) return;
  const search = document.getElementById("batch-result-search").value.trim().toLowerCase();
  let rows = [...data.results];

  if (search) {
    rows = rows.filter((r) => r.name.toLowerCase().includes(search) || (r.modifications || []).join(" ").toLowerCase().includes(search));
  }

  if (state.batchSort.col) {
    const col = state.batchSort.col;
    const dir = state.batchSort.asc ? 1 : -1;
    rows.sort((a, b) => {
      const getVal = (r) => {
        if (col === "name") return r.name || "";
        if (col === "mw") return r.mw ?? -Infinity;
        if (col === "logd") return parseFloat(r.key_metrics?.logd) || -Infinity;
        if (col === "hemolysis") return r.key_metrics?.hemolysis_risk_val ?? (parseFloat(r.key_metrics?.hemolysis_risk) || -Infinity);
        return r.key_metrics?.[col] ?? -Infinity;
      };
      const va = getVal(a);
      const vb = getVal(b);
      return typeof va === "string" ? va.localeCompare(vb) * dir : (va - vb) * dir;
    });
  }

  const tbody = document.getElementById("batch-results-tbody");
  tbody.innerHTML = rows
    .map((r) => {
      if (r.status !== "ok") {
        return `<tr><td></td><td><strong>${escapeHtml(r.name)}</strong></td><td colspan="8" style="color:var(--rose)">${escapeHtml(r.error)}</td></tr>`;
      }
      const km = r.key_metrics || {};
      const checked = state.batchSelectedIds.has(r.index) ? "checked" : "";
      const modChip = (r.modifications || [])[0] || "";
      return `
        <tr>
          <td><input type="checkbox" class="batch-plot-chk" data-idx="${r.index}" ${checked} aria-label="Plot ${escapeHtml(r.name)}" /></td>
          <td>
            <strong class="mol-cell-name">${escapeHtml(r.name)}</strong>
            ${modChip ? `<div class="mol-cell-mod">${escapeHtml(modChip)}</div>` : ""}
          </td>
          <td class="num">${r.mw ? r.mw.toFixed(0) : "—"}</td>
          <td class="num" style="${r.mechanism_anchor?.detected ? "color:#B45309;font-weight:700" : ""}">${escapeHtml(km.half_life || "—")}</td>
          <td class="num">${escapeHtml(km.ppb || "—")}</td>
          <td class="num">${escapeHtml(km.pampa || "—")}</td>
          <td class="num">${escapeHtml(km.caco2 || "—")}</td>
          <td class="nowrap"><span class="ep-risk risk-${escapeHtml(km.hemolysis_risk_level || "info")}"><span class="risk-dot"></span>${escapeHtml(km.hemolysis_risk || "—")}</span></td>
          <td class="num">${escapeHtml(km.logd || "—")}</td>
          <td class="nowrap"><button type="button" class="btn-inspect" data-inspect-idx="${r.index}">Inspect →</button></td>
        </tr>`;
    })
    .join("");

  tbody.querySelectorAll(".batch-plot-chk").forEach((chk) => {
    chk.addEventListener("change", () => {
      const idx = Number(chk.dataset.idx);
      if (chk.checked) {
        if (state.batchSelectedIds.size >= 4) {
          const oldest = state.batchSelectedIds.values().next().value;
          state.batchSelectedIds.delete(oldest);
        }
        state.batchSelectedIds.add(idx);
      } else {
        state.batchSelectedIds.delete(idx);
      }
      renderBatchResultsTable();
      updateBatchRadar();
    });
  });

  tbody.querySelectorAll(".btn-inspect").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.inspectIdx);
      const item = data.results.find((r) => r.index === idx);
      if (!item) return;
      switchTab("predict");
      const queryVal = item.resolved_type === "drug_name" ? item.name : item.smiles || item.sequence || item.name;
      document.getElementById("smart-query-input").value = queryVal;
      document.getElementById("adv-sequence").value = "";
      document.getElementById("adv-smiles").value = "";
      updateLiveDetection(queryVal);
      runSinglePrediction();
    });
  });
}

function updateBatchRadar() {
  if (!state.batchResults) return;
  const selected = state.batchResults.results.filter((r) => r.status === "ok" && state.batchSelectedIds.has(r.index));
  renderBatchRadarOverlay(
    document.getElementById("batch-radar-stage"),
    document.getElementById("batch-radar-legend"),
    selected
  );
}

function exportBatchCsv() {
  if (!state.batchResults) return;
  const cols = ["index", "name", "status", "mw", "logp", "net_charge_ph74", "half_life", "ppb", "pampa", "caco2", "hemolysis_risk", "hc50", "logd", "solubility"];
  const lines = [cols.join(",")];
  state.batchResults.results.forEach((r) => {
    const km = r.key_metrics || {};
    const row = [
      r.index, r.name, r.status, r.mw ?? "", r.logp ?? "", r.net_charge_ph74 ?? "",
      km.half_life ?? "", km.ppb ?? "", km.pampa ?? "", km.caco2 ?? "",
      km.hemolysis_risk ?? "", km.hc50 ?? "", km.logd ?? "", km.solubility ?? "",
    ].map((x) => `"${String(x).replace(/"/g, '""')}"`);
    lines.push(row.join(","));
  });
  downloadBlob(lines.join("\n"), "omnipanel_v33_batch_results.csv", "text/csv;charset=utf-8");
}

function exportBatchJson() {
  if (!state.batchResults) return;
  downloadBlob(JSON.stringify(state.batchResults, null, 2), "omnipanel_v33_batch_results.json", "application/json");
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
  document.getElementById("val-domain-filter").addEventListener("change", (e) => {
    state.valDomain = e.target.value;
    renderValidationTable();
  });
  document.getElementById("val-search").addEventListener("input", (e) => {
    state.valSearch = e.target.value.trim().toLowerCase();
    renderValidationTable();
  });
}

async function loadEndpointsCatalog() {
  try {
    const res = await fetch(apiUrl("/api/v1/endpoints"));
    const data = await res.json();
    state.catalogData = data;
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
  const meanAuc =
    avg(clsEps.map((e) => e.holdout_score_v33).filter((v) => v != null)) ?? clsM.v33_champion_roc_auc;
  const meanRho =
    avg(regEps.map((e) => e.holdout_score_v33).filter((v) => v != null)) ?? regM.v33_champion_spearman;
  const meanCov80 = avg(regEps.map((e) => e.conformal_coverage_80).filter((v) => v != null));

  const bAll = document.getElementById("val-btn-all");
  const bT1 = document.getElementById("val-btn-t1");
  const bT2 = document.getElementById("val-btn-t2");
  if (bAll) bAll.textContent = `All Endpoints (${eps.length})`;
  if (bT1) bT1.textContent = `Tier 1 · Core ADMET (${nT1})`;
  if (bT2) bT2.textContent = `Tier 2 · Antimicrobial & Biofilm (${nT2})`;

  if (data.manifest_generated_at) {
    const sub = document.getElementById("val-manifest-subtitle");
    if (sub) {
      sub.innerHTML = `100% audited metrics hydrated live from <code>v33_manifest.json</code> (built ${escapeHtml(data.manifest_generated_at)}) — zero-holdout-peeking grouped CV &amp; untouched 20% holdout.`;
    }
  }

  document.getElementById("val-kpi-grid").innerHTML = `
    <div class="vkpi">
      <span class="vkpi-lbl">Curated Peptides</span>
      <span class="vkpi-val">${Number(nComp).toLocaleString()}</span>
      <span class="vkpi-sub">Unified peptide-only corpus</span>
    </div>
    <div class="vkpi">
      <span class="vkpi-lbl">Unified V33 Endpoints</span>
      <span class="vkpi-val">${eps.length}</span>
      <span class="vkpi-sub">Tier 1: ${nT1} · Tier 2: ${nT2}</span>
    </div>
    <div class="vkpi">
      <span class="vkpi-lbl">Harmonized Records</span>
      <span class="vkpi-val">${Number(nRows).toLocaleString()}</span>
      <span class="vkpi-sub">${Number(nCens).toLocaleString()} censored-aware labels</span>
    </div>
    <div class="vkpi">
      <span class="vkpi-lbl">Holdout ROC-AUC (${clsEps.length} Cls)</span>
      <span class="vkpi-val" style="color:var(--cobalt)">${meanAuc != null ? Number(meanAuc).toFixed(3) : "—"}</span>
      <span class="vkpi-sub">Mean Brier ${clsM.v33_champion_brier != null ? Number(clsM.v33_champion_brier).toFixed(3) : "—"} · ECE ${clsM.v33_champion_ece != null ? Number(clsM.v33_champion_ece).toFixed(3) : "—"}</span>
    </div>
    <div class="vkpi">
      <span class="vkpi-lbl">Holdout Spearman (${regEps.length} Reg)</span>
      <span class="vkpi-val" style="color:var(--emerald)">${meanRho != null ? Number(meanRho).toFixed(3) : "—"}</span>
      <span class="vkpi-sub">Mean 80% Conformal Cov: ${meanCov80 != null ? (Number(meanCov80) * 100).toFixed(1) + "%" : "—"}</span>
    </div>`;
}

function renderValidationHighlights(data) {
  // 1. Strict Leave-Case17-Out Clinical Benchmark Card
  const c17 = data.case17_scorecard || {};
  const hlPure = c17.half_life_pure_ml_route_matched_log10 || {};
  const hlAnch = c17.half_life_with_mechanism_anchor_log10 || {};
  const ppb = c17.ppb_dual || {};
  const hem = c17.hemolysis_false_positive_check || {};

  document.getElementById("val-case17-card").innerHTML = `
    <div class="card-head" style="margin-bottom:10px;">
      <div>
        <h2 class="card-title">Strict Leave-Case17-Out Clinical Benchmark</h2>
        <span class="card-sub">Zero SMILES / InChIKey-14 / scaffold cluster overlap across 17 FDA peptide drugs</span>
      </div>
      <span class="ad-pill ad-in_domain">0 / 17 Hemolysis FP</span>
    </div>
    <div class="c17-metrics-grid">
      <div class="c17-stat">
        <span class="dm-lbl">Pure-ML T½ (Route-Matched)</span>
        <span class="c17-val">ρ = ${hlPure.spearman != null ? Number(hlPure.spearman).toFixed(3) : "—"}</span>
        <span class="c17-sub">R² = ${hlPure.r2 != null ? Number(hlPure.r2).toFixed(3) : "—"} · MAE ${hlPure.mae != null ? Number(hlPure.mae).toFixed(3) : "—"} log₁₀ h</span>
      </div>
      <div class="c17-stat">
        <span class="dm-lbl">T½ + Acylation Anchor</span>
        <span class="c17-val" style="color:var(--emerald)">MAE ${hlAnch.mae != null ? Number(hlAnch.mae).toFixed(3) : "—"}</span>
        <span class="c17-sub">R² = ${hlAnch.r2 != null ? Number(hlAnch.r2).toFixed(3) : "—"} · ρ = ${hlAnch.spearman != null ? Number(hlAnch.spearman).toFixed(3) : "—"} (n=${hlAnch.n ?? 16})</span>
      </div>
      <div class="c17-stat">
        <span class="dm-lbl">Clinical PPB (Dual Track)</span>
        <span class="c17-val">ρ = ${ppb.spearman != null ? Number(ppb.spearman).toFixed(3) : "—"}</span>
        <span class="c17-sub">Pearson r = ${ppb.pearson != null ? Number(ppb.pearson).toFixed(3) : "—"} (n=${ppb.n ?? 14})</span>
      </div>
      <div class="c17-stat">
        <span class="dm-lbl">FDA Hemolysis Safety Check</span>
        <span class="c17-val" style="color:var(--emerald)">${hem.n_drugs_gt_0_50 ?? 0} / 17 FP</span>
        <span class="c17-sub">Max P = ${hem.max_prob_across_17_fda_drugs != null ? (Number(hem.max_prob_across_17_fda_drugs) * 100).toFixed(1) + "%" : "—"} (vs V32: ${hem.v32_n_drugs_gt_0_60 ?? 8}/17 &gt;60%)</span>
      </div>
    </div>`;

  // 2. 4-Foundation-Model Representation Dominance Strip
  const rep = data.representation_ablation_summary || {};
  const wins =
    rep.single_foundation_model_holdout_wins ||
    rep.single_foundation_model_holdout_wins_out_of_24 ||
    {};
  const famCounts = (data.overall_summary || {}).dual_modality_winning_family_counts || {};
  const totalEp = (data.endpoints || []).length || 24;
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
          <div class="fm-bar-track">
            <div class="fm-bar-fill" style="width:${Math.max(8, pct)}%;background:${meta.color};"></div>
          </div>
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
        <h2 class="card-title">4-Foundation-Model Representation Dominance</h2>
        <span class="card-sub">Single-FM untouched holdout wins (${fmDenom} benchmarked) + champion family split (${totalEp} active EPs)</span>
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
      const dm = DOMAIN_META[e.domain] || { title: e.domain, dot: "dot-physicochemical" };
      const famTone =
        e.winning_family === "super_ensemble"
          ? "tone-cobalt"
          : e.winning_family === "fusion"
            ? "tone-teal"
            : e.winning_family === "deep"
              ? "tone-violet"
              : "tone-slate";

      const v32Str =
        e.holdout_score_v32 != null
          ? `<span title="V32 Holdout (${e.cv_metric_name}); CV ${e.cv_score_v32 != null ? e.cv_score_v32.toFixed(3) : "—"}">${e.holdout_score_v32.toFixed(3)}</span>`
          : e.cv_score_v32 != null
            ? `<span title="V32 CV (${e.cv_metric_name})">${e.cv_score_v32.toFixed(3)}*</span>`
            : "—";

      let covOrCal = "—";
      if (e.kind === "classification") {
        if (e.brier_holdout != null && e.ece_holdout != null) {
          covOrCal = `<span title="Holdout Brier / Expected Calibration Error">B ${e.brier_holdout.toFixed(3)} · E ${e.ece_holdout.toFixed(3)}</span>`;
        }
      } else if (e.conformal_coverage_80 != null) {
        const covPct = (e.conformal_coverage_80 * 100).toFixed(1) + "%";
        covOrCal = `<span title="Untouched Holdout 80% Conformal Coverage (90%: ${e.conformal_coverage_90 != null ? (e.conformal_coverage_90 * 100).toFixed(1) + "%" : "—"})">${covPct}</span>`;
      }

      return `
        <tr>
          <td>
            <strong>${escapeHtml(e.display_en)}</strong>
            <div style="font-size:11.2px;color:#64748B">${escapeHtml(e.display_zh)} · <code style="font-size:10.5px">${escapeHtml(e.endpoint)}</code></div>
          </td>
          <td class="nowrap">
            <span class="mod-badge ${e.tier === 1 ? "tone-cobalt" : "tone-amber"}">Tier ${e.tier}</span>
            <div style="font-size:11px;color:#64748B;margin-top:2px">${escapeHtml(dm.title.split(" ")[0])}</div>
          </td>
          <td class="nowrap">
            <span class="mono" style="font-size:11.5px">${escapeHtml(e.kind === "classification" ? "cls · AUC" : "reg · ρ")}</span>
            <div style="font-size:11px;color:#64748B">${escapeHtml(e.unit)}</div>
          </td>
          <td class="num">${e.n_train.toLocaleString()}</td>
          <td class="num">${(e.n_holdout || 0).toLocaleString()}</td>
          <td class="num">${e.n_censored ? e.n_censored.toLocaleString() : "0"}</td>
          <td>
            <span class="mod-badge ${famTone}" style="font-size:11px;">${escapeHtml(e.winning_family)}</span>
            <div style="font-size:11px;color:#64748B;margin-top:2px;">${escapeHtml(e.model_family)}</div>
          </td>
          <td class="num">${v32Str}</td>
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
    body: { query: "Semaglutide", include_tier2: false, include_svg: false, conditions: { rbc_species: "human", pk_context: "human_sc" } },
  },
  predict_amp: {
    method: "POST",
    url: "/api/v1/predict",
    body: { query: "HISEAVKIFGCKWKENWKWNSNC", include_tier2: true, include_svg: false },
  },
  render_csa: {
    method: "POST",
    url: "/api/v1/render-svg",
    body: { query: "Cyclosporine A", width: 560, height: 340 },
  },
  health: { method: "GET", url: "/api/v1/health", body: null },
  endpoints: { method: "GET", url: "/api/v1/endpoints", body: null },
};

function wireApiWorkspace() {
  renderCodeSnippet("curl");
  document.querySelectorAll(".code-tab").forEach((t) => {
    t.addEventListener("click", () => {
      state.codeLang = t.dataset.lang;
      document.querySelectorAll(".code-tab").forEach((x) => x.classList.toggle("active", x.dataset.lang === state.codeLang));
      renderCodeSnippet(state.codeLang);
    });
  });

  document.getElementById("btn-copy-snippet").addEventListener("click", async () => {
    const txt = document.getElementById("api-code-snippet").textContent;
    await navigator.clipboard?.writeText(txt);
    const btn = document.getElementById("btn-copy-snippet");
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = "Copy"), 1400);
  });

  const sel = document.getElementById("api-preset-select");
  const bodyEl = document.getElementById("api-play-body");
  const syncPreset = () => {
    const p = API_PRESETS[sel.value];
    bodyEl.value = p.body ? JSON.stringify(p.body, null, 2) : "// GET request — no JSON body required";
    bodyEl.disabled = !p.body;
  };
  sel.addEventListener("change", syncPreset);
  syncPreset();

  document.getElementById("btn-api-try").addEventListener("click", async () => {
    const p = API_PRESETS[sel.value];
    const statusEl = document.getElementById("api-play-status");
    const outEl = document.getElementById("api-play-output");
    statusEl.textContent = "Sending…";
    const t0 = performance.now();
    try {
      const opts = { method: p.method, headers: {} };
      if (p.method === "POST") {
        opts.headers["Content-Type"] = "application/json";
        opts.body = bodyEl.value;
      }
      const res = await fetch(apiUrl(p.url), opts);
      const data = await res.json();
      const dt = (performance.now() - t0).toFixed(0);
      statusEl.textContent = `HTTP ${res.status} · ${dt} ms`;
      if (data.svg && data.svg.length > 200) {
        data.svg = data.svg.slice(0, 140) + `… [${data.svg.length} chars SVG truncated in preview]`;
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
  if (lang === "curl") {
    el.textContent = `curl -sS -X POST "${baseFull}/api/v1/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "Semaglutide",
    "include_tier2": true,
    "conditions": {"rbc_species": "human", "pk_context": "human_sc"}
  }' | jq '.adapter_mode, .mechanism_anchor, .radar'`;
  } else {
    el.textContent = `import httpx

resp = httpx.post(
    "${baseFull}/api/v1/predict",
    json={
        "query": "Semaglutide",
        "include_tier2": True,
        "conditions": {"rbc_species": "human", "pk_context": "human_sc"},
    },
    timeout=120.0,
)
data = resp.json()
print("Engine:", data["adapter_mode"])
print("Anchor T1/2:", data["mechanism_anchor"]["range_h"])
print("PPB:", next(p["value_display"] for p in data["predictions"] if p["endpoint"] == "PPB"))`;
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

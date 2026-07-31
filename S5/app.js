const PLAN_URL = "mixture-plan.json";
const numberIN = new Intl.NumberFormat("en-IN");

// Keeps the interactive allocation visible when index.html is opened directly
// from disk, where browsers commonly block fetch() for adjacent JSON files.
const offlinePlan = {
  model: { main_pretraining_tokens: 18_000_000_000_000 },
  anneal: {
    reserve_tokens: 2_000_000_000_000,
    editable_guardrails_percent: { indic: 8, agentic: 8, safety_grounding: 1 },
  },
  lanes: [
    { id: "general", target_percent: 26, protected_floor_percent: 12, anneal_percent: 8 },
    { id: "code", target_percent: 24, protected_floor_percent: 18, anneal_percent: 22 },
    { id: "agentic", target_percent: 12, protected_floor_percent: 8, anneal_percent: 20 },
    { id: "reasoning", target_percent: 14, protected_floor_percent: 8, anneal_percent: 20 },
    { id: "indic", target_percent: 12, protected_floor_percent: 8, anneal_percent: 17 },
    { id: "long_context", target_percent: 8, protected_floor_percent: 4, anneal_percent: 10 },
    { id: "india_domain", target_percent: 3, protected_floor_percent: 2, anneal_percent: 2 },
    { id: "safety_grounding", target_percent: 1, protected_floor_percent: 1, anneal_percent: 1 },
  ],
};

const laneMeta = {
  general: {
    name: "General web",
    short: "Broad knowledge and language",
    color: "#c9c4b7",
    benchmarks: ["MMLU-Pro", "ARC-C", "HellaSwag", "Factuality held-out"],
    supplyTokens: 18_500_000_000_000,
    supplyLabel: ">18.5T FineWeb published",
    supplyNote: "Abundant before V5 licence, quality and decontamination gates.",
  },
  code: {
    name: "Code",
    short: "Repositories, tests and patches",
    color: "#75b8ff",
    benchmarks: ["LiveCodeBench", "HumanEval+", "MBPP+", "SWE-bench Verified"],
    supplyTokens: 4_300_000_000_000,
    supplyLabel: "≈4.3T prior training evidence",
    supplyNote: "The Stack v2 is large, but V5's accepted permissive and deduplicated inventory will be smaller.",
  },
  agentic: {
    name: "Agentic",
    short: "Plan, call, observe and recover",
    color: "#ff795e",
    benchmarks: ["BFCL", "τ-bench", "Tool success", "Recovery rate", "SWE-bench Verified"],
    supplyTokens: null,
    supplyLabel: "Unmeasured accepted tokens",
    supplyNote: "ToolACE has only 11,300 public rows. Repository trajectories must be replayed or generated and execution-verified.",
  },
  reasoning: {
    name: "Reasoning, mathematics & science",
    short: "STEM knowledge and verifiable depth",
    color: "#b695ff",
    benchmarks: ["GSM8K", "MATH-500", "GPQA", "BBH", "Reasoning calibration"],
    supplyTokens: 5_400_000_000_000,
    supplyLabel: "5.4T relaxed FineWeb-Edu",
    supplyNote: "Only 1.3T is in the strict tier; generated solutions still require answer verification and family deduplication.",
  },
  indic: {
    name: "Indic",
    short: "Native and code-mixed fluency",
    color: "#ffcc66",
    benchmarks: ["IndicXTREME", "IndicGenBench", "IN22", "FLORES", "Worst-language score"],
    supplyTokens: 251_321_000_000,
    supplyLabel: "251.321B Sangraha headline",
    supplyNote: "This includes verified, unverified, translated and romanised data; V5 must keep those ledgers separate.",
  },
  long_context: {
    name: "Long context",
    short: "Repositories and multi-document state",
    color: "#57bde8",
    benchmarks: ["RULER 32K", "RULER 64K", "RULER 128K", "LongBench", "Repo QA"],
    supplyTokens: 6_400_000_000,
    supplyLabel: "6.4B dedicated ProLong-64K books",
    supplyNote: "Most long-context supply must be reconstructed from complete repositories, papers, books and judgments.",
  },
  india_domain: {
    name: "India domain",
    short: "Institutions, law and daily defaults",
    color: "#f3bf55",
    benchmarks: ["IndQA", "MILU", "Jurisdiction QA", "UPI/GST tasks", "Date accuracy"],
    supplyTokens: null,
    supplyLabel: "No tokenised cross-source inventory",
    supplyNote: "India Code, Gazette, regulators, courts and education sources are identified but not yet measured after cleaning.",
  },
  safety_grounding: {
    name: "Safety & grounding",
    short: "Secure, bounded and evidence-led",
    color: "#ffb6a6",
    benchmarks: ["Indic red team", "XSTest", "HarmBench-style", "Over-refusal", "Secure code"],
    supplyTokens: null,
    supplyLabel: "Fragmented; audit pending",
    supplyNote: "Safe completions, refusals and corrections need balanced, multilingual, domain-grounded construction.",
  },
};

const presetCopy = {
  main: {
    name: "Main pretraining",
    note: "Across the 18T main phase, general web is the largest single lane while code and agentic work jointly receive 36%.",
  },
  anneal: {
    name: "Annealing",
    note: "Across the protected 2T reserve, general web falls while scarce, verified capabilities are deliberately concentrated.",
  },
  naive: {
    name: "Naive web-heavy",
    note: "Across an 18T phase, abundant web expands until Indic and agentic hit their floors—the failure mode the guardrail exposes.",
  },
  custom: {
    name: "Custom hypothesis",
    note: "One lane moved; all remaining surplus was proportionally renormalised without crossing a protected floor.",
  },
};

let plan;
let lanes;
let state = {};
let activeLane = "general";
let activePreset = "main";
let mainBudget;
let annealBudget;
let activeBudget;

function compactTokens(value) {
  if (value >= 1_000_000_000_000) return `${(value / 1_000_000_000_000).toFixed(2)}T`;
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  return numberIN.format(Math.round(value));
}

function valuesForPreset(name) {
  if (name === "main") return Object.fromEntries(lanes.map((lane) => [lane.id, lane.target_percent]));
  if (name === "anneal") return Object.fromEntries(lanes.map((lane) => [lane.id, lane.anneal_percent]));
  return {
    general: 49,
    code: 20,
    agentic: 8,
    reasoning: 8,
    indic: 8,
    long_context: 4,
    india_domain: 2,
    safety_grounding: 1,
  };
}

function floorForLane(lane) {
  if (activeBudget !== annealBudget) return lane.protected_floor_percent;
  return plan.anneal.editable_guardrails_percent[lane.id] || 0;
}

function laneMaximum(id) {
  return 100 - lanes.filter((lane) => lane.id !== id)
    .reduce((sum, lane) => sum + floorForLane(lane), 0);
}

function adjustLane(id, requested) {
  const lane = lanes.find((item) => item.id === id);
  const laneFloor = floorForLane(lane);
  const next = Math.min(laneMaximum(id), Math.max(laneFloor, requested));
  const others = lanes.filter((item) => item.id !== id);
  const otherFloor = others.reduce((sum, item) => sum + floorForLane(item), 0);
  const surplusToAllocate = Math.max(0, 100 - next - otherFloor);
  let weights = others.map((item) => Math.max(0, state[item.id] - floorForLane(item)));
  let weightTotal = weights.reduce((sum, value) => sum + value, 0);

  if (weightTotal < 1e-9) {
    weights = others.map((item) => Math.max(0, item.target_percent - floorForLane(item)));
    weightTotal = weights.reduce((sum, value) => sum + value, 0);
  }
  if (weightTotal < 1e-9) {
    weights = others.map(() => 1);
    weightTotal = others.length;
  }

  state[id] = next;
  others.forEach((item, index) => {
    state[item.id] = floorForLane(item) + surplusToAllocate * weights[index] / weightTotal;
  });
  activeLane = id;
  activePreset = "custom";
  render();
}

function buildControls() {
  const controls = document.querySelector("#lane-controls");
  lanes.forEach((lane) => {
    const meta = laneMeta[lane.id];
    const row = document.createElement("div");
    row.className = "lane-row";
    row.dataset.lane = lane.id;
    row.style.setProperty("--lane", meta.color);
    row.innerHTML = `
      <div class="lane-name"><i class="lane-dot"></i><div><strong>${meta.name}</strong><span>${meta.short}</span></div></div>
      <div class="range-wrap">
        <input type="range" min="${lane.protected_floor_percent}" max="${laneMaximum(lane.id)}" step="0.1" aria-label="${meta.name} allocation">
        <div class="range-labels"><span>floor ${lane.protected_floor_percent}%</span><span>max ${laneMaximum(lane.id)}%</span></div>
      </div>
      <div class="lane-value"><strong>0.0%</strong><span>0 tokens</span></div>`;
    row.addEventListener("click", () => { activeLane = lane.id; render(); });
    row.querySelector("input").addEventListener("input", (event) => {
      event.stopPropagation();
      adjustLane(lane.id, Number(event.target.value));
    });
    controls.append(row);
  });
}

function renderControls() {
  document.querySelectorAll(".lane-row").forEach((row) => {
    const id = row.dataset.lane;
    const lane = lanes.find((item) => item.id === id);
    const floor = floorForLane(lane);
    const value = state[id];
    row.classList.toggle("active", id === activeLane);
    const input = row.querySelector("input");
    input.min = floor;
    input.max = laneMaximum(id);
    input.value = value;
    row.querySelector(".range-labels").innerHTML = `<span>floor ${floor}%</span><span>max ${laneMaximum(id)}%</span>`;
    row.querySelector(".lane-value strong").textContent = `${value.toFixed(1)}%`;
    row.querySelector(".lane-value span").textContent = compactTokens(activeBudget * value / 100);
  });
}

function renderDonut() {
  let cursor = 0;
  const stops = [];
  lanes.forEach((lane) => {
    const start = cursor;
    cursor += state[lane.id];
    stops.push(`${laneMeta[lane.id].color} ${start}% ${cursor}%`);
  });
  document.querySelector("#mixture-donut").style.background = `conic-gradient(${stops.join(",")})`;
  const total = Object.values(state).reduce((sum, value) => sum + value, 0);
  document.querySelector("#total-share").textContent = `${total.toFixed(1)}%`;
  document.querySelector("#priority-share").textContent = `${(state.code + state.agentic).toFixed(1)}%`;
  document.querySelector("#extended-priority-share").textContent = `${(state.code + state.agentic + state.reasoning + state.long_context).toFixed(1)}%`;
  document.querySelector("#preset-name").textContent = presetCopy[activePreset].name;
  document.querySelector("#preset-note").textContent = presetCopy[activePreset].note;
  document.querySelector("#phase-budget").textContent = compactTokens(activeBudget);
  const floors = lanes.reduce((sum, lane) => sum + floorForLane(lane), 0);
  document.querySelector("#floor-note").textContent = activeBudget === annealBudget
    ? `Anneal guardrails: ${floors}% protected (Indic 8% · agentic 8% · safety 1%).`
    : `Main selector floor: ${floors}% of every window is non-negotiable.`;
}

function renderBenchmarks() {
  const meta = laneMeta[activeLane];
  document.querySelector("#active-lane-name").textContent = meta.name;
  document.querySelector("#benchmark-list").innerHTML = meta.benchmarks.map((item) => `<span>${item}</span>`).join("");
}

function renderSupply() {
  const grid = document.querySelector("#supply-grid");
  grid.innerHTML = lanes.map((lane) => {
    const meta = laneMeta[lane.id];
    const required = activeBudget * state[lane.id] / 100;
    const measured = meta.supplyTokens !== null;
    const ratio = measured ? meta.supplyTokens / required : 0;
    const coverage = Math.min(100, ratio * 100);
    const status = !measured ? "Unmeasured" : ratio >= 1 ? "Published scale covers target" : `${(ratio * 100).toFixed(ratio < .01 ? 2 : 1)}% coverage`;
    const gap = !measured ? "Unknown" : ratio >= 1 ? "No headline gap" : compactTokens(required - meta.supplyTokens);
    return `<article class="supply-card" style="--lane:${meta.color}">
      <div class="supply-top"><h3>${meta.name}</h3><strong>${status}</strong></div>
      <div class="supply-numbers">
        <div><span>Required exposure</span><strong title="${numberIN.format(Math.round(required))} tokens">${compactTokens(required)}</strong></div>
        <div><span>Published / measured</span><strong>${meta.supplyLabel}</strong></div>
      </div>
      <div class="coverage-track${measured ? "" : " unmeasured"}"><i style="--coverage:${coverage}%"></i></div>
      <p><strong>Visible gap: ${gap}.</strong> ${meta.supplyNote}</p>
    </article>`;
  }).join("");
}

function renderPresets() {
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === activePreset);
  });
}

function render() {
  renderControls();
  renderDonut();
  renderBenchmarks();
  renderSupply();
  renderPresets();
}

function loadPreset(name) {
  state = valuesForPreset(name);
  activePreset = name;
  activeBudget = name === "anneal" ? annealBudget : mainBudget;
  activeLane = name === "naive" ? "agentic" : "general";
  render();
}

async function init() {
  plan = offlinePlan;
  if (window.location.protocol !== "file:") {
    try {
      const response = await fetch(PLAN_URL);
      if (!response.ok) throw new Error(`Plan returned ${response.status}`);
      plan = await response.json();
    } catch (error) {
      console.warn("Using embedded mixture plan because the JSON plan was unavailable.", error);
    }
  }
  lanes = plan.lanes;
  mainBudget = plan.model.main_pretraining_tokens;
  annealBudget = plan.anneal.reserve_tokens;
  activeBudget = mainBudget;
  buildControls();
  document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => loadPreset(button.dataset.preset));
  });
  loadPreset("main");
}

init().catch((error) => {
  console.error(error);
  document.body.dataset.status = "plan-unavailable";
});

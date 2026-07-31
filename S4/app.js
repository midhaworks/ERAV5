const reportUrl = "data/cleanup-report.json";
const numberIN = new Intl.NumberFormat("en-IN");
const percent = new Intl.NumberFormat("en-IN", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const strategyNotes = {
  "Extract": {
    why: "Raw containers are not training examples. Parsing and schema validation separate usable records from malformed or structurally ambiguous input.",
    result: "61,000 JSONL rows parsed; 11 nested summaries failed the required four-field schema.",
    time: 430,
  },
  "Normalize": {
    why: "Equivalent Unicode and newline forms should not become different tokens or evade later comparisons. Structure that carries meaning must remain intact.",
    result: "16,086 user fields changed under NFC, newline, control-character and trailing-space normalization.",
    time: 554,
  },
  "Language ID": {
    why: "A folder or dataset label is not proof of language. This corpus also contains code and translations, so aggressive monolingual filtering would destroy valid reasoning.",
    result: "57,469 retained rows are Latin-dominant and 59 are mixed-script. Mixed rows were reported, not deleted.",
    time: 790,
  },
  "Quality filter": {
    why: "Malformed, empty, implausibly short or long, incomplete and non-compressive examples spend compute without teaching the intended summarisation behaviour.",
    result: "1,954 rows failed length, completeness, schema, compression or prompt-injection gates.",
    time: 1019,
  },
  "Deduplicate": {
    why: "Exact copies overweight examples; lightly edited copies do the same while escaping byte-level hashes. Both must be considered.",
    result: "1 exact duplicate and 5 near-duplicates were quarantined after canonical SHA-256 and SimHash checks.",
    time: 1100,
  },
  "PII scrub": {
    why: "Email addresses and phone numbers are rarely useful reasoning targets. High-confidence replacement preserves the surrounding task while reducing memorisation risk.",
    result: "1,075 email and 383 contextual phone patterns were replaced. No secret-key pattern was detected.",
    time: 1244,
  },
  "Decontaminate": {
    why: "Benchmark traces invalidate evaluation, while records with unresolved upstream rights create a separate provenance risk.",
    result: "10 benchmark-marked records and 1,502 code-dominant records with untraceable source provenance were quarantined.",
    time: 1393,
  },
  "Manifest": {
    why: "A clean file without its source, rules, counts and hashes cannot be independently audited or reproduced.",
    result: "Input, clean output, quarantine and pipeline SHA-256 hashes are recorded with policy and distribution statistics.",
    time: 1493,
  },
};

const rejectionLabels = {
  "user_length_outlier": "User length outlier",
  "code_provenance_unknown": "Code provenance unknown",
  "incomplete_or_placeholder_task": "Incomplete / placeholder",
  "invalid_summary_schema": "Invalid summary schema",
  "benchmark_contamination": "Benchmark contamination",
  "near_duplicate": "Near duplicate",
  "exact_duplicate": "Exact duplicate",
  "non_compressive_summary": "Non-compressive summary",
  "prompt_injection": "Prompt injection",
};

const mutationLabels = {
  "normalized_user_rows": "Normalised user rows",
  "email_redactions": "Email redactions",
  "phone_redactions": "Phone redactions",
  "normalized_summary_fields": "Normalised summary fields",
  "secret_redactions": "Secret redactions",
};

let cleanupReport;

function compactIndian(value) {
  if (value >= 10_000_000) return `${(value / 10_000_000).toFixed(2)} cr`;
  if (value >= 100_000) return `${(value / 100_000).toFixed(2)} lakh`;
  return numberIN.format(value);
}

function setText(key, value) {
  document.querySelectorAll(`[data-value="${key}"]`).forEach((node) => {
    node.textContent = value;
  });
}

function hydrateValues(report) {
  const {
    counts,
    mutations,
    script_profiles: profiles,
    lengths,
    dataset,
  } = report;
  const size = dataset.size_evidence;
  setText("input_rows", numberIN.format(counts.input_rows));
  setText("output_rows", numberIN.format(counts.output_rows));
  setText("rejected_rows", numberIN.format(counts.rejected_rows));
  setText("rejection_percent", `${percent.format(counts.rejection_rate * 100)}%`);
  setText("retention_percent", `${percent.format(counts.retention_rate * 100)}%`);
  setText("input_mb", `${percent.format(counts.input_megabytes_decimal)} MB`);
  setText("output_mb", `${percent.format(counts.output_megabytes_decimal)} MB`);
  setText("hf_size_mb", `${size.hugging_face_listed_size_mb.toFixed(1)} MB`);
  setText("local_size_mb", `${size.local_input_mb_decimal.toFixed(6)} MB`);
  setText("assignment_min_bytes", numberIN.format(size.assignment_min_bytes));
  setText("assignment_max_bytes", numberIN.format(size.assignment_max_bytes));
  setText("size_verdict", size.within_assignment_band ? "PASS" : "FAIL");
  setText("normalized_user_rows", numberIN.format(mutations.normalized_user_rows));
  setText("estimated_tokens_compact", compactIndian(counts.estimated_output_tokens_at_4_chars));
  setText("estimated_input_tokens", numberIN.format(counts.estimated_input_tokens_at_4_chars));
  setText("estimated_tokens", numberIN.format(counts.estimated_output_tokens_at_4_chars));
  setText("estimated_tokens_removed", `−${numberIN.format(counts.estimated_tokens_removed_at_4_chars)}`);
  setText("estimated_token_reduction", `−${percent.format(counts.estimated_token_reduction * 100)}%`);
  setText("input_bytes", numberIN.format(counts.input_bytes));
  setText("output_bytes", numberIN.format(counts.output_bytes));
  setText("bytes_removed", `−${numberIN.format(counts.bytes_removed)}`);
  setText("rows_removed", `−${numberIN.format(counts.rejected_rows)}`);
  setText("row_reduction", `−${percent.format(counts.rejection_rate * 100)}%`);
  setText("input_characters", numberIN.format(counts.input_characters));
  setText("output_characters", numberIN.format(counts.output_characters));
  setText("characters_removed", `−${numberIN.format(counts.characters_removed)}`);
  setText("character_reduction", `−${percent.format(counts.character_reduction * 100)}%`);
  setText("byte_reduction", `${percent.format(counts.byte_reduction * 100)}%`);
  setText("email_redactions", numberIN.format(mutations.email_redactions));
  setText("phone_redactions", numberIN.format(mutations.phone_redactions));
  setText("pii_redactions", numberIN.format(mutations.email_redactions + mutations.phone_redactions));
  setText("user_p50", numberIN.format(lengths.user.p50));
  setText("user_p95", numberIN.format(lengths.user.p95));
  setText("user_min", numberIN.format(lengths.user.min));
  setText("user_max", numberIN.format(lengths.user.max));
  setText("summary_p50", numberIN.format(lengths.summary.p50));
  setText("summary_p95", numberIN.format(lengths.summary.p95));
  setText("summary_min", numberIN.format(lengths.summary.min));
  setText("summary_max", numberIN.format(lengths.summary.max));

  const retained = profiles.retained_rows;
  const latin = retained.latin_dominant || 0;
  const mixed = retained.mixed_script || 0;
  setText("latin_rows", numberIN.format(latin));
  setText("mixed_rows", numberIN.format(mixed));
  setText("latin_percent", `${percent.format((latin / counts.output_rows) * 100)}%`);

  document.querySelectorAll("[data-ring='retention']").forEach((ring) => {
    ring.style.strokeDasharray = `${counts.retention_rate * 100} 100`;
  });
  const funnel = document.querySelector("[data-funnel-fill]");
  if (funnel) funnel.style.width = `${counts.retention_rate * 100}%`;

  Object.entries(report.dataset).forEach(([key, value]) => {
    document.querySelectorAll(`[data-hash="${key}"]`).forEach((node) => {
      node.textContent = value;
      node.title = value;
    });
  });
}

function renderStrategies(report) {
  const rail = document.querySelector("#strategy-rail");
  rail.innerHTML = "";
  report.stages.forEach((stage, index) => {
    const button = document.createElement("button");
    button.className = `strategy-tab${index === 0 ? " active" : ""}`;
    button.type = "button";
    button.role = "tab";
    button.id = `strategy-tab-${stage.number}`;
    button.setAttribute("aria-controls", "strategy-detail");
    button.setAttribute("aria-selected", String(index === 0));
    button.innerHTML = `<span>${String(stage.number).padStart(2, "0")}</span><strong>${stage.name}</strong><i>→</i>`;
    button.addEventListener("click", () => selectStrategy(stage, button));
    rail.append(button);
  });
  selectStrategy(report.stages[0], rail.firstElementChild);
}

function selectStrategy(stage, button) {
  document.querySelectorAll(".strategy-tab").forEach((tab) => {
    const active = tab === button;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  const notes = strategyNotes[stage.name];
  document.querySelector("#detail-number").textContent = String(stage.number).padStart(2, "0");
  document.querySelector("#detail-name").textContent = stage.name;
  document.querySelector("#detail-why").textContent = notes.why;
  document.querySelector("#detail-applied").textContent = stage.applied;
  document.querySelector("#detail-result").textContent = notes.result;
  document.querySelector("#detail-source").href = `https://www.youtube.com/watch?v=GpS-oisqkqA&t=${notes.time}s`;
}

function renderChart(mode) {
  const chart = document.querySelector("#bar-chart");
  const isRejections = mode === "rejections";
  const data = isRejections ? cleanupReport.primary_rejections : cleanupReport.mutations;
  const labels = isRejections ? rejectionLabels : mutationLabels;
  const rows = Object.entries(data)
    .filter(([, value]) => value > 0)
    .sort((a, b) => b[1] - a[1]);
  const max = Math.max(...rows.map(([, value]) => value), 1);

  chart.innerHTML = "";
  rows.forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label">${labels[key] || key}</span>
      <span class="bar-track"><i class="bar-fill" style="width:${(value / max) * 100}%"></i></span>
      <strong class="bar-value">${numberIN.format(value)}</strong>`;
    chart.append(row);
  });

  document.querySelector("#chart-title").textContent = isRejections ? "Quarantine reasons" : "In-place mutations";
  document.querySelector("#chart-note").textContent = isRejections
    ? "Primary reason per rejected row; categories therefore sum exactly to the quarantine total."
    : "Mutation occurrences are pattern or row counts. A retained record may contain more than one redaction.";
  document.querySelectorAll("[data-chart-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.chartMode === mode);
  });
}

function setupControls() {
  document.querySelectorAll("[data-chart-mode]").forEach((button) => {
    button.addEventListener("click", () => renderChart(button.dataset.chartMode));
  });
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const value = cleanupReport.dataset[button.dataset.copy];
      try {
        await navigator.clipboard.writeText(value);
        const previous = button.textContent;
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = previous; }, 1200);
      } catch {
        button.textContent = "Select hash";
      }
    });
  });
}

async function init() {
  try {
    const response = await fetch(reportUrl);
    if (!response.ok) throw new Error(`Report returned ${response.status}`);
    cleanupReport = await response.json();
    hydrateValues(cleanupReport);
    renderStrategies(cleanupReport);
    renderChart("rejections");
    setupControls();
  } catch (error) {
    console.error(error);
    document.body.dataset.reportStatus = "unavailable";
  }
}

init();

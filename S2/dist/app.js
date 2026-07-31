// Copyright 2026 Avnish Midha. All rights reserved.
// Author: Avnish Midha
// GitHub: avnishbm
// Purpose: Power the interactive tokenizer review, rankings, and filters.

(() => {
  "use strict";

  const { PieceVocab, BPE, faithfulUnits } = window.ReviewTokenizers;
  const DIRECTION_OBSERVATION = {
    title: "RTL changes display order, not the stored sequence",
    copy: "Unicode stores right-to-left text in logical character order. The bidirectional algorithm changes its visual presentation; it does not reverse the serialized code-point sequence consumed by a tokenizer or an autoregressive model.",
    idea: "Writing direction is therefore not an eligibility criterion. All 306 measured languages—including the 19 MediaWiki marks as RTL—remain in the ranking.",
    note: "With only the content filter removed, Yiddish is the global PieceVocab winner at 18,33,650.92. At the selected 5,000-visible-word threshold it is excluded because its page contains only 2,404 visible word runs, not because it is RTL.",
    links: [
      ["Unicode Bidirectional Algorithm ↗", "https://unicode.org/reports/tr9/"],
      ["MediaWiki language metadata ↗", "https://www.mediawiki.org/wiki/API:Languageinfo"],
    ],
  };
  const MEASUREMENT_OBSERVATION = {
    title: "Filter source content, not generated tokens",
    copy: "Eligibility should use a measure available before tokenizer training. Token counts depend on the tokenizer, its vocabulary allocation, and the script, so a token threshold would be circular and would create different candidate sets for PieceVocab and BPE.",
    idea: "The selected floor is 5,000 visible Unicode word runs: 86 of 306 candidates qualify. The interactive table keeps neighboring thresholds visible because the PieceVocab leader changes at 6,000, while the BPE leader remains stable.",
    note: "A Unicode word run is a transparent heuristic, not a universal linguistic word count. Languages without explicit word separators need a future script-aware check using segmentation or visible grapheme/character volume.",
    links: [["Unicode text segmentation ↗", "https://unicode.org/reports/tr29/"]],
  };
  const CONFIG = {
    piece: {
      kind: "Custom lossless piece vocabulary",
      title: "PieceVocab",
      description: "Frequent whitespace-prefixed pieces, literal character fallback, and reversible Unicode escapes.",
      strategy: "Build each corpus from faithful Markdown of Wikipedia's India page. Fix English, Hindi, and Telugu, evaluate all 306 fourth-language candidates regardless of writing direction, then select from the 86 pages containing at least 5,000 visible word runs.",
      languages: [
        ["en", "English", "https://en.wikipedia.org/wiki/India"],
        ["hi", "Hindi", "https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4"],
        ["te", "Telugu", "https://te.wikipedia.org/wiki/%E0%B0%AD%E0%B0%BE%E0%B0%B0%E0%B0%A4%E0%B0%A6%E0%B1%87%E0%B0%B6%E0%B0%82"],
        ["yo", "Yoruba", "https://yo.wikipedia.org/wiki/%C3%8Dnd%C3%AD%C3%A0"],
      ],
      score: 786647.1950729131,
      spread: 0.0012712179058965711,
      ratios: { en: .7771984594, hi: .7782002965, te: .7784696773, yo: .7772450146 },
      tokens: { en: 144890, hi: 68761, te: 28253, yo: 24477 },
      units: { en: 186426, hi: 88359, te: 36293, yo: 31492 },
      sample: "India's population is 1,42,86,27,663. भारत की जनसंख्या विशाल है। భారతదేశం దక్షిణ ఆసియాలో ఉంది. Índíà jẹ́ orílẹ̀-èdè kan ní Gúúsù Ásíà.",
      data: "downloads/piecevocab.tokenizer.json",
      implementation: "downloads/piecevocab.py",
      summary: "downloads/piecevocab.summary.md",
      guide: "downloads/piecevocab.SUBMISSION.md",
      implementationName: "piecevocab.py",
      loaderRequired: true,
      note: "REQUIRED: submit and load piecevocab.py with tokenizer.json. The custom JSON is data only and has no callable decode() method by itself.",
      method: [
        ["1 · Search, then apply content eligibility", "English, Hindi, and Telugu were fixed. All 306 fourth-language pages received the same 20-pass fertility-balancing search. Only the pre-tokenizer content threshold was then applied."],
        ["2 · Lossless representation", "Keep leading whitespace inside pieces. Selected pieces cost one token; other text uses literal characters. Seventeen reserved tokens encode any unseen Unicode code point."],
        ["3 · Eligible top score", "Yoruba ranked first among the 86 content-eligible candidates. Complete pages and arbitrary unseen Unicode reconstruct exactly."],
      ],
      highlight: {
        label: "Eligible top score",
        value: "Yoruba · 7,86,647.20",
        copy: "Yoruba produced a fertility spread of 0.00127122 and the maximum score after applying only the 5,000-visible-word content rule.",
        secondary: {
          label: "Best eligible Indic candidate",
          value: "Odia · 1,98,765.84",
          copy: "Spread 0.00503105 · 12th among eligible candidates",
        },
      },
      indic: {
        name: "Odia",
        score: 198765.8352,
        spread: .0050310457,
        languages: [
          ["en", "English", "https://en.wikipedia.org/wiki/India"],
          ["hi", "Hindi", "https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4"],
          ["te", "Telugu", "https://te.wikipedia.org/wiki/%E0%B0%AD%E0%B0%BE%E0%B0%B0%E0%B0%A4%E0%B0%A6%E0%B1%87%E0%B0%B6%E0%B0%82"],
          ["or", "Odia", "https://or.wikipedia.org/wiki/%E0%AC%AD%E0%AC%BE%E0%AC%B0%E0%AC%A4"],
        ],
        tokens: { en: 147534, hi: 69667, te: 28539, or: 11463 },
        units: { en: 186426, hi: 88359, te: 36293, or: 14544 },
        ratios: { en: .7913810305, hi: .7884539209, te: .7863499848, or: .7881600660 },
      },
      observation: DIRECTION_OBSERVATION,
      measurement: MEASUREMENT_OBSERVATION,
      improvements: [
        ["Script-aware content measurement", "Supplement visible word runs with Unicode segmentation or visible grapheme volume for scripts that do not consistently use spaces."],
        ["Page-quality validation", "Reject stubs, disambiguation pages, navigation-heavy pages, and pages whose visible article content is too incomplete for tokenizer evaluation."],
        ["Translation and topic comparability", "Measure whether each localized page covers comparable India-related concepts and sections—potentially using translation-assisted semantic checks in addition to raw size."],
        ["Preserve both views", "Keep the complete 306-candidate ranking beside the 86-candidate eligible ranking so every exclusion remains transparent."],
      ],
    },
    bpe: {
      kind: "Shared Hugging Face BPE with Metaspace",
      title: "BPE",
      description: "A standard shared BPE model with no Unicode normalization and an exact 10,000-entry vocabulary.",
      strategy: "Build each corpus from faithful Markdown of Wikipedia's India page. Fix English, Hindi, and Telugu, evaluate the same 16 profiles for all 306 fourth-language candidates regardless of writing direction, then select from the 86 pages containing at least 5,000 visible word runs.",
      languages: [
        ["en", "English", "https://en.wikipedia.org/wiki/India"],
        ["hi", "Hindi", "https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4"],
        ["te", "Telugu", "https://te.wikipedia.org/wiki/%E0%B0%AD%E0%B0%BE%E0%B0%B0%E0%B0%A4%E0%B0%A6%E0%B1%87%E0%B0%B6%E0%B0%82"],
        ["sd", "Sindhi", "https://sd.wikipedia.org/wiki/%DA%80%D8%A7%D8%B1%D8%AA"],
      ],
      score: 86311.01784757704,
      spread: 0.011586006340070898,
      ratios: { en: .6366172100, hi: .6269197252, te: .6372027664, sd: .6385057316 },
      tokens: { en: 118682, hi: 55394, te: 23126, sd: 47123 },
      units: { en: 186426, hi: 88359, te: 36293, sd: 73802 },
      sample: "India's population is 1,42,86,27,663. भारत की जनसंख्या विशाल है। భారతదేశం దక్షిణ ఆసియాలో ఉంది. ڀارت ڏکڻ ايشيا جو هڪ ملڪ آهي.",
      data: "downloads/bpe.tokenizer.json",
      implementation: "downloads/bpe_common.py",
      summary: "downloads/bpe.summary.md",
      implementationName: "bpe_common.py",
      note: "The JSON is Hugging Face Tokenizers compatible. The helper shows the exact trainer, Metaspace configuration, weights, and evaluation policy.",
      method: [
        ["1 · Search, then apply content eligibility", "Every one of 306 fourth-language candidates was evaluated across the same 16 declared training-weight profiles. Only the pre-tokenizer content threshold was then applied."],
        ["2 · Faithful BPE", "Use one shared Hugging Face BPE, no normalizer, a Metaspace pre-tokenizer/decoder, min_frequency=1, and [UNK] as the sole special token."],
        ["3 · Eligible top score", "Sindhi produced the top eligible score with weights English 3, Hindi 4, Telugu 6, and Sindhi 4. All four complete pages decode exactly with zero unknown tokens."],
      ],
      highlight: {
        label: "Eligible top score",
        value: "Sindhi · 86,311.02",
        copy: "The next-best eligible candidates are Italian at 55,234.82 and Persian at 53,811.11.",
        secondary: {
          label: "Best eligible Indic candidate",
          value: "Sindhi · 86,311.02",
          copy: "The overall winner is also the best eligible Indic candidate.",
        },
      },
      indic: {
        name: "Sindhi",
        score: 86311.01784757704,
        spread: .011586006340070898,
        languages: [
          ["en", "English", "https://en.wikipedia.org/wiki/India"],
          ["hi", "Hindi", "https://hi.wikipedia.org/wiki/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4"],
          ["te", "Telugu", "https://te.wikipedia.org/wiki/%E0%B0%AD%E0%B0%BE%E0%B0%B0%E0%B0%A4%E0%B0%A6%E0%B1%87%E0%B0%B6%E0%B0%82"],
          ["sd", "Sindhi", "https://sd.wikipedia.org/wiki/%DA%80%D8%A7%D8%B1%D8%AA"],
        ],
        tokens: { en: 118682, hi: 55394, te: 23126, sd: 47123 },
        units: { en: 186426, hi: 88359, te: 36293, sd: 73802 },
        ratios: { en: .6366172100, hi: .6269197252, te: .6372027664, sd: .6385057316 },
      },
      observation: DIRECTION_OBSERVATION,
      measurement: MEASUREMENT_OBSERVATION,
      improvements: [
        ["Script-aware content measurement", "Supplement visible word runs with Unicode segmentation or visible grapheme volume for scripts that do not consistently use spaces."],
        ["Page-quality validation", "Reject stubs, disambiguation pages, navigation-heavy pages, and pages whose visible article content is too incomplete for tokenizer evaluation."],
        ["Translation and topic comparability", "Use translation-assisted semantic checks to confirm that localized India pages cover sufficiently comparable concepts and sections."],
        ["Preserve both views", "Keep the complete 306-candidate ranking beside the 86-candidate eligible ranking so every exclusion is auditable."],
      ],
    },
  };

  const state = { approach: "piece", panel: "summary", view: "tokens", models: {}, bundles: {}, rankings: null, filters: { piece: 5000, bpe: 5000 }, page: 1, query: "", ids: [] };
  const $ = id => document.getElementById(id);
  const format = value => Number(value).toLocaleString("en-IN", { maximumFractionDigits: 2 });
  const visibleToken = token => token.replaceAll("\u0000", "␀").replaceAll(" ", "·").replaceAll("\n", "↵\n").replaceAll("\t", "⇥");

  async function loadModel(name) {
    if (state.models[name]) return state.models[name];
    const response = await fetch(CONFIG[name].data);
    if (!response.ok) throw new Error(`Could not load ${CONFIG[name].data}`);
    const bundle = await response.json();
    state.bundles[name] = bundle;
    state.models[name] = name === "piece" ? new PieceVocab(bundle) : new BPE(bundle);
    return state.models[name];
  }

  async function loadRankings() {
    if (state.rankings) return state.rankings;
    const response = await fetch("language-rankings.json");
    if (!response.ok) throw new Error("Could not load the language rankings");
    state.rankings = await response.json();
    return state.rankings;
  }

  async function selectApproach(name) {
    state.approach = name;
    state.page = 1;
    state.query = "";
    $("vocabSearch").value = "";
    document.querySelectorAll(".approach-tab").forEach(button => {
      const active = button.dataset.approach === name;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active);
    });
    await Promise.all([loadModel(name), loadRankings()]);
    renderHeader();
    $("textInput").value = CONFIG[name].sample;
    renderSummary();
    renderDownloads();
    tokenize();
    renderVocabulary();
  }

  function renderHeader() {
    const config = CONFIG[state.approach];
    $("approachKind").textContent = config.kind;
    $("approachTitle").textContent = config.title;
    $("approachDescription").textContent = config.description;
    $("score").textContent = format(config.score);
    $("spread").textContent = `spread ${config.spread.toFixed(8)}`;
    $("languagePills").innerHTML = config.languages.map(([code, name], index) =>
      `<span><b>${code.toUpperCase()}</b>${name}${index === 3 ? " · selected fourth" : ""}</span>`
    ).join("");
  }

  function tokenize() {
    const text = $("textInput").value;
    const model = state.models[state.approach];
    state.ids = model.encode(text);
    const decoded = model.decode(state.ids);
    const units = faithfulUnits(text);
    const exact = decoded === text;
    $("tokenCount").textContent = format(state.ids.length);
    $("unitCount").textContent = format(units);
    $("fertility").textContent = units ? (state.ids.length / units).toFixed(4) : "—";
    $("roundtripStat").textContent = exact ? "Exact" : "Mismatch";
    $("roundtripBadge").textContent = exact ? "✓ exact decode" : "! decode differs";
    $("roundtripBadge").className = exact ? "pass-badge" : "pass-badge fail";
    $("decodedText").textContent = decoded || "Decoded text will appear here.";
    renderTokens();
  }

  function renderTokens() {
    const model = state.models[state.approach];
    const output = $("tokenOutput");
    if (!state.ids.length) {
      output.innerHTML = '<p class="empty">Enter text to inspect its tokenization.</p>';
      return;
    }
    output.innerHTML = state.ids.slice(0, 800).map((id, index) => {
      const text = state.view === "ids" ? id : visibleToken(model.token(id));
      return `<span class="token t${index % 7}" title="ID ${id}">${escapeHtml(String(text))}</span>`;
    }).join("") + (state.ids.length > 800 ? `<span class="empty">Showing first 800 of ${format(state.ids.length)} tokens.</span>` : "");
  }

  function vocabularyRows() {
    const model = state.models[state.approach];
    const query = state.query.toLocaleLowerCase();
    return model.vocab.map((token, id) => ({ token, id })).filter(row => !query || row.token.toLocaleLowerCase().includes(query));
  }

  function renderVocabulary() {
    if (!state.models[state.approach]) return;
    const rows = vocabularyRows();
    const perPage = 120;
    const pages = Math.max(1, Math.ceil(rows.length / perPage));
    state.page = Math.min(state.page, pages);
    const slice = rows.slice((state.page - 1) * perPage, state.page * perPage);
    $("vocabSize").textContent = format(state.models[state.approach].vocab.length);
    $("vocabGrid").innerHTML = slice.map(({ token, id }, index) =>
      `<span class="vocab-token t${index % 7}"><b>${escapeHtml(visibleToken(token) || "∅")}</b><small>${id}</small></span>`
    ).join("") || '<p class="empty">No tokens match that search.</p>';
    $("pageInfo").textContent = `Page ${state.page} of ${pages} · ${format(rows.length)} matches`;
    $("prevPage").disabled = state.page === 1;
    $("nextPage").disabled = state.page === pages;
  }

  function renderSummary() {
    const config = CONFIG[state.approach];
    const minimumWords = state.filters[state.approach];
    const eligibleRanking = state.rankings[state.approach].filter(
      row => row.visible_word_runs >= minimumWords
    );
    const secondaryHighlight = config.highlight.secondary ? `
      <div class="secondary-highlight">
        <span>${config.highlight.secondary.label}</span>
        <b>${config.highlight.secondary.value}</b>
        <small>${config.highlight.secondary.copy}</small>
      </div>` : "";
    const languageRows = config.languages.map(([code, name, url]) => `
      <tr><td><a class="wiki-link" href="${url}" target="_blank" rel="noreferrer"><b>${name}</b><small>${code} · Wikipedia ↗</small></a></td><td>${format(config.tokens[code])}</td><td>${format(config.units[code])}</td><td>${config.ratios[code].toFixed(6)}</td></tr>`).join("");
    const rankingRows = eligibleRanking.slice(0, 10).map((row, index) => `
      <tr><td class="rank-cell">${index + 1}<small>raw ${row.rank}</small></td><td><a class="wiki-link" href="${row.article_url}" target="_blank" rel="noreferrer"><b>${row.name}</b><small>${row.code} · Wikipedia ↗</small></a></td><td>${format(row.visible_word_runs)}</td><td>${format(row.tokens)}</td><td>${format(row.faithful_units)}</td><td>${row.fourth_fertility.toFixed(6)}</td><td>${format(row.score)}</td><td>${row.spread.toFixed(8)}</td></tr>`).join("");
    const filteredWinner = eligibleRanking[0];
    const indicOutcome = config.indic ? `
      <article class="card metrics-card indic-outcome">
        <div class="card-head"><div><p class="eyebrow">Best Indic outcome</p><h3>${config.indic.name} candidate</h3></div><div class="mini-score"><b>${format(config.indic.score)}</b><span>score · spread ${config.indic.spread.toFixed(8)}</span></div></div>
        <div class="table-wrap"><table><thead><tr><th>Language</th><th>Tokens</th><th>Faithful units</th><th>Fertility</th></tr></thead><tbody>
          ${config.indic.languages.map(([code, name, url]) => `<tr><td><a class="wiki-link" href="${url}" target="_blank" rel="noreferrer"><b>${name}</b><small>${code} · Wikipedia ↗</small></a></td><td>${format(config.indic.tokens[code])}</td><td>${format(config.indic.units[code])}</td><td>${config.indic.ratios[code].toFixed(6)}</td></tr>`).join("")}
        </tbody></table></div>
      </article>` : "";
    const commonReferences = `
      <li><a href="https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q/sessions/cmrirwdhc0afw08nmp64282xo/lesson" target="_blank" rel="noreferrer">ERA V5 Assignment 2 Reference Solution from Admin</a></li>
      <li><a href="https://dcsuiova318m4.cloudfront.net" target="_blank" rel="noreferrer">Layout inspiration — submission from a peer</a></li>`;
    const technicalReference = state.approach === "bpe"
      ? '<li><a href="https://huggingface.co/docs/tokenizers/main/components" target="_blank" rel="noreferrer">Hugging Face Tokenizers components — BPE and Metaspace</a></li>'
      : "";
    const referencesList = state.approach === "bpe" ? `${technicalReference}${commonReferences}` : commonReferences;
    const observation = config.observation ? `
      <article class="observation-card">
        <div><p class="eyebrow">Interesting observation</p><h3>${config.observation.title}</h3><p>${config.observation.copy}</p><p class="observation-idea">${config.observation.idea}</p><small>${config.observation.note}</small></div>
        <div class="observation-links">${config.observation.links.map(([label, url]) => `<a href="${url}" target="_blank" rel="noreferrer">${label}</a>`).join("")}</div>
      </article>` : "";
    const measurement = config.measurement ? `
      <article class="observation-card data-observation">
        <div><p class="eyebrow">Filter decision</p><h3>${config.measurement.title}</h3><p>${config.measurement.copy}</p><p class="observation-idea">${config.measurement.idea}</p><small>${config.measurement.note}</small></div>
        <div class="observation-links">${config.measurement.links.map(([label, url]) => `<a href="${url}" target="_blank" rel="noreferrer">${label}</a>`).join("")}</div>
      </article>` : "";
    const improvements = config.improvements ? `
      <article class="card improvements-card"><p class="eyebrow">Future work</p><h3>Possible improvements</h3><div class="improvements-grid">
        ${config.improvements.map(([title, copy], index) => `<section><span>${index + 1}</span><div><h4>${title}</h4><p>${copy}</p></div></section>`).join("")}
      </div></article>` : "";
    const loaderNotice = config.loaderRequired ? `
      <article class="loader-notice">
        <div><p class="eyebrow">Required evaluation interface</p><h3>Load <code>piecevocab.py</code> with <code>tokenizer.json</code></h3>
        <p><strong>Do not evaluate the JSON alone.</strong> It contains vocabulary data, while <code>piecevocab.py</code> provides the required <code>encode()</code> and <code>decode()</code> methods.</p></div>
        <pre><code>from piecevocab import load
tokenizer = load("tokenizer.json")
assert tokenizer.decode(tokenizer.encode(text)) == text</code></pre>
      </article>` : "";
    $("summary").innerHTML = `
      <article class="strategy-card">
        <p class="eyebrow">Primary strategy</p>
        <h3>Evaluate every possible Wikipedia “India” page language</h3>
        <p>${config.strategy}</p>
        <div class="strategy-flow"><span>306 measured</span><b>→</b><span>all directions retained</span><b>→</b><span>86 with 5,000+ visible words</span><b>→</b><span>eligible top score</span></div>
      </article>
      ${loaderNotice}
      <div class="method-grid">
        <article class="card steps"><p class="eyebrow">Unique idea & process</p><h3>${config.title} summary</h3>
          ${config.method.map(([title, copy]) => `<section><h4>${title}</h4><p>${copy}</p></section>`).join("")}
        </article>
        <aside class="highlight-card"><p class="eyebrow">${config.highlight.label}</p><strong>${config.highlight.value}</strong><p>${config.highlight.copy}</p>${secondaryHighlight}</aside>
      </div>
      ${observation}
      ${measurement}
      <article class="card metrics-card"><div class="card-head"><div><p class="eyebrow">Outcome</p><h3>Per-language evaluation</h3></div><span class="exact-pill">Exact full-page round trips</span></div>
        <div class="table-wrap"><table><thead><tr><th>Language</th><th>Tokens</th><th>Faithful units</th><th>Fertility</th></tr></thead><tbody>${languageRows}</tbody></table></div>
      </article>${indicOutcome}
      <article class="card metrics-card ranking-card">
        <div class="card-head"><div><p class="eyebrow">Interactive filtered ranking</p><h3>Top 10 eligible fourth-language candidates</h3></div><span class="exact-pill">306 evaluated · 19 RTL included</span></div>
        <div class="filter-bar">
          <div><label for="visibleWordFilter">Minimum visible word runs</label><input id="visibleWordFilter" type="number" min="0" step="500" value="${minimumWords}"></div>
          <div class="filter-presets">
            ${[0, 5000, 6000, 7000, 8000, 10000].map(value => `<button class="${minimumWords === value ? "active" : ""}" data-filter="${value}">${value === 0 ? "No word filter" : format(value)}</button>`).join("")}
          </div>
          <div class="filter-result"><b>${eligibleRanking.length}/306 eligible</b><span>${filteredWinner ? `displayed leader · ${filteredWinner.name}` : "no eligible candidates"}</span></div>
        </div>
        <p class="filter-note">The selected submission policy defaults to 5,000 visible word runs to remove very small pages while retaining 86 candidates. Change the threshold to explore all 306 candidates; the table, eligible count, and displayed leader update immediately. Eligibility is intentionally pre-tokenizer—using generated token counts would be circular and approach-dependent. The PieceVocab winner changes to Pashto at 6,000, so that sensitivity is reported rather than hidden. This view does not retrain or replace the submitted Yoruba/Sindhi artifacts. “Raw” rank is the position in the complete ranking.</p>
        <div class="table-wrap"><table><thead><tr><th>Eligible rank</th><th>Fourth language</th><th>Visible words</th><th>Tokens</th><th>Faithful units</th><th>Fertility</th><th>Score</th><th>Spread</th></tr></thead><tbody>${rankingRows || '<tr><td colspan="8">No candidates meet this threshold.</td></tr>'}</tbody></table></div>
      </article>
      ${improvements}
      <article class="card references-card"><p class="eyebrow">Sources</p><h3>References</h3><ol>${referencesList}</ol><div class="acknowledgement"><span>Acknowledgements</span><div class="acknowledgement-items"><b class="program-ack">Program: <a href="https://theschoolof.ai" target="_blank" rel="noopener noreferrer">Extensive &amp; Reimagined AI Program - The School of AI</a><a class="linkedin-mark" href="https://www.linkedin.com/in/the-school-of-ai-78288b194/" target="_blank" rel="noopener noreferrer" aria-label="The School of AI on LinkedIn">in</a></b><b>Intern who helped code: Codex :)</b></div></div></article>`;
    $("visibleWordFilter").addEventListener("change", event => {
      state.filters[state.approach] = Math.max(0, Number(event.target.value) || 0);
      renderSummary();
    });
    document.querySelectorAll(".filter-presets button").forEach(button => button.addEventListener("click", () => {
      state.filters[state.approach] = Number(button.dataset.filter);
      renderSummary();
    }));
  }

  function renderDownloads() {
    const config = CONFIG[state.approach];
    $("downloadTitle").textContent = `${config.title} submission bundle`;
    $("downloadNote").textContent = config.note;
    const tokenizerLink = `<a class="download-link ${config.loaderRequired ? "" : "primary"}" href="${config.data}" download><span>${config.loaderRequired ? "Required tokenizer data" : "Tokenizer artifact"}</span><b>tokenizer.json ↓</b></a>`;
    const implementationLink = `<a class="download-link ${config.loaderRequired ? "primary" : ""}" href="${config.implementation}" download><span>${config.loaderRequired ? "Required executable definition" : "Executable definition"}</span><b>${config.implementationName} ↓</b></a>`;
    const guideLink = config.guide ? `<a class="download-link required-guide" href="${config.guide}" download><span>Read before evaluation</span><b>SUBMISSION.md ↓</b></a>` : "";
    $("downloadLinks").innerHTML = `${implementationLink}${tokenizerLink}
      ${guideLink}<a class="download-link" href="${config.summary}" download><span>Idea, process & outcome</span><b>summary.md ↓</b></a>`;
  }

  function selectPanel(panel) {
    state.panel = panel;
    document.querySelectorAll(".subtab").forEach(button => button.classList.toggle("active", button.dataset.panel === panel));
    document.querySelectorAll(".panel").forEach(section => section.classList.toggle("active", section.id === `panel-${panel}`));
    if (panel === "vocabulary") renderVocabulary();
  }

  async function fetchWikipedia() {
    const button = $("fetchWiki");
    const match = $("wikiUrl").value.trim().match(/^https?:\/\/([a-z-]+)\.wikipedia\.org\/wiki\/(.+)$/i);
    if (!match) { $("status").textContent = "Use a URL such as https://en.wikipedia.org/wiki/India"; return; }
    button.disabled = true;
    button.textContent = "Fetching…";
    try {
      const url = `https://${match[1]}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&origin=*&titles=${encodeURIComponent(decodeURIComponent(match[2]))}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Wikipedia returned HTTP ${response.status}`);
      const data = await response.json();
      const page = Object.values(data.query.pages)[0];
      if (!page.extract) throw new Error("No article text was returned");
      $("textInput").value = page.extract;
      $("status").textContent = `Loaded ${page.title} · ${format(page.extract.length)} characters`;
      tokenize();
    } catch (error) { $("status").textContent = error.message; }
    finally { button.disabled = false; button.textContent = "Fetch article"; }
  }

  function escapeHtml(value) {
    return value.replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  }

  document.querySelectorAll(".approach-tab").forEach(button => button.addEventListener("click", () => selectApproach(button.dataset.approach)));
  document.querySelectorAll(".subtab").forEach(button => button.addEventListener("click", () => selectPanel(button.dataset.panel)));
  document.querySelectorAll(".view-button").forEach(button => button.addEventListener("click", () => {
    state.view = button.dataset.view;
    document.querySelectorAll(".view-button").forEach(item => item.classList.toggle("active", item === button));
    renderTokens();
  }));
  let timer;
  $("textInput").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(tokenize, 100); });
  $("resetSample").addEventListener("click", () => { $("textInput").value = CONFIG[state.approach].sample; $("status").textContent = ""; tokenize(); });
  $("fetchWiki").addEventListener("click", fetchWikipedia);
  $("wikiUrl").addEventListener("keydown", event => { if (event.key === "Enter") fetchWikipedia(); });
  $("vocabSearch").addEventListener("input", event => { state.query = event.target.value; state.page = 1; renderVocabulary(); });
  $("prevPage").addEventListener("click", () => { state.page -= 1; renderVocabulary(); });
  $("nextPage").addEventListener("click", () => { state.page += 1; renderVocabulary(); });

  selectApproach("piece").catch(error => { $("status").textContent = error.message; });
})();

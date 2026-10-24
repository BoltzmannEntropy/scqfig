// scqfig site: gallery (three.js wafer + cards), example pages with code, LaTeX package, styles, start.
// All content comes from data/site.json, written by tools/build_site.py.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private window: keep the default */ } },
};
const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const S = {
  data: null, byId: {}, famColor: {}, wafer: null, gl: 'loading three.js', lastId: null, q: '', off: new Set(),
  style: store.get('scqfig.style') === 'folio' ? 'folio' : 'field',
  tab: ['py', 'tex', 'run'].includes(store.get('scqfig.tab')) ? store.get('scqfig.tab') : 'py',
};

// ---- syntax colouring: one combined regex per language, non-capturing groups inside each rule

const LANG = {
  py: [
    [/#[^\n]*/, 'c'],
    [/"""[\s\S]*?"""|'''[\s\S]*?'''|\b[rbfuRBFU]{1,2}(?:"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'/, 's'],
    [/\b(?:def|class|return|import|from|as|for|in|if|elif|else|while|with|lambda|yield|try|except|finally|raise|not|and|or|is|None|True|False|pass|break|continue|global|nonlocal|assert|del)\b/, 'k'],
    [/@\w+/, 'd'],
    [/\b\d+(?:\.\d*)?(?:[eE][-+]?\d+)?j?\b/, 'n'],
    [/\b[A-Za-z_]\w*(?=\()/, 'f'],
  ],
  tex: [
    [/%[^\n]*/, 'c'],
    [/\\(?:[A-Za-z@]+\*?|.)/, 'f'],
    [/\$[^$\n]*\$/, 's'],
    [/[{}[\]]/, 'b'],
    [/\b\d+(?:\.\d+)?(?:pt|cm|mm|in|em|ex)?\b/, 'n'],
  ],
  sh: [
    [/#[^\n]*/, 'c'],
    [/^(?:python|pip|pdflatex|git|cd)\b/, 'k'],
    [/(?:^|\s)--?[A-Za-z][\w-]*/, 'n'],
  ],
};
const LANG_RE = Object.fromEntries(Object.entries(LANG).map(([k, rules]) =>
  [k, new RegExp(rules.map(([r]) => `(${r.source})`).join('|'), 'gm')]));

function hl(src, lang) {
  const rules = LANG[lang];
  if (!rules) return esc(src);
  const re = LANG_RE[lang];
  re.lastIndex = 0;
  let out = '', i = 0, m;
  while ((m = re.exec(src))) {
    if (!m[0]) { re.lastIndex++; continue; }
    const g = m.slice(1).findIndex(x => x !== undefined);
    out += esc(src.slice(i, m.index)) + `<span class="tk-${rules[g][1]}">${esc(m[0])}</span>`;
    i = m.index + m[0].length;
  }
  return out + esc(src.slice(i));
}
const guessLang = c => (/^\s*(?:import|from)\s/m.test(c) ? 'py' : /^\s*(?:pip|python|git)\s/m.test(c) ? 'sh' : /\\[A-Za-z]/.test(c) ? 'tex' : '');
const codebox = (code, lang = guessLang(code)) =>
  `<div class="codebox"><button type="button" class="copy">Copy</button><pre><code>${hl(code.replace(/\n$/, ''), lang)}</code></pre></div>`;

// tiny Markdown for the README paragraphs: inline code, bold, links as text, and "- " lists
const inl = s => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
function md(block) {
  let html = '', para = [], list = [];
  const flushP = () => { if (para.length) html += `<p>${inl(para.join(' '))}</p>`; para = []; };
  const flushL = () => { if (list.length) html += `<ul>${list.map(l => `<li>${inl(l)}</li>`).join('')}</ul>`; list = []; };
  for (const l of block.split('\n')) {
    if (/^\s*[-*] /.test(l)) { flushP(); list.push(l.replace(/^\s*[-*] /, '')); } else { flushL(); para.push(l.trim()); }
  }
  flushP(); flushL();
  return html;
}

async function copy(text, btn) {
  try { await navigator.clipboard.writeText(text); } catch {
    const ta = Object.assign(document.createElement('textarea'), { value: text });
    ta.style.cssText = 'position:fixed;opacity:0';
    document.body.append(ta);
    ta.select();
    try { document.execCommand('copy'); } catch { /* nothing else to try */ }
    ta.remove();
  }
  btn.textContent = 'Copied';
  btn.classList.add('done');
  setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('done'); }, 1400);
}

const other = st => (st === 'field' ? 'folio' : 'field');
const segHTML = () => `<div class="seg" role="group" aria-label="Figure style">${['field', 'folio'].map(s =>
  `<button type="button" data-style="${s}" aria-pressed="${S.style === s}">${s}</button>`).join('')}</div>`;
const famDot = f => `<span class="fam"><i style="--c:${S.famColor[f] || '#888'}"></i>${esc(f)}</span>`;

// ---- gallery ---------------------------------------------------------------------------------

const cardHTML = e => {
  const im = e.images[S.style];
  return `<a class="card" href="#/example/${e.id}" data-id="${e.id}"><div class="thumb"><img loading="lazy" decoding="async"
    src="${im.thumb}" width="${im.w}" height="${im.h}" alt="${esc(e.title)}, ${S.style} style"></div><div class="meta">
    <span class="num">${e.num}</span><b>${esc(e.title)}</b>${famDot(e.family)}</div></a>`;
};

function stageCaption() {
  return `<span class="long">One die per example figure, drawn in the <b>${S.style}</b> style. Drag to turn the wafer, scroll or `
    + 'pinch to zoom, point at a die to lift it, and click it to open its code.</span>'
    + '<span class="short">Drag to turn. Tap a die to lift it, tap again to open it.</span>';
}

// what the library is, for a visitor who lands on the gallery first; the counts come from the build
const introHTML = d => `
  <section class="intro" aria-label="About scqfig">
    <h1>Figures for superconducting-qubit and chip papers</h1>
    <p>scqfig is a Python library, built on matplotlib, that draws the figures these papers keep repeating:
      spectroscopy against flux, coherence decays, IQ blobs, frequency plans, chip layouts, cryostat wiring,
      pulse sequences and gate-model circuits. Levels, wavefunctions, chevrons and Wigner functions come from
      solvers in <code>scqfig.physics</code>, not from sketches.</p>
    <p>Every figure comes in two matched styles, <b>field</b> and <b>folio</b>, drawn at print width and switched
      with one call. The companion LaTeX package <code>scqfig.sty</code> gives TikZ, pgfplots and quantikz the same
      palette. Below are all ${d.examples.length} example figures in ${d.families.length} families, each with its
      Python and LaTeX code.</p>
    <p class="intro-links"><a class="btn" href="#/start">Install and use</a><a class="btn quiet" href="#/latex">LaTeX
      package</a><a class="btn quiet" href="#/styles">Compare the styles</a><a class="btn quiet"
      href="https://github.com/BoltzmannEntropy/scqfig">Source code on GitHub</a></p>
  </section>`;

function buildGallery() {
  const d = S.data, counts = {};
  d.examples.forEach(e => { counts[e.family] = (counts[e.family] || 0) + 1; });
  $('#gallery').innerHTML = `${introHTML(d)}
    <div class="stage-grid">
      <div class="stage loading" id="stage">
        <div class="in-stage-panel"><label><input type="checkbox" id="spin"> Spin</label>
          <button type="button" id="home">Reset view</button></div>
        <div class="tip" id="tip" hidden></div>
        <div class="caption" id="stagecap">${stageCaption()}</div>
      </div>
      <aside class="legend-wrap" aria-label="Filter by family">
        <h3>Families</h3>
        <div class="ctrl"><button type="button" data-fam="all">All</button><button type="button" data-fam="none">None</button></div>
        <ul class="fx-list" id="fams">${d.families.map(f => `<li title="${esc(f.text)}"><label><i style="--c:${f.color}"></i>
          <span>${esc(f.name)}</span><em>${counts[f.name] || 0}</em><input type="checkbox" checked value="${esc(f.name)}"></label></li>`).join('')}</ul>
        <div class="peek-wrap"><h3>Die</h3><div class="peek" id="peek"></div></div>
      </aside>
    </div>
    <div class="cards-head"><h2>All figures</h2><span id="shown"></span></div>
    <div class="cards" id="cards">${d.examples.map(cardHTML).join('')}</div>
    <p class="empty" id="empty" hidden>No figure matches. Clear the search or turn a family back on.</p>`;
  peek(null);

  $('#fams').addEventListener('change', e => {
    if (e.target.checked) S.off.delete(e.target.value); else S.off.add(e.target.value);
    applyFilter();
  });
  $('#gallery .ctrl').addEventListener('click', e => {
    const b = e.target.closest('[data-fam]');
    if (!b) return;
    S.off = b.dataset.fam === 'all' ? new Set() : new Set(d.families.map(f => f.name));
    applyFilter();
  });
  const cards = $('#cards');
  cards.addEventListener('pointerover', e => {
    const c = e.target.closest('.card');
    if (!c) return;
    peek(S.byId[c.dataset.id]);
    S.wafer?.highlight(c.dataset.id);
  });
  cards.addEventListener('pointerleave', () => S.wafer?.highlight(null));
}

function peek(ex) {
  $('#peek').innerHTML = ex
    ? `<img src="${ex.images[S.style].thumb}" alt=""><b><span class="num">${ex.num}</span> ${esc(ex.title)}</b>
       <p>${esc(ex.caption)}</p><div class="fns">${ex.functions.map(f => `<code>${esc(f)}</code>`).join('')}</div>`
    : '<p>Point at a die or a card to see its figure here.</p>';
}

function matches(e, terms) {
  const hay = `${e.id} ${e.title} ${e.family} ${e.caption} ${e.functions.join(' ')}`.toLowerCase();
  return terms.every(t => hay.includes(t));
}

// ---- left sidebar: every example by family; it follows the route and the search box
function buildNav() {
  const d = S.data;
  $('#sidenav').innerHTML = `<a class="all" href="#/gallery">All ${d.examples.length} figures</a>` + d.families.map(f => {
    const list = d.examples.filter(e => e.family === f.name);
    return `<section><h3><i style="--c:${f.color}"></i>${esc(f.name)}<small>${list.length}</small></h3><ul>${list.map(e =>
      `<li data-id="${e.id}"><a class="ex" href="#/example/${e.id}"><span class="num">${e.num}</span><span>${esc(e.title)}</span></a></li>`
    ).join('')}</ul></section>`;
  }).join('');
}

function syncNav() {
  const { view, arg } = currentView(), nav = $('#sidenav');
  $$('a', nav).forEach(a => a.removeAttribute('aria-current'));
  const a = view === 'example' ? $(`li[data-id="${arg}"] a`, nav) : view === 'gallery' ? $('.all', nav) : null;
  if (!a) return;
  a.setAttribute('aria-current', 'page');
  const r = a.getBoundingClientRect(), n = nav.getBoundingClientRect();   // scroll the sidebar only, never the page
  if (r.top < n.top || r.bottom > n.bottom) nav.scrollTop += r.top - n.top - n.height / 3;
}

function setNav(open) {
  $('#sidenav').classList.toggle('open', open);
  $('#navtoggle').setAttribute('aria-expanded', open);
}

function applyFilter() {
  const terms = S.q.toLowerCase().split(/\s+/).filter(Boolean);
  const ex = S.data.examples;
  const ids = new Set(ex.filter(e => !S.off.has(e.family) && matches(e, terms)).map(e => e.id));
  const hits = new Set(ex.filter(e => matches(e, terms)).map(e => e.id));   // the sidebar ignores the family checkboxes
  $$('#sidenav li').forEach(li => { li.hidden = !hits.has(li.dataset.id); });
  $$('#sidenav section').forEach(s => { s.hidden = !$$('li', s).some(li => !li.hidden); });
  $$('#cards .card').forEach(c => { c.hidden = !ids.has(c.dataset.id); });
  $$('#fams li').forEach(li => {
    const box = $('input', li);
    box.checked = !S.off.has(box.value);
    li.classList.toggle('off', !box.checked);
  });
  $('#shown').textContent = ids.size === ex.length ? `${ex.length} figures` : `${ids.size} of ${ex.length} shown`;
  $('#empty').hidden = ids.size > 0;
  S.wafer?.setFilter(ids);
}

async function initWafer() {
  const stage = $('#stage'), tip = $('#tip');
  try {
    const { createWafer } = await import(`./wafer.js?v=${S.data.build}`);   // the stamp changes when the scripts do
    S.wafer = createWafer(stage, {
      examples: S.data.examples, style: S.style,
      onHover(ex, e) {
        peek(ex);
        $$('#cards .card.hot').forEach(c => c.classList.remove('hot'));
        if (!ex || !e) { tip.hidden = true; return; }
        $(`#cards .card[data-id="${ex.id}"]`)?.classList.add('hot');
        const r = stage.getBoundingClientRect();
        tip.innerHTML = `<b>${ex.num}</b>${esc(ex.title)}`;
        tip.hidden = false;
        const x = e.clientX - r.left, flip = x > r.width - 260;
        tip.style.left = `${flip ? x - 24 - tip.offsetWidth : x}px`;
        tip.style.top = `${e.clientY - r.top}px`;
      },
      onPick(ex) { location.hash = `#/example/${ex.id}`; },
    });
    S.gl = 'WebGL, three.js r160';
    $('#spin').checked = !still;
    $('#spin').addEventListener('change', e => S.wafer.setSpin(e.target.checked));
    $('#home').addEventListener('click', () => S.wafer.resetView());
    applyFilter();
  } catch (err) {
    console.warn('3D wafer unavailable:', err);
    S.gl = 'no WebGL: grid only';
    stage.classList.add('flat');
    stage.innerHTML = '<div class="nogl"><p>The 3D wafer needs WebGL and three.js, and neither loaded here.<br>Every figure is in the grid below.</p></div>';
    stage.style.height = '120px';
    stage.style.minHeight = '0';
  } finally {
    stage.classList.remove('loading');
    status();
  }
}

// ---- example page ----------------------------------------------------------------------------

function codeFor(ex, tab) {
  const st = S.style, file = `${ex.id}.py`;
  if (tab === 'py') return ex.code;
  if (tab === 'tex') {
    return ex.latex.replace('[field,figdir', `[${st},figdir`).replace('% or [folio,...]', `% or [${other(st)},...]`);
  }
  return [
    '# from a clone of the repository',
    'git clone https://github.com/BoltzmannEntropy/scqfig.git && cd scqfig',
    'pip install -e .',
    '',
    `# this figure in the ${st} style: writes ${ex.id}_${st}.pdf and .png`,
    `python examples/${file}${st === 'folio' ? ' --style folio' : ''}`,
    '',
    '# with LaTeX setting the text instead of matplotlib mathtext',
    `python examples/${file} --style ${st} --usetex`,
    '',
    '# every example in both styles, the contact sheets and gallery.pdf',
    'python examples/build_gallery.py',
  ].join('\n');
}
const TAB_LANG = { py: 'py', tex: 'tex', run: 'sh' };

function renderExample(id) {
  const list = S.data.examples, i = list.findIndex(e => e.id === id), page = $('#page');
  if (i < 0) { renderMissing(); return; }
  const ex = list[i], st = S.style, im = ex.images[st], ot = other(st);
  const prev = list[(i - 1 + list.length) % list.length], next = list[(i + 1) % list.length];
  S.lastId = id;
  document.title = `${ex.title} · scqfig`;
  const lines = ex.code.split('\n').length;
  page.innerHTML = `
    <div class="crumbs"><a href="#/gallery">Gallery</a><span>/</span>${famDot(ex.family)}<span>/</span><span class="num">${ex.num}</span></div>
    <div class="ex-grid${ex.page ? ' wide' : ''}">
      <div class="figpane">
        <h1>${esc(ex.title)}</h1>
        <div class="figcard"><img src="${im.png}" width="${im.w}" height="${im.h}" style="max-width:${Math.round(im.w * 0.8)}px"
          alt="${esc(ex.title)}, ${st} style. ${esc(ex.caption)}"></div>
        <div class="figbar">${segHTML()}
          <span class="badge${ex.page ? ' page' : ''}">${ex.page ? 'page width' : 'column width'}, ${im.width_in} in</span>
          <a class="dl" href="${im.pdf}" download>PDF</a><a class="dl" href="${im.png}" download>PNG</a></div>
        <p class="cap">${esc(ex.caption)}</p>
        <div class="fns"><span>Draws with</span>${ex.functions.map(f => `<code>${esc(f)}</code>`).join('')}</div>
        <button type="button" class="other" data-style="${ot}"><img src="${ex.images[ot].thumb}" alt="">
          <span><b>Same figure in ${ot}</b>Switch every figure on the site to the ${ot} style</span></button>
      </div>
      <div class="codepane">
        <div class="codetabs" role="tablist" aria-label="Code for this figure">
          ${[['py', 'Python', `${lines} lines`], ['tex', 'LaTeX', 'include'], ['run', 'Run', 'commands']].map(([k, l, s]) =>
            `<button type="button" class="tab" role="tab" data-tab="${k}" aria-selected="${S.tab === k}">${l}<small>${s}</small></button>`).join('')}
          <button type="button" class="copy" id="copycode">Copy</button>
        </div>
        <pre class="code" tabindex="0"><code id="code"></code></pre>
      </div>
    </div>
    <nav class="pager" aria-label="Other figures">
      <a href="#/example/${prev.id}"><small>previous</small>${prev.num} ${esc(prev.title)}</a>
      <a href="#/example/${next.id}" style="text-align:right"><small>next</small>${next.num} ${esc(next.title)}</a>
    </nav>`;
  const show = () => {
    $('#code').innerHTML = hl(codeFor(ex, S.tab), TAB_LANG[S.tab]);
    $$('.codetabs [data-tab]', page).forEach(b => b.setAttribute('aria-selected', b.dataset.tab === S.tab));
  };
  show();
  $('.codetabs', page).addEventListener('click', e => {
    const b = e.target.closest('[data-tab]');
    if (!b) return;
    S.tab = b.dataset.tab;
    store.set('scqfig.tab', S.tab);
    show();
  });
  $('#copycode').addEventListener('click', e => copy(codeFor(ex, S.tab), e.currentTarget));
}

function renderMissing() {
  document.title = 'Not found · scqfig';
  $('#page').innerHTML = '<div class="lead"><h1>No such figure</h1><p>The link names an example that is not in this build. <a href="#/gallery">Back to the gallery</a>.</p></div>';
}

// ---- LaTeX package page ----------------------------------------------------------------------

function renderLatex() {
  const d = S.data, st = S.style;
  document.title = 'LaTeX package · scqfig';
  const uniq = a => [...new Set(a)];
  const roles = uniq([...d.sty.matchAll(/\\colorlet\{(scq[A-Za-z0-9]+)\}/g)].map(m => m[1]));
  const palette = uniq([...d.sty.matchAll(/\\definecolor\{(scq[A-Za-z0-9]+)\}/g)].map(m => m[1]))
    .filter(n => !/^scq(Field|Folio)/.test(n) && !roles.includes(n));
  const has = k => d.sty.includes(k);
  const rows = [
    ['\\scqgraphics[opts]{name}', 'Loads <code>&lt;figdir&gt;/&lt;style&gt;/name</code>, so one package option switches every figure.'],
    ['scqfig', 'pgfplots axis in the house style: frame, ticks, fonts, series colours and markers.'],
    ['scqfig data, scqfig model, scqfig ref', 'Plot styles for measured points, fits and theory, and guide lines.'],
    ['scq tile=<kind>', 'TikZ node on the pale element tile: capacitive, nonlinear, inductive or coupling.'],
  ];
  if (has('scqfig circuit')) rows.push(['scqfig circuit', 'quantikz preset: gate tiles, white meters, slices in the model colour.'],
    ['scq slice label', 'Tilted slice label, as in the matplotlib circuits.']);
  $('#page').innerHTML = `
    <div class="lead"><h1>The LaTeX package</h1>
      <p><code>scqfig.sty</code> carries the same palette, colour roles and plot styles as the Python library. It is generated from
      <code>themes.py</code>, so the two cannot drift. Pick <code>field</code> or <code>folio</code> once in the preamble, and figures,
      native pgfplots axes, TikZ tiles, quantikz circuits and coloured text all follow.</p>
      <div class="actions"><a class="btn" href="downloads/scqfig.sty" download>Download scqfig.sty</a>
        <a class="btn quiet" href="downloads/scqfig-gallery.pdf">Gallery PDF</a></div></div>
    <div class="ref">
      <div class="panel"><h3>Options</h3><table class="kv">
        <tr><td><code>field</code></td><td>Sans text, closed box, filled data, blue fits. The default.</td></tr>
        <tr><td><code>folio</code></td><td>Serif text, open L frame, open data, black fits.</td></tr>
        <tr><td><code>figdir=&lt;dir&gt;</code></td><td>Folder holding <code>field/</code> and <code>folio/</code> renders. Default <code>figures</code>.</td></tr>
      </table></div>
      <div class="panel"><h3>Commands and styles</h3><table class="kv">
        ${rows.map(([k, v]) => `<tr><td><code>${esc(k)}</code></td><td>${v}</td></tr>`).join('')}</table></div>
      <div class="panel"><h3>Colour names</h3><p class="fns">Roles follow the active style:</p>
        <div class="names">${roles.map(n => `<code>${n}</code>`).join('')}</div>
        <p class="fns" style="margin-top:10px">Fixed palette:</p><div class="names">${palette.map(n => `<code>${n}</code>`).join('')}</div></div>
    </div>
    ${d.latex.map(x => `<section class="demo">
      <div><h2>${esc(x.title)}</h2><p>${esc(x.text)}</p>
        <div class="figcard">${x.images[st] ? `<img src="${x.images[st].png}" alt="${esc(x.title)}, ${st} style, typeset by pdflatex">` : '<p>Not rendered in this build.</p>'}</div></div>
      <div>${codebox(x.code[st], 'tex')}</div></section>`).join('')}
    <details class="src"><summary>scqfig.sty source, ${d.sty.split('\n').length} lines, generated from themes.py</summary>
      <pre><code>${hl(d.sty, 'tex')}</code></pre></details>`;
}

// ---- styles page -----------------------------------------------------------------------------

const sw = (name, hex) => `<div class="sw" title="${esc(name)} ${hex}"><i style="--c:${hex}"></i><span>${esc(name)}<br><code>${hex}</code></span></div>`;
const ramp = stops => `linear-gradient(90deg, ${stops.map(([p, c]) => `${c} ${(p * 100).toFixed(1)}%`).join(', ')})`;
const PAIRS = ['34_coherence_decays', '41_wigner', '72_lumped_circuit', '78_quantum_circuit'];

function renderStyles() {
  const d = S.data;
  document.title = 'Styles · scqfig';
  const panel = name => {
    const t = d.styles[name], n = t.notes;
    return `<div class="panel"><h2>${name}<small>${esc(n.face)}</small></h2>
      <table class="kv">${[['frame', n.frame], ['data and fits', n.data], ['series', n.series], ['sequential map', n.map],
        ['circuit tiles', n.tiles], ['panel letters', n.letters]].map(([k, v]) => `<tr><th>${k}</th><td>${esc(v)}</td></tr>`).join('')}</table>
      <h3 style="margin-top:14px">Roles</h3><div class="swatches">${Object.entries(t.roles).map(([k, v]) => sw(k, v)).join('')}</div>
      <h3>Qubits</h3><div class="swatches">${t.qubits.map((v, i) => sw(`Q${i + 1}`, v)).join('')}</div>
      <h3>Levels</h3><div class="swatches">${t.levels.map((v, i) => sw(`|${i}⟩`, v)).join('')}</div>
      <h3>Series</h3><div class="swatches">${t.series.map((v, i) => sw(`series ${i}`, v)).join('')}</div>
      <h3>Element tiles</h3><div class="swatches">${Object.entries(t.tiles).map(([k, v]) => sw(k, v)).join('')}</div>
      <h3>Chip drawings</h3><div class="swatches">${Object.entries(t.chip).map(([k, v]) => sw(k, v)).join('')}</div>
      <h3>Sequential map</h3><div class="ramp" style="background:${ramp(t.cmap)}"></div><div class="ramp-l"><span>low</span><span>high</span></div>
      <h3>Diverging map</h3><div class="ramp" style="background:${ramp(t.cmap_div)}"></div><div class="ramp-l"><span>negative</span><span>0</span><span>positive</span></div>
    </div>`;
  };
  const pairs = PAIRS.map(id => S.byId[id]).filter(Boolean);
  $('#page').innerHTML = `
    <div class="lead"><h1>Two styles, one palette</h1>
      <p>Both styles draw from four hues, black and greys, and nothing else. <b>field</b> suits sans-serif journals and talks;
      <b>folio</b> suits serif journals and theses. A figure switches style with one argument and keeps its data, layout and size.</p></div>
    <div class="hues">${Object.entries(d.palette).map(([k, v]) => `<div class="hue"><i style="--c:${v}"></i><span>${esc(k)}<code>${v}</code></span></div>`).join('')}</div>
    <div class="styles2">${panel('field')}${panel('folio')}</div>
    <div class="pairs"><h2>The same figure in both</h2>
      ${pairs.map(e => `<div class="pair">${['field', 'folio'].map(s => `<figure><a class="figcard" href="#/example/${e.id}">
        <img src="${e.images[s].png}" alt="${esc(e.title)}, ${s} style" loading="lazy"></a>
        <figcaption>${e.num} ${esc(e.title)}, ${s}</figcaption></figure>`).join('')}</div>`).join('')}</div>`;
}

// ---- start page ------------------------------------------------------------------------------

function renderStart() {
  const r = S.data.readme;
  document.title = 'Start · scqfig';
  const blocks = sec => sec.map(b => ('code' in b ? codebox(b.code, b.lang || guessLang(b.code)) : md(b.text))).join('');
  $('#page').innerHTML = `
    <div class="steps">
      <div class="lead"><h1>Start</h1>${blocks(r.why)}</div>
      <section><h2>Install</h2>${blocks(r.install)}</section>
      <section><h2>Quick start</h2>${blocks(r.quickstart)}</section>
      <section><h2>In your paper</h2>
        <p>Render each figure in both styles into <code>figures/field/</code> and <code>figures/folio/</code>, then choose the style
        once in the preamble. Every <code>\\scqgraphics</code> follows the option, so a journal change costs one word.</p>
        ${codebox('\\usepackage[folio,figdir=figures]{scqfig}\n\n\\begin{figure}[t]\n  \\centering\n  \\scqgraphics[width=\\columnwidth]{35_rb}\n  \\caption{Randomized benchmarking.}\n\\end{figure}', 'tex')}
        <p>The <a href="#/latex">LaTeX page</a> shows the native pgfplots, TikZ and quantikz styles.</p></section>
      <section><h2>Add a figure</h2>
        <p>Each example is one file, <code>examples/NN_name.py</code>. Its docstring holds the caption, <code>TITLE</code> and
        <code>FAMILY</code> name it, and <code>build()</code> returns the figure. Then rebuild the gallery and this site:</p>
        ${codebox('python examples/build_gallery.py\npython tools/build_site.py', 'sh')}</section>
      <section><h2>Downloads</h2><div class="actions">
        <a class="btn" href="downloads/scqfig.sty" download>scqfig.sty</a>
        <a class="btn quiet" href="downloads/scqfig-gallery.pdf">Gallery PDF, every figure in both styles</a></div></section>
      <section><h2>License</h2><p>MIT.</p></section>
    </div>`;
}

// ---- shell: routing, style, search, status --------------------------------------------------

const RENDER = { example: renderExample, latex: renderLatex, styles: renderStyles, start: renderStart };

function currentView() {
  const [view, arg] = location.hash.replace(/^#\/?/, '').split('/');
  return { view: view in RENDER ? view : 'gallery', arg: decodeURIComponent(arg || '') };
}

function route(keepScroll = false) {
  const { view, arg } = currentView(), gal = view === 'gallery', y = window.scrollY;
  $$('.tab[data-view]').forEach(t =>
    t.setAttribute('aria-current', t.dataset.view === view || (view === 'example' && t.dataset.view === 'gallery') ? 'page' : 'false'));
  $('#gallery').hidden = !gal;
  $('#page').hidden = gal;
  if (gal) {
    document.title = 'scqfig';
    const c = S.lastId && $(`#cards .card[data-id="${S.lastId}"]`);
    if (c && !keepScroll) {
      c.scrollIntoView({ block: 'center' });
      c.classList.add('hot');
      setTimeout(() => c.classList.remove('hot'), 1200);
    }
  } else {
    RENDER[view](arg);
  }
  if (keepScroll) window.scrollTo(0, y); else if (!gal) window.scrollTo(0, 0);
  syncNav();
  status();
}

function setStyle(st) {
  if (st === S.style || !['field', 'folio'].includes(st)) return;
  S.style = st;
  store.set('scqfig.style', st);
  $$('[data-style]').forEach(b => { if (b.closest('.seg')) b.setAttribute('aria-pressed', b.dataset.style === st); });
  $$('#cards .card').forEach(c => {
    const e = S.byId[c.dataset.id], img = $('img', c);
    img.src = e.images[st].thumb;
    img.alt = `${e.title}, ${st} style`;
  });
  $('#stagecap') && ($('#stagecap').innerHTML = stageCaption());
  S.wafer?.setStyle(st);
  peek(null);
  if (!$('#page').hidden) route(true);
  status();
}

// Light unless the visitor chose dark; index.html applies a stored "dark" before first paint.
function setTheme(dark, save = true) {
  const root = document.documentElement;
  if (dark) root.dataset.theme = 'dark'; else delete root.dataset.theme;
  if (save) store.set('scqfig.theme', dark ? 'dark' : 'light');
  $('#theme').setAttribute('aria-pressed', dark);
  S.wafer?.setTheme();
}

function status() {
  const d = S.data;
  $('#status').innerHTML = `<span><b>scqfig ${d.version}</b></span><span>${d.examples.length} example figures</span>
    <span>${d.latex.length} native LaTeX demos</span><span>style <b>${S.style}</b></span><span class="r">${esc(S.gl)}</span>`;
}

async function init() {
  try {
    const res = await fetch('data/site.json');
    if (!res.ok) throw new Error(res.status);
    S.data = await res.json();
  } catch (err) {
    $('#status').textContent = `Could not load data/site.json (${err.message}). Serve the docs folder over HTTP.`;
    return;
  }
  const d = S.data;
  d.examples.forEach(e => { S.byId[e.id] = e; });
  d.families.forEach(f => { S.famColor[f.name] = f.color; });
  $('#ver').textContent = `v${d.version}`;
  $('#count').textContent = `${d.examples.length} figures`;
  $$('.appbar [data-style]').forEach(b => b.setAttribute('aria-pressed', b.dataset.style === S.style));
  setTheme(document.documentElement.dataset.theme === 'dark', false);
  const bar = $('.appbar');   // the sticky sidebar starts below the app bar, whose height changes as it wraps
  const barHeight = () => document.documentElement.style.setProperty('--appbar-h', `${bar.offsetHeight}px`);
  barHeight();
  new ResizeObserver(barHeight).observe(bar);
  buildNav();
  buildGallery();
  applyFilter();

  document.addEventListener('click', e => {
    const s = e.target.closest('button[data-style]');
    if (s) { setStyle(s.dataset.style); return; }
    if (e.target.closest('#theme')) { setTheme(document.documentElement.dataset.theme !== 'dark'); return; }
    if (e.target.closest('#navtoggle')) { setNav(!$('#sidenav').classList.contains('open')); return; }
    if (e.target.closest('#sidenav a')) setNav(false);
    const c = e.target.closest('.codebox .copy');
    if (c) copy($('pre', c.parentElement).textContent, c);
  });
  const q = $('#q');
  q.addEventListener('input', () => {
    S.q = q.value;
    if (currentView().view !== 'gallery') location.hash = '#/gallery';
    applyFilter();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && $('#sidenav').classList.contains('open')) { setNav(false); $('#navtoggle').focus(); return; }
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === '/') { e.preventDefault(); q.focus(); return; }
    if (currentView().view === 'example' && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      const a = $$('.pager a')[e.key === 'ArrowLeft' ? 0 : 1];
      if (a) location.hash = a.getAttribute('href');
    }
  });
  window.addEventListener('hashchange', () => route());
  route();
  initWafer();
}

init();

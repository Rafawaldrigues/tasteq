'use strict';

// ---------------------------------------------------------------- helpers

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const saved = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
  remove(key) { try { localStorage.removeItem(key); } catch { /* private mode */ } },
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (value !== false && value != null) node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}

async function api(path, { method = 'GET', body } = {}) {
  const response = await fetch('/api' + path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    let detail = '';
    try { detail = (await response.json()).detail; } catch { /* not JSON */ }
    const error = new Error(typeof detail === 'string' && detail ? detail : `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function showError(container, error) {
  const box = $('.error', container);
  box.textContent = error ? (error.message || String(error)) : '';
  box.hidden = !error;
}

// The browser's own validation bubbles and alert boxes speak the browser's
// language, not the site's; every message the visitor sees comes from here.
const REQUIRED_MESSAGES = {
  device: 'Please choose the device you will listen on.',
  mode: 'Please choose how long you have.',
  adult: 'Please confirm that you are 18 or older.',
  consent: 'Please agree to take part to continue.',
  difficulty: 'Please rate how hard it was to choose.',
  fatigue: 'Please rate how tired your ears were.',
  experience: 'Please tell us your experience with music or audio.',
  uses_eq: 'Please tell us whether you use an equalizer.',
};

function validate(form) {
  const invalid = $$('input, textarea', form).find((input) => !input.checkValidity());
  if (!invalid) { showError(form, null); return true; }
  showError(form, new Error(REQUIRED_MESSAGES[invalid.name] || 'Please fill in this field.'));
  const target = invalid.closest('fieldset, label') || invalid;
  target.scrollIntoView({ block: 'center', behavior: 'smooth' });
  invalid.focus({ preventScroll: true });
  return false;
}

let toastTimer = null;
function notify(message) {
  const toast = $('.toast');
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 6000);
}

// ---------------------------------------------------------------- state

let config = null;
let code = saved.get('tasteq_code');
let status = null;
let question = null;
let lastScreen = 'welcome';
const introSeen = {};
const tracksById = {};

const PHASES = {
  screening: {
    eyebrow: 'Part 1 of 3',
    title: 'Instrumental loops',
    name: 'Part 1 · Loops',
    body: (n) => `First, ${n} pairs of short instrumental loops made for this study (no real songs yet). They tell the model about your general taste: more or less bass, more or less brightness, and so on.`,
  },
  music: {
    eyebrow: 'Part 2 of 3',
    title: 'Your music',
    name: 'Part 2 · Your music',
    body: (n) => `Now ${n} pairs of excerpts from songs in the styles you chose. The model starts from what it learned in part 1 and adapts to each song.`,
  },
  blind: {
    eyebrow: 'Part 3 of 3',
    title: 'Blind test',
    name: 'Part 3 · Blind test',
    body: (n) => `Finally, ${n} pairs that check what the model learned. One version always uses the EQ learned for you; the other uses no EQ, what the model predicted from part 1 alone, or a random EQ of the same strength. You are not told which is which, and these answers do not change the model. Some songs may be new to you.`,
  },
};

const PAIR_LABELS = {
  flat: 'no EQ',
  phase1: 'the EQ predicted from part 1 alone',
  random: 'a random EQ of the same strength',
};

// ---------------------------------------------------------------- audio

// Both versions play in sync, looped; A/B switches with a short crossfade at
// the same position, so the listener compares the same moment of the excerpt.
const player = {
  ctx: null,
  buffers: [],
  sources: [],
  gains: [],
  current: -1,
  startedAt: 0,
  offset: 0,
  segmentStart: 0,
  stats: null,
  onchange: () => {},

  context() {
    if (!this.ctx) {
      const Context = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Context();
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  },

  async load(urls) {
    this.stop();
    this.reference = false;
    this.buffers = [];
    this.stats = null;
    const ctx = this.context();
    const id = (this.loadId = (this.loadId || 0) + 1);
    const buffers = await Promise.all(urls.map(async (url) => {
      const response = await fetch(url);
      if (!response.ok) throw Object.assign(new Error('Audio is no longer available'), { status: response.status });
      const data = await response.arrayBuffer();
      return new Promise((resolve, reject) => ctx.decodeAudioData(data, resolve, reject));
    }));
    // A newer load (the next question, or leaving the volume page) wins.
    if (id !== this.loadId) throw Object.assign(new Error('superseded'), { superseded: true });
    this.buffers = buffers;
    this.offset = 0;
    this.stats = { plays: this.buffers.map(() => 0), switches: 0, listen_ms: 0 };
  },

  play(index) {
    if (!this.buffers[index]) return;
    const ctx = this.context();
    if (this.current === index) { this.stop(); return; }
    if (this.current === -1) {
      this.gains = [];
      this.sources = this.buffers.map((buffer, i) => {
        const source = ctx.createBufferSource();
        const gain = ctx.createGain();
        source.buffer = buffer;
        source.loop = true;
        gain.gain.value = i === index ? 1 : 0;
        source.connect(gain).connect(ctx.destination);
        this.gains[i] = gain;
        return source;
      });
      const when = ctx.currentTime + 0.02;
      this.sources.forEach((source) => source.start(when, this.offset));
      this.startedAt = when - this.offset;
      this.segmentStart = when;
    } else {
      // Short linear crossfade that ends exactly at 0 and 1: the other version
      // is fully silent (an exponential approach never quite reaches 0).
      const t = ctx.currentTime;
      this.gains.forEach((gain, i) => {
        const param = gain.gain;
        param.cancelScheduledValues(t);
        param.setValueAtTime(param.value, t);
        param.linearRampToValueAtTime(i === index ? 1 : 0, t + 0.012);
      });
      this.stats.switches += 1;
    }
    this.current = index;
    this.stats.plays[index] += 1;
    this.onchange();
  },

  stop() {
    if (this.current !== -1 && this.ctx) {
      const duration = this.buffers[0].duration;
      this.offset = (this.ctx.currentTime - this.startedAt) % duration;
      this.stats.listen_ms += Math.max(0, Math.round(1000 * (this.ctx.currentTime - this.segmentStart)));
      this.sources.forEach((source) => { try { source.stop(); } catch { /* already stopped */ } });
    }
    this.sources = [];
    this.current = -1;
    this.onchange();
  },
};

// ---------------------------------------------------------------- screens

function show(name) {
  player.stop();
  $$('.screen').forEach((screen) => { screen.hidden = screen.dataset.screen !== name; });
  if (name !== 'credits') lastScreen = name;
  const inStudy = Boolean(code && status && !['done', 'withdrawn'].includes(status.stage));
  $('.leave-wrap').hidden = !inStudy;
  $('.progress').hidden = !(inStudy && status.stage !== 'volume');
  $$('.code').forEach((node) => { node.textContent = code || ''; });
  window.scrollTo(0, 0);
}

function updateProgress() {
  if (!status) return;
  const fraction = status.total ? status.answered / status.total : 0;
  $('.progress-bar span').style.width = `${Math.round(100 * fraction)}%`;
  $('.progress-label').textContent = `${status.answered} / ${status.total}`;
  // Stops on the journey line: where part 2 and the blind test begin.
  const mode = config && status.mode ? config.modes[status.mode] : null;
  const stops = $('.stops');
  stops.textContent = '';
  if (mode && status.total) {
    const c = mode.counts;
    [['Loops', 0], ['Your music', c.screening], ['Blind test', c.screening + c.music]].forEach(([label, at]) => {
      const stop = el('li', { class: status.answered >= at ? 'reached' : '' }, el('span', { text: label }));
      stop.style.left = `${(100 * at) / status.total}%`;
      stops.append(stop);
    });
  }
}

async function route() {
  updateProgress();
  const stage = status.stage;
  if (stage === 'volume') return show('volume');
  if (stage === 'styles') return show('styles');
  if (stage === 'questionnaire') return show('questionnaire');
  if (stage === 'done') return showResults();
  if (stage === 'withdrawn') return showLeft(false);
  if (PHASES[stage]) {
    if (status.position === 0 && !introSeen[stage]) return showIntro(stage);
    return showQuestion();
  }
}

function showIntro(stage) {
  const phase = PHASES[stage];
  const screen = $('[data-screen="intro"]');
  $('.meta', screen).textContent = phase.eyebrow;
  $('h1', screen).textContent = phase.title;
  $('.body', screen).textContent = phase.body(status.stage_total);
  screen.dataset.stage = stage;
  show('intro');
}

// ---------------------------------------------------------------- welcome

function renderWelcome() {
  const contact = $('.contact');
  contact.textContent = '';
  if (config.contact.email) {
    contact.append(el('a', { href: `mailto:${config.contact.email}`, text: config.contact.email }));
    if (config.contact.url) contact.append(' or ');
  }
  if (config.contact.url) {
    contact.append(el('a', { href: config.contact.url, target: '_blank', rel: 'noopener', text: 'the project page' }));
  }

  const devices = $('#devices');
  Object.entries(config.devices).forEach(([key, label], i) => {
    devices.append(el('label', {}, el('input', { type: 'radio', name: 'device', value: key, required: i === 0 }), label));
  });

  const modes = $('#modes');
  Object.entries(config.modes).forEach(([key, mode], i) => {
    const c = mode.counts;
    modes.append(el('label', { class: 'mode' },
      el('input', { type: 'radio', name: 'mode', value: key, required: i === 0, checked: i === 0 }),
      el('strong', { text: mode.label }),
      el('span', { class: 'minutes', text: `~${mode.minutes} min` }),
      el('span', { class: 'detail', text: `${c.screening + c.music + c.blind} pairs · ${c.screening} loops, ${c.music} songs, ${c.blind} blind` }),
    ));
  });

  const styles = $('#styles');
  config.styles.filter((s) => s.n_tracks > 0).forEach((style) => {
    styles.append(el('label', {},
      el('input', { type: 'checkbox', name: 'styles', value: style.id }),
      style.label, el('span', { class: 'count', text: `${style.n_tracks} songs` })));
  });
  $('.min-styles').textContent = config.min_styles;

  $$('.scale').forEach((scale) => {
    scale.append(el('span', { class: 'end', text: scale.dataset.low }));
    for (let v = 1; v <= 5; v++) {
      scale.append(el('label', {}, el('input', { type: 'radio', name: scale.dataset.name, value: v, required: v === 1 }), String(v)));
    }
    scale.append(el('span', { class: 'end', text: scale.dataset.high }));
  });

  config.tracks.forEach((track) => { tracksById[track.id] = track; });
  renderCredits();
}

$('#consent-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  if (!validate(form)) return;
  const data = new FormData(form);
  const button = $('button[type="submit"]', form);
  button.disabled = true;
  try {
    status = await api('/participants', {
      method: 'POST',
      body: { mode: data.get('mode'), device: data.get('device'), consent: data.has('consent'), adult: data.has('adult') },
    });
    code = status.code;
    saved.set('tasteq_code', code);
    showError(form, null);
    route();
  } catch (error) {
    showError(form, error);
  } finally {
    button.disabled = false;
  }
});

$('#resume-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const box = event.currentTarget.closest('details');
  const value = new FormData(event.currentTarget).get('code').trim().toUpperCase();
  if (!value) { showError(box, new Error('Please enter your code, like K7QF-2MZD.')); return; }
  try {
    status = await api(`/p/${encodeURIComponent(value)}`);
    code = status.code;
    saved.set('tasteq_code', code);
    showError(box, null);
    route();
  } catch (error) {
    showError(box, error.status === 404 ? new Error('No participant with this code.') : error);
  }
});

// ---------------------------------------------------------------- volume

async function playReference() {
  const button = $('[data-action="ref"]');
  if (player.current !== -1) { player.stop(); return; }
  try {
    if (!player.reference) {
      button.disabled = true;
      button.textContent = 'Loading…';
      await player.load(['/api/reference']);
      player.reference = true;
    }
    player.play(0);
  } catch (error) {
    if (!error.superseded) notify(`Could not load the reference: ${error.message}`);
  } finally {
    button.disabled = false;
    player.onchange(); // the label was frozen while loading
  }
}

// ---------------------------------------------------------------- questions

function creditLine(track) {
  const line = document.createDocumentFragment();
  line.append(`by ${track.artist} · `);
  line.append(el('a', { href: track.license_url, target: '_blank', rel: 'noopener', text: track.license }));
  line.append(' · ');
  line.append(el('a', { href: track.fma_url, target: '_blank', rel: 'noopener', text: 'Free Music Archive' }));
  line.append(' · excerpt modified (equalized)');
  return line;
}

async function showQuestion() {
  const screen = $('[data-screen="question"]');
  show('question');
  setAnswering(false);
  $('.loading', screen).hidden = false;
  $('.loading-text', screen).textContent = 'Preparing the pair…';
  if (PHASES[status.stage]) $('.phase-name', screen).textContent = PHASES[status.stage].name;
  $('.track-title', screen).textContent = '\u00a0';
  $('.track-credit', screen).textContent = '';
  $$('.version', screen).forEach((button) => { button.disabled = true; button.classList.remove('heard'); });
  try {
    question = await api(`/p/${code}/question`, { method: 'POST' });
    status = question.status;
    updateProgress();
    const phase = PHASES[question.stage];
    $('.phase-name', screen).textContent = `${phase.name} · ${question.position + 1} of ${question.stage_total}`;
    const credit = $('.track-credit', screen);
    credit.textContent = '';
    const track = tracksById[question.track.catalog_id];
    if (track) {
      $('.track-title', screen).textContent = track.title;
      credit.append(creditLine(track));
    } else {
      $('.track-title', screen).textContent = question.track.name;
      credit.textContent = 'Instrumental loop, made for this study';
    }
    await player.load(question.audio);
    $$('.version', screen).forEach((button) => { button.disabled = false; });
    $('.loading', screen).hidden = true;
    $('.hint', screen).textContent = 'Listen to both versions to answer.';
    player.onchange();
  } catch (error) {
    if (error.superseded) return;
    $('.loading-text', screen).textContent = `Something went wrong: ${error.message}. Retrying…`;
    setTimeout(showQuestion, 2500);
  }
}

function setAnswering(enabled) {
  $$('.answer').forEach((button) => { button.disabled = !enabled; });
}

player.onchange = () => {
  const screen = $('[data-screen="question"]');
  $$('.version', screen).forEach((button, i) => {
    const playing = player.current === i;
    button.setAttribute('aria-pressed', String(playing));
    $('.state', button).textContent = playing ? 'Playing' : 'Play';
    button.classList.toggle('heard', Boolean(player.stats && player.stats.plays[i] > 0));
  });
  const ref = $('[data-action="ref"]');
  if (ref && !ref.disabled) {
    ref.textContent = player.current === -1 ? 'Play reference' : 'Stop';
    ref.dataset.playing = String(player.current !== -1);
  }
  const volume = $('[data-screen="volume"]');
  if (volume) volume.dataset.playing = String(player.current !== -1 && Boolean(player.reference));
  if (question && player.stats && player.stats.plays.length === 2) {
    const both = player.stats.plays.every((n) => n > 0);
    if (both && $('.answer').disabled && $('.loading', screen).hidden) {
      setAnswering(true);
      $('.hint', screen).textContent = 'Keys: 1 / 2 switch, space stops.';
    }
  }
};

async function answer(value) {
  if (!question) return;
  const screen = $('[data-screen="question"]');
  const stats = player.stats;
  player.stop();
  setAnswering(false);
  $$('.version', screen).forEach((button) => { button.disabled = true; });
  $('.loading', screen).hidden = false;
  $('.loading-text', screen).textContent = 'Saving… the model is choosing the next pair.';
  const qid = question.qid;
  question = null;
  try {
    status = await api(`/p/${code}/answer`, {
      method: 'POST',
      body: { qid, answer: value, plays_first: stats.plays[0], plays_second: stats.plays[1], switches: stats.switches, listen_ms: stats.listen_ms },
    });
    route();
  } catch (error) {
    if (error.status === 409) return showQuestion();
    $('.loading-text', screen).textContent = `Could not save: ${error.message}`;
    setTimeout(() => api(`/p/${code}`).then((s) => { status = s; route(); }), 2500);
  }
}

$$('.version').forEach((button) => button.addEventListener('click', () => player.play(Number(button.dataset.version))));
$$('.answer').forEach((button) => button.addEventListener('click', () => answer(button.dataset.answer)));

document.addEventListener('keydown', (event) => {
  if (event.target.closest('input, textarea, select') || event.metaKey || event.ctrlKey || event.altKey) return;
  if ($('[data-screen="question"]').hidden || !player.buffers.length || !question) return;
  const key = event.key.toLowerCase();
  if (key === '1' || key === 'a') player.play(0);
  else if (key === '2' || key === 'b') player.play(1);
  else if (key === ' ') { event.preventDefault(); player.stop(); }
  else return;
});

// ---------------------------------------------------------------- styles + questionnaire

$('#styles-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const styles = new FormData(form).getAll('styles');
  if (styles.length < config.min_styles) {
    showError(form, new Error(`Please pick at least ${config.min_styles} styles.`));
    return;
  }
  const button = $('button[type="submit"]', form);
  button.disabled = true;
  button.textContent = 'Preparing your songs…';
  try {
    status = await api(`/p/${code}/styles`, { method: 'POST', body: { styles } });
    showError(form, null);
    route();
  } catch (error) {
    showError(form, error);
  } finally {
    button.disabled = false;
    button.textContent = 'Continue';
  }
});

$('#questionnaire-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  if (!validate(form)) return;
  const data = new FormData(form);
  try {
    status = await api(`/p/${code}/questionnaire`, {
      method: 'POST',
      body: {
        difficulty: Number(data.get('difficulty')),
        fatigue: Number(data.get('fatigue')),
        experience: data.get('experience'),
        uses_eq: data.get('uses_eq'),
        comment: data.get('comment') || '',
      },
    });
    showError(form, null);
    route();
  } catch (error) {
    showError(form, error);
  }
});

// ---------------------------------------------------------------- results

function chart(freqs, curves) {
  const W = 640, H = 260, L = 38, R = 10, T = 12, B = 26;
  const limit = Math.max(12, Math.ceil(Math.max(...curves.flatMap((c) => c.values.map(Math.abs))) / 3) * 3);
  const x = (f) => L + (W - L - R) * Math.log(f / 20) / Math.log(1000);
  const y = (db) => T + (H - T - B) * (limit - db) / (2 * limit);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'EQ curve: gain in dB by frequency');
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text) node.textContent = text;
    svg.append(node);
    return node;
  };
  for (let db = -limit; db <= limit; db += limit / 2) {
    add('line', { x1: L, x2: W - R, y1: y(db), y2: y(db), class: db === 0 ? 'zero' : 'grid' });
    add('text', { x: L - 6, y: y(db) + 4, 'text-anchor': 'end', class: 'axis' }, `${db > 0 ? '+' : ''}${db}`);
  }
  [[31, '31'], [63, '63'], [125, '125'], [250, '250'], [500, '500'], [1000, '1k'], [2000, '2k'], [4000, '4k'], [8000, '8k'], [16000, '16 kHz']].forEach(([f, label]) => {
    add('line', { x1: x(f), x2: x(f), y1: T, y2: H - B, class: 'grid' });
    add('text', { x: x(f), y: H - 8, 'text-anchor': 'middle', class: 'axis' }, label);
  });
  curves.forEach((c) => {
    add('polyline', { points: c.values.map((v, i) => `${x(freqs[i]).toFixed(1)},${y(v).toFixed(1)}`).join(' '), class: c.cls });
  });
  return svg;
}

async function showResults() {
  show('results');
  const screen = $('[data-screen="results"]');
  let results;
  try {
    results = await api(`/p/${code}/results`);
  } catch (error) {
    $('.verdict', screen).textContent = `Could not load results: ${error.message}`;
    return;
  }
  // Verdict: learned EQ against every alternative, pooled.
  const total = results.blind.reduce((acc, r) => ({ wins: acc.wins + r.wins, losses: acc.losses + r.losses, ties: acc.ties + r.ties }), { wins: 0, losses: 0, ties: 0 });
  const verdict = $('.verdict', screen);
  verdict.textContent = '';
  verdict.append(
    'In the blind test you picked ',
    el('strong', { text: `your learned EQ ${total.wins} time${total.wins === 1 ? '' : 's'}` }),
    `, the alternative ${total.losses} time${total.losses === 1 ? '' : 's'}, and heard no difference ${total.ties} time${total.ties === 1 ? '' : 's'}. `,
    `The model learned from ${results.n_comparisons} of your answers.`,
  );

  const tabs = $('.track-tabs', screen);
  const holder = $('.chart', screen);
  tabs.textContent = '';
  const draw = (item, button) => {
    $$('button', tabs).forEach((b) => b.setAttribute('aria-selected', String(b === button)));
    holder.textContent = '';
    holder.append(chart(results.freqs, [
      { values: item.phase1_curve, cls: 'phase1' },
      { values: item.curve, cls: 'learned' },
    ]));
    holder.append(el('div', { class: 'legend' },
      el('span', {}, el('i'), 'Learned for this song'),
      el('span', {}, el('i', { class: 'dash' }), 'Predicted after part 1')));
  };
  results.tracks.forEach((item, i) => {
    const track = tracksById[item.catalog_id];
    const button = el('button', { type: 'button', role: 'tab' }, track ? track.title : `Song ${i + 1}`);
    if (!item.heard_in_music) button.append(el('span', { class: 'new', text: 'new' }));
    button.addEventListener('click', () => draw(item, button));
    tabs.append(button);
    if (i === 0) draw(item, button);
  });

  const table = el('table', {},
    el('thead', {}, el('tr', {},
      el('th', { text: 'Learned EQ vs' }), el('th', { text: 'Learned' }), el('th', { text: 'Other' }),
      el('th', { text: 'Same' }), el('th', { text: 'Preferred' }))));
  const body = el('tbody');
  results.blind.forEach((r) => {
    const rate = r.win_rate == null ? null : Math.round(100 * r.win_rate);
    body.append(el('tr', {},
      el('td', { text: PAIR_LABELS[r.pair[1]] || r.pair[1] }),
      el('td', { class: 'num', text: r.wins }), el('td', { class: 'num', text: r.losses }), el('td', { class: 'num', text: r.ties }),
      el('td', {}, rate == null ? '–' : el('div', { title: `${rate}%` }, el('div', { class: 'bar' }, (() => { const s = el('span'); s.style.width = `${rate}%`; return s; })()), `${rate}%`))));
  });
  table.append(body);
  const holderTable = $('.blind-table', screen);
  holderTable.textContent = '';
  holderTable.append(table);
}

// ---------------------------------------------------------------- leaving

function showLeft(deleted) {
  $('.left-title').textContent = deleted ? 'Your answers were deleted' : 'You left the study';
  $('.left-body').textContent = deleted
    ? 'Everything you answered has been erased. Thank you for trying the study.'
    : `Thank you! The answers you gave are kept anonymously under the code ${code}. Quote it if you later want them removed.`;
  show('left');
  saved.remove('tasteq_code');
}

$('#leave-dialog').addEventListener('close', async () => {
  const choice = $('#leave-dialog').returnValue;
  if (!code || !['keep', 'delete'].includes(choice)) return;
  try {
    if (choice === 'delete') {
      await api(`/p/${code}`, { method: 'DELETE' });
      status = { stage: 'withdrawn' };
      showLeft(true);
    } else {
      status = await api(`/p/${code}/withdraw`, { method: 'POST' });
      showLeft(false);
    }
  } catch (error) {
    notify(`Could not leave: ${error.message}`);
  }
});

// ---------------------------------------------------------------- credits

function renderCredits() {
  const list = $('.credits-list');
  config.tracks.filter((t) => t.available).forEach((track) => {
    const tags = el('div', { class: 'tags' });
    if (track.explicit) tags.append(el('span', { text: 'may be explicit' }));
    if (track.ai_generated) tags.append(el('span', { text: 'AI-generated (per FMA)' }));
    list.append(el('div', { class: 'credit' },
      el('div', { class: 'title' }, `“${track.title}” — ${track.artist}`),
      el('div', { class: 'meta' }, `${track.genre} · `,
        el('a', { href: track.license_url, target: '_blank', rel: 'noopener', text: track.license }), ' · ',
        el('a', { href: track.fma_url, target: '_blank', rel: 'noopener', text: 'source' }),
        ' · equalized excerpts'),
      track.notes ? el('div', { class: 'meta', text: track.notes }) : null,
      tags.childElementCount ? tags : null));
  });
}

// ---------------------------------------------------------------- actions

document.addEventListener('click', (event) => {
  const target = event.target.closest('[data-action]');
  if (!target) return;
  const action = target.dataset.action;
  if (target.tagName === 'A') event.preventDefault();
  if (action === 'home') {
    if (code && status && !['done', 'withdrawn'].includes(status.stage)) {
      $('#resume-form input').value = code;
      $('.resume').open = true;
    }
    show('welcome');
  } else if (action === 'credits') show('credits');
  else if (action === 'back') {
    if (lastScreen === 'question' || lastScreen === 'results') route();
    else show(lastScreen);
  } else if (action === 'leave') {
    player.stop();
    $('#leave-dialog').showModal();
  } else if (action === 'ref') playReference();
  else if (action === 'volume-done') {
    player.stop();
    player.loadId = (player.loadId || 0) + 1;
    player.reference = false;
    api(`/p/${code}/volume`, { method: 'POST' }).then((s) => { status = s; route(); }).catch((error) => notify(error.message));
  } else if (action === 'intro-done') {
    introSeen[$('[data-screen="intro"]').dataset.stage] = true;
    showQuestion();
  }
});

// ---------------------------------------------------------------- start

(async function start() {
  try {
    config = await api('/config');
  } catch (error) {
    $('#app').textContent = `The study is not available right now (${error.message}). Please try again later.`;
    return;
  }
  renderWelcome();
  if (code) {
    try {
      status = await api(`/p/${code}`);
      if (status.stage === 'withdrawn') {
        saved.remove('tasteq_code');
        code = null;
        status = null;
        show('welcome');
      } else {
        route();
      }
      return;
    } catch {
      saved.remove('tasteq_code');
      code = null;
    }
  }
  show('welcome');
})();

const html = htm.bind(React.createElement);
const { useState, useEffect, useRef, useCallback } = React;

const J = (url, body) => fetch(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
}).then(r => r.json());
const G = (url) => fetch(url).then(r => r.json());

// ---------------------------------------------------------------- canvas viewer + HITL
function Viewer({ empiar, stem, picks, factor, brush, tool, onCorrect }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());
  const view = useRef({ z: 1, ox: 0, oy: 0 });
  const drag = useRef(null);
  const R = Math.max(6, Math.round((108 / factor) / 2));

  const applyStyle = () => {
    const c = canvasRef.current; if (!c) return;
    const v = view.current;
    c.style.transform = `translate(${v.ox}px,${v.oy}px) scale(${v.z})`;
  };
  const fit = () => {
    const w = wrapRef.current, c = canvasRef.current;
    if (!w || !c || !c.width) return;
    const z = Math.min(w.clientWidth / c.width, w.clientHeight / c.height) * 0.97;
    view.current = { z, ox: (w.clientWidth - c.width * z) / 2, oy: (w.clientHeight - c.height * z) / 2 };
    applyStyle();
  };
  const zoomAt = (cx, cy, fac) => {
    const v = view.current;
    const ix = (cx - v.ox) / v.z, iy = (cy - v.oy) / v.z;
    const nz = Math.max(0.1, Math.min(25, v.z * fac));
    view.current = { z: nz, ox: cx - ix * nz, oy: cy - iy * nz };
    applyStyle();
  };

  const draw = (sel) => {
    const cv = canvasRef.current, img = imgRef.current;
    if (!cv || !img.complete || !img.naturalWidth) return;
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    const ctx = cv.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const lw = 1.4 / Math.max(view.current.z, 1);
    if (picks) {
      ctx.lineWidth = lw;
      for (let i = 0; i < picks.x.length; i++) {
        const junk = picks.junk[i];
        ctx.strokeStyle = junk ? 'rgba(229,72,77,.85)' : 'rgba(41,195,147,.95)';
        ctx.beginPath(); ctx.arc(picks.x[i], picks.y[i], junk ? R * 0.7 : R, 0, 6.2832); ctx.stroke();
      }
    }
    if (sel) {
      ctx.strokeStyle = brush === 'keep' ? '#29c393' : '#e5484d';
      ctx.fillStyle = brush === 'keep' ? 'rgba(41,195,147,.12)' : 'rgba(229,72,77,.12)';
      ctx.lineWidth = lw; ctx.setLineDash([5, 4]);
      const x = Math.min(sel.x0, sel.x1), y = Math.min(sel.y0, sel.y1);
      ctx.fillRect(x, y, Math.abs(sel.x1 - sel.x0), Math.abs(sel.y1 - sel.y0));
      ctx.strokeRect(x, y, Math.abs(sel.x1 - sel.x0), Math.abs(sel.y1 - sel.y0));
      ctx.setLineDash([]);
    }
  };

  useEffect(() => {
    const img = imgRef.current;
    img.onload = () => { draw(); fit(); };
    img.src = `/api/img/${empiar}/${stem}.png`;
  }, [empiar, stem]);
  useEffect(() => { draw(); }, [picks]);
  useEffect(() => {
    const w = wrapRef.current; if (!w) return;
    const onWheel = (e) => { e.preventDefault(); const r = w.getBoundingClientRect();
      zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.15 : 1 / 1.15); };
    w.addEventListener('wheel', onWheel, { passive: false });
    return () => w.removeEventListener('wheel', onWheel);
  }, []);

  const imgCoord = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (canvasRef.current.width / r.width),
             y: (e.clientY - r.top) * (canvasRef.current.height / r.height) };
  };
  const onDown = (e) => {
    if (tool === 'pan') { drag.current = { pan: true, x: e.clientX, y: e.clientY }; return; }
    const p = imgCoord(e); drag.current = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, moved: false };
  };
  const onMove = (e) => {
    const d = drag.current; if (!d) return;
    if (d.pan) { const v = view.current; v.ox += e.clientX - d.x; v.oy += e.clientY - d.y;
      d.x = e.clientX; d.y = e.clientY; applyStyle(); return; }
    const p = imgCoord(e); d.x1 = p.x; d.y1 = p.y;
    if (Math.abs(p.x - d.x0) + Math.abs(p.y - d.y0) > 4 / view.current.z) d.moved = true;
    draw(d);
  };
  const onUp = () => {
    const d = drag.current; drag.current = null;
    if (!d || d.pan || !picks) { draw(); return; }
    const lo = { x: Math.min(d.x0, d.x1), y: Math.min(d.y0, d.y1) };
    const hi = { x: Math.max(d.x0, d.x1), y: Math.max(d.y0, d.y1) };
    let sel = [];
    if (d.moved) {
      for (let i = 0; i < picks.x.length; i++)
        if (picks.x[i] >= lo.x && picks.x[i] <= hi.x && picks.y[i] >= lo.y && picks.y[i] <= hi.y) sel.push(i);
    } else {
      let best = -1, bd = 1e9;
      for (let i = 0; i < picks.x.length; i++) {
        const dd = (picks.x[i] - d.x1) ** 2 + (picks.y[i] - d.y1) ** 2;
        if (dd < bd) { bd = dd; best = i; }
      }
      if (best >= 0 && bd < (R * 2.5) ** 2) sel = [best];
    }
    if (sel.length) onCorrect(sel, brush);
    draw();
  };

  return html`<div class="canvas-wrap" ref=${wrapRef}
       onMouseDown=${onDown} onMouseMove=${onMove} onMouseUp=${onUp} onMouseLeave=${onUp}
       style=${{ cursor: tool === 'pan' ? 'grab' : 'crosshair' }}>
    <canvas ref=${canvasRef} style=${{ position: 'absolute', transformOrigin: '0 0' }}></canvas>
    <div class="zoomctl">
      <button onClick=${() => { const r = wrapRef.current.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1.3); }}>+</button>
      <button onClick=${() => { const r = wrapRef.current.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1 / 1.3); }}>−</button>
      <button onClick=${fit}>Fit</button>
    </div>
  </div>`;
}

// ---------------------------------------------------------------- plotly helpers
function Plot({ data, layout, h }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    Plotly.react(ref.current, data, {
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: { color: '#7e8da3', size: 10 }, margin: { l: 34, r: 8, t: 8, b: 24 },
      showlegend: false, ...layout
    }, { displayModeBar: false, responsive: true });
  }, [data, layout]);
  return html`<div class="chart" style=${{ height: (h || 150) + 'px' }} ref=${ref}></div>`;
}

function Metric({ k, v, d, big }) {
  return html`<div class=${'card' + (big ? ' big' : '')}>
    <div class="v">${v}</div><div class="k">${k}</div>
    ${d ? html`<div class=${'d ' + (d.up ? 'up' : 'down')}>${d.t}</div>` : null}</div>`;
}

// ---------------------------------------------------------------- app
function App() {
  const [info, setInfo] = useState(null);
  const [idx, setIdx] = useState(0);
  const [picks, setPicks] = useState(null);
  const [met, setMet] = useState(null);
  const [threshold, setTh] = useState(0.5);
  const [mode, setMode] = useState('model');
  const [clfModel, setClf] = useState('lgbm');
  const [brush, setBrush] = useState('junk');
  const [tool, setTool] = useState('select');
  const [f1hist, setF1] = useState([]);
  const [classes, setClasses] = useState(null);
  const [busy2d, setBusy2d] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [timeline, setTimeline] = useState([]);
  const wsRef = useRef(null);

  const empiar = info && info.empiar;
  const mic = info && info.micrographs[idx];
  const stem = mic && mic.stem;

  useEffect(() => { G('/api/state').then(s => { setInfo(s); setTh(s.threshold); setClf(s.clf_model); }); }, []);

  const load = useCallback(() => {
    if (!empiar || !stem) return;
    G(`/api/picks/${empiar}/${stem}`).then(setPicks);
    G(`/api/metrics/${empiar}/${stem}`).then(setMet);
  }, [empiar, stem]);
  useEffect(() => { load(); }, [load]);

  const onCorrect = (sel, b) => {
    const body = { empiar, stem, dump_idx: b === 'keep' ? [] : sel, keep_idx: b === 'keep' ? sel : [] };
    J('/api/correct', body).then(r => {
      setPicks(p => ({ ...p, score: r.score, junk: r.junk }));
      setF1(r.f1_history); G(`/api/metrics/${empiar}/${stem}`).then(setMet);
    });
  };

  const changeTh = (t) => { setTh(t); J('/api/threshold', { empiar, threshold: t }).then(() => load()); };
  const switchMode = (m) => { setMode(m); J('/api/mode', { empiar, mode: m }).then(() => { setF1([]); load(); }); };
  const switchClf = (m) => { setClf(m); J('/api/clf_model', { empiar, clf_model: m }).then(() => load()); };
  const reset = () => { J('/api/reset', { empiar, coldstart: mode === 'learn' }).then(() => { setF1([]); load(); }); };
  const run2d = () => { setBusy2d(true); J('/api/classify2d', { empiar }).then(r => { setClasses(r); setBusy2d(false); }); };
  const savePng = (type) => {
    const c = document.querySelector('.canvas-wrap canvas'); if (!c) return;
    c.toBlob((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${stem}_overlay.${type === 'image/jpeg' ? 'jpg' : 'png'}`;
      a.click(); URL.revokeObjectURL(a.href);
    }, type, 0.92);
  };

  // keyboard k / d / u apply to nothing without a selection; we use brush toggle instead.
  useEffect(() => {
    const h = (e) => { if (e.key === 'k') setBrush('keep'); if (e.key === 'd') setBrush('junk'); };
    window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h);
  }, []);

  const toggleStream = () => {
    if (streaming) { wsRef.current && wsRef.current.close(); setStreaming(false); return; }
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
    ws.onopen = () => ws.send(JSON.stringify({ speed: 0.9 }));
    ws.onmessage = (m) => {
      const d = JSON.parse(m.data);
      setIdx(d.i % info.micrographs.length);
      setTimeline(t => [...t.slice(-60), d.junk_pct]);
    };
    ws.onclose = () => setStreaming(false);
    wsRef.current = ws; setStreaming(true);
  };

  if (!info) return html`<div style=${{ padding: 40, color: '#7e8da3' }}>Loading CryoClear…</div>`;
  const pa = met && met.picking_after, pb = met && met.picking_before;
  const jr = met && met.junk_rejection;

  return html`<div class="app">
    <div class="hdr">
      <div class="brand">Cryo<span>Clear</span></div>
      <div class="sub">real-time particle-picking &amp; junk-triage copilot</div>
      <div class="spacer"></div>
      <div class="pill">dataset <b>EMPIAR-${empiar}</b></div>
      <div class="pill">${info.micrographs.length} micrographs</div>
      <div class="pill"><span class=${'dot ' + (streaming ? 'on' : 'off')}></span>${streaming ? 'streaming' : 'idle'}</div>
    </div>

    <div class="main">
      <div class="viewer">
        <div class="toolbar">
          <button class="sm" onClick=${() => setIdx(i => Math.max(0, i - 1))}>◀ prev</button>
          <select value=${stem} onChange=${e => setIdx(info.micrographs.findIndex(m => m.stem === e.target.value))}
             style=${{ width: '230px' }}>
            ${info.micrographs.map((m, i) => html`<option key=${m.stem} value=${m.stem}>${i + 1}. ${m.stem}</option>`)}
          </select>
          <button class="sm" onClick=${() => setIdx(i => Math.min(info.micrographs.length - 1, i + 1))}>next ▶</button>
          <div style=${{ width: '12px' }}></div>
          <span class="lbl">brush:</span>
          <button class=${'sm junk' + (brush === 'junk' ? ' active' : '')} onClick=${() => setBrush('junk')}>dump <span class="kbd">d</span></button>
          <button class=${'sm keep' + (brush === 'keep' ? ' active' : '')} onClick=${() => setBrush('keep')}>keep <span class="kbd">k</span></button>
          <div style=${{ width: '12px' }}></div>
          <span class="lbl">tool:</span>
          <button class=${'sm' + (tool === 'select' ? ' primary' : '')} onClick=${() => setTool('select')}>select</button>
          <button class=${'sm' + (tool === 'pan' ? ' primary' : '')} onClick=${() => setTool('pan')}>pan</button>
          <div class="spacer"></div>
          <span class="lbl">scroll = zoom · ${tool === 'pan' ? 'drag = pan' : `drag a box to ${brush} · click toggles one`}</span>
        </div>
        <${Viewer} empiar=${empiar} stem=${stem} picks=${picks} factor=${info.factor}
           brush=${brush} tool=${tool} onCorrect=${onCorrect} />
        <div class="legend">
          <span><span class="sw" style=${{ background: '#29c393' }}></span>keep ${picks ? picks.junk.filter(v => !v).length : 0}</span>
          <span><span class="sw" style=${{ background: '#e5484d' }}></span>junk ${picks ? picks.junk.filter(v => v).length : 0}</span>
          <span class="spacer" style=${{ flex: 1 }}></span>
          <span class="tag">${met ? met.n_candidates : 0} candidates</span>
        </div>
      </div>

      <div class="side">
        <div class="group">
          <h4>Scoreboard — picking vs CryoPPP ground truth</h4>
          <div class="cards">
            <${Metric} big k="Picking F1 after junk triage" v=${pa ? pa.f1.toFixed(3) : '—'}
              d=${pa && pb ? { up: pa.f1 >= pb.f1, t: (pa.f1 - pb.f1 >= 0 ? '+' : '') + (pa.f1 - pb.f1).toFixed(3) + ' vs raw picks' } : null} />
            <${Metric} k="Raw picking F1" v=${pb ? pb.f1.toFixed(3) : '—'} />
            <${Metric} k="Junk-rejection F1" v=${jr ? jr.junk_f1.toFixed(3) : '—'} />
            <${Metric} k="Kept particles" v=${met ? met.n_kept : '—'} />
            <${Metric} k="Junk removed" v=${met ? met.junk_pct.toFixed(1) + '%' : '—'} />
          </div>
          ${pa && pb ? html`<${Plot} h=${130} data=${[{
            type: 'bar', x: ['precision', 'recall', 'F1'],
            y: [pb.precision, pb.recall, pb.f1], name: 'raw', marker: { color: '#3a4759' } },
            { type: 'bar', x: ['precision', 'recall', 'F1'], y: [pa.precision, pa.recall, pa.f1],
              name: 'after triage', marker: { color: '#29c393' } }]}
            layout=${{ barmode: 'group', yaxis: { range: [0, 1] } }} />` : null}
        </div>

        <div class="divider"></div>
        <div class="group">
          <h4>Junk threshold</h4>
          <div class="row between"><span class="lbl">keep if junk-prob &lt; ${threshold.toFixed(2)}</span></div>
          <input type="range" min="0.1" max="0.95" step="0.05" value=${threshold}
             onChange=${e => changeTh(parseFloat(e.target.value))} />
        </div>

        <div class="divider"></div>
        <div class="group">
          <h4>Junk classifier</h4>
          <select value=${clfModel} disabled=${mode === 'learn'} onChange=${e => switchClf(e.target.value)}>
            ${Object.entries(info.clf_options).map(([k, v]) =>
              html`<option key=${k} value=${k}>${v.label} · held-out F1 ${v.heldout.toFixed(2)}</option>`)}
          </select>
          <div class="note">Held-out picking F1 (not in-sample). LightGBM (boosted trees) extracts the
            most signal; RandomForest collapses to all-junk; the CNN is comparable. β-gal junk is hard —
            the honest gain is modest until a stronger picker is used.</div>
        </div>

        <div class="divider"></div>
        <div class="group">
          <h4>Learning mode</h4>
          <div class="row">
            <button class=${mode === 'model' ? 'primary sm' : 'sm'} onClick=${() => switchMode('model')}>Trained model</button>
            <button class=${mode === 'learn' ? 'primary sm' : 'sm'} onClick=${() => switchMode('learn')}>Active-learning</button>
            <button class="sm" onClick=${reset}>Reset</button>
          </div>
          <div class="note">${mode === 'learn'
            ? 'Cold-start: the model starts junk-blind. Dump/keep particles and the junk-rejection F1 climbs as it learns.'
            : 'Pre-trained junk classifier. Your dump/keep corrections override individual particles instantly.'}</div>
          ${f1hist.length > 1 ? html`<${Plot} h=${130}
            data=${[{ type: 'scatter', mode: 'lines+markers', y: f1hist, line: { color: '#38bdf8' } }]}
            layout=${{ yaxis: { range: [0, 1], title: 'junk-F1' }, xaxis: { title: 'corrections' } }} />` : null}
        </div>

        <div class="divider"></div>
        <div class="group">
          <h4>Real-time stream</h4>
          <div class="row">
            <button class=${streaming ? 'sm junk active' : 'primary sm'} onClick=${toggleStream}>${streaming ? 'Stop stream' : 'Start stream'}</button>
          </div>
          ${timeline.length > 1 ? html`<${Plot} h=${110}
            data=${[{ type: 'scatter', mode: 'lines', y: timeline, line: { color: '#d8a657' }, fill: 'tozeroy', fillcolor: 'rgba(216,166,87,.1)' }]}
            layout=${{ yaxis: { range: [0, 100], title: '%junk' } }} />` : null}
        </div>

        <div class="divider"></div>
        <div class="group">
          <h4>Export</h4>
          <div class="row" style=${{ flexWrap: 'wrap' }}>
            <button class="sm" onClick=${() => window.open(`/api/export/coords/${empiar}/${stem}?fmt=star`)}>.star</button>
            <button class="sm" onClick=${() => window.open(`/api/export/coords/${empiar}/${stem}?fmt=box`)}>.box</button>
            <button class="sm" onClick=${() => savePng('image/png')}>PNG</button>
            <button class="sm" onClick=${() => savePng('image/jpeg')}>JPEG</button>
            <button class="sm" onClick=${() => window.open(`/api/export/report/${empiar}`)}>PDF report</button>
          </div>
          <div class="note">.star / .box = kept-particle coordinates for RELION &amp; cryoSPARC;
            PNG/JPEG = overlay snapshot; PDF = honest scorecard.</div>
        </div>

        <div class="divider"></div>
        <div class="group">
          <h4>2D class averages (kept particles)</h4>
          <button class="sm" onClick=${run2d} disabled=${busy2d}>${busy2d ? 'Classifying…' : 'Compute 2D classes'}</button>
          ${classes ? html`<div class="montage">${classes.classes.map((c, i) =>
            html`<figure key=${i}><img src=${c.png} /><figcaption>${c.count} ptcls</figcaption></figure>`)}</div>
            <div class="note">${classes.n_particles} kept particles → coherent protein density = real picks.</div>` : null}
        </div>
      </div>
    </div>

    <div class="ftr">
      <span>pickers: blob (LoG) + CryoSegNet (SAM)</span>
      <span>junk classifier: RandomForest</span>
      <span>open source · MIT</span>
      <span class="spacer"></span>
      <span class="tag">corrections fed: ${met ? '' : ''}${(f1hist.length)}</span>
    </div>
  </div>`;
}

ReactDOM.createRoot(document.getElementById('root')).render(html`<${App} />`);

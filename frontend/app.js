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
  const baseRef = useRef(null);          // offscreen layer: image + all pick circles (static during drag)
  const rafRef = useRef(0);
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

  // Render the image + every pick circle to the offscreen base ONCE (on image/picks change).
  const renderBase = () => {
    const img = imgRef.current;
    if (!img.complete || !img.naturalWidth) return;
    if (!baseRef.current) baseRef.current = document.createElement('canvas');
    const base = baseRef.current;
    base.width = img.naturalWidth; base.height = img.naturalHeight;
    const bx = base.getContext('2d');
    bx.drawImage(img, 0, 0);
    if (picks) {
      bx.lineWidth = 1.4;
      for (let i = 0; i < picks.x.length; i++) {
        const junk = picks.junk[i];
        bx.strokeStyle = junk ? 'rgba(229,72,77,.85)' : 'rgba(41,195,147,.95)';
        bx.beginPath(); bx.arc(picks.x[i], picks.y[i], junk ? R * 0.7 : R, 0, 6.2832); bx.stroke();
      }
    }
  };
  // Blit the static base + draw only the selection rectangle (cheap, O(1) — used during drag).
  const blit = (sel) => {
    const cv = canvasRef.current, base = baseRef.current;
    if (!cv || !base || !base.width) return;
    if (cv.width !== base.width) { cv.width = base.width; cv.height = base.height; }
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(base, 0, 0);
    if (sel) {
      const lw = 1.5 / Math.max(view.current.z, 1);
      ctx.strokeStyle = brush === 'keep' ? '#29c393' : '#e5484d';
      ctx.fillStyle = brush === 'keep' ? 'rgba(41,195,147,.12)' : 'rgba(229,72,77,.12)';
      ctx.lineWidth = lw; ctx.setLineDash([5, 4]);
      const x = Math.min(sel.x0, sel.x1), y = Math.min(sel.y0, sel.y1);
      ctx.fillRect(x, y, Math.abs(sel.x1 - sel.x0), Math.abs(sel.y1 - sel.y0));
      ctx.strokeRect(x, y, Math.abs(sel.x1 - sel.x0), Math.abs(sel.y1 - sel.y0));
      ctx.setLineDash([]);
    }
  };
  const draw = (sel) => { renderBase(); blit(sel); };  // full redraw (image/picks changed)

  useEffect(() => {
    const img = imgRef.current;
    img.onload = () => { draw(); fit(); };
    img.src = `/api/img/${empiar}/${stem}.png`;
  }, [empiar, stem]);
  useEffect(() => { draw(); }, [picks]);
  useEffect(() => {
    const w = wrapRef.current; if (!w) return;
    const onWheel = (e) => { e.preventDefault(); const r = w.getBoundingClientRect();
      // proportional + clamped so wheels/trackpads zoom smoothly instead of jumping
      const fac = Math.min(1.25, Math.max(0.8, Math.exp(-e.deltaY * 0.0012)));
      zoomAt(e.clientX - r.left, e.clientY - r.top, fac); };
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
    if (!rafRef.current) rafRef.current = requestAnimationFrame(() => { rafRef.current = 0; blit(drag.current); });
  };
  const onUp = () => {
    const d = drag.current; drag.current = null;
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = 0; }
    if (!d || d.pan || !picks) { blit(); return; }
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
    blit();   // clear the selection rect now; new picks (from onCorrect) re-render via the picks effect
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
  const [picker, setPicker] = useState('blob');
  const [brush, setBrush] = useState('junk');
  const [tool, setTool] = useState('select');
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [uploadMsg, setUploadMsg] = useState('');
  const [datasets, setDatasets] = useState([]);
  const [selEmpiar, setSelEmpiar] = useState(null);
  const [f1hist, setF1] = useState([]);
  const [classes, setClasses] = useState(null);
  const [busy2d, setBusy2d] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [timeline, setTimeline] = useState([]);
  const wsRef = useRef(null);

  const empiar = info && info.empiar;
  const mic = info && info.micrographs[idx];
  const stem = mic && mic.stem;

  useEffect(() => {
    G('/api/datasets').then(d => {
      const ds = (d.datasets || []).filter(x => x.ready);
      setDatasets(ds);
      setSelEmpiar(((ds.find(x => x.empiar === '10017') || ds[0]) || {}).empiar || '10017');
    });
  }, []);
  useEffect(() => {
    if (!selEmpiar) return;
    setInfo(null); setPicks(null); setMet(null); setIdx(0); setF1([]); setClasses(null);
    G(`/api/state?empiar=${selEmpiar}`).then(s => { setInfo(s); setTh(s.threshold); setClf(s.clf_model); setPicker(s.picker || 'blob'); });
  }, [selEmpiar]);

  const load = useCallback(() => {
    if (!empiar || !stem) return;
    G(`/api/picks/${empiar}/${stem}`).then(setPicks);
    G(`/api/metrics/${empiar}/${stem}`).then(setMet);
  }, [empiar, stem, picker]);
  useEffect(() => { load(); }, [load]);
  // prefetch adjacent micrographs (image + picks) so prev/next is lag-free
  useEffect(() => {
    if (!info || !empiar) return;
    [idx - 1, idx + 1, idx + 2].forEach(j => {
      if (j >= 0 && j < info.micrographs.length) {
        const st = info.micrographs[j].stem;
        const im = new Image(); im.src = `/api/img/${empiar}/${st}.png`;
        G(`/api/picks/${empiar}/${st}`);
      }
    });
  }, [idx, empiar, info, picker]);

  const onCorrect = (sel, b) => {
    const body = { empiar, stem, dump_idx: b === 'keep' ? [] : sel, keep_idx: b === 'keep' ? sel : [] };
    J('/api/correct', body).then(r => {
      setPicks(p => ({ ...p, score: r.score, junk: r.junk }));
      setF1(r.f1_history); setCanUndo(!!r.can_undo); setCanRedo(!!r.can_redo);
      G(`/api/metrics/${empiar}/${stem}`).then(setMet);
    });
  };

  const applyUndoRedo = (r) => {
    if (!r) return;
    setCanUndo(!!r.can_undo); setCanRedo(!!r.can_redo);
    if (!r.ok) return;
    if (r.f1_history) setF1(r.f1_history);
    if (r.stem !== stem && info) {                 // correction was on another micrograph → jump there
      const j = info.micrographs.findIndex(m => m.stem === r.stem);
      if (j >= 0) setIdx(j);
    } else {
      setPicks(p => ({ ...p, score: r.score, junk: r.junk }));
      G(`/api/metrics/${empiar}/${stem}`).then(setMet);
    }
  };
  const undo = () => J('/api/undo', { empiar, stem }).then(applyUndoRedo);
  const redo = () => J('/api/redo', { empiar, stem }).then(applyUndoRedo);
  const uploadMrc = (files) => {
    const f = files && files[0]; if (!f) return;
    setUploading(true); setUploadPct(0); setUploadMsg('');
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/upload?empiar=${empiar}&filename=${encodeURIComponent(f.name)}`);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) setUploadPct(Math.round(100 * e.loaded / e.total)); };
    xhr.onload = () => {
      setUploading(false); setUploadPct(0);
      let r = {}; try { r = JSON.parse(xhr.responseText); } catch (_) { /* */ }
      if (!r.ok) { alert('Upload failed: ' + (r.error || ('HTTP ' + xhr.status))); return; }
      setUploadMsg(`uploaded ${r.stem} · ${r.n_picks} picks`);
      G(`/api/state?empiar=${empiar}`).then(s => { setInfo(s); const j = s.micrographs.findIndex(m => m.stem === r.stem); if (j >= 0) setIdx(j); });
    };
    xhr.onerror = () => { setUploading(false); setUploadPct(0); alert('Upload error'); };
    xhr.send(f);
  };
  const clearUploads = () => {
    J('/api/upload/clear', { empiar }).then(() => {
      setUploadMsg('');
      G(`/api/state?empiar=${empiar}`).then(s => { setInfo(s); setIdx(0); });
    });
  };

  const changeTh = (t) => { setTh(t); J('/api/threshold', { empiar, threshold: t }).then(() => load()); };
  const switchMode = (m) => { setMode(m); J('/api/mode', { empiar, mode: m }).then(() => { setF1([]); load(); }); };
  const switchClf = (m) => { setClf(m); J('/api/clf_model', { empiar, clf_model: m }).then(r => { if (r && r.threshold != null) setTh(r.threshold); load(); }); };
  const switchPicker = (p) => {
    const curStem = stem;
    J('/api/picker', { empiar, picker: p }).then(r => {
      if (!r || !r.ok) { alert('Picker not available for this dataset'); return; }
      setPicker(p); setPicks(null); setMet(null); setF1([]);
      G(`/api/state?empiar=${empiar}`).then(s => {
        setInfo(s);
        const j = s.micrographs.findIndex(m => m.stem === curStem);
        setIdx(j >= 0 ? j : 0);   // keep the SAME micrograph so the picker difference is clear
      });
    });
  };
  const reset = () => { J('/api/reset', { empiar, coldstart: mode === 'learn' }).then(() => { setF1([]); setCanUndo(false); setCanRedo(false); load(); }); };
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

  // keyboard: d/k toggle brush; Ctrl/Cmd+Z undo, Ctrl/Cmd+Shift+Z (or Ctrl+Y) redo.
  const kbd = useRef({});
  kbd.current = { undo, redo };
  useEffect(() => {
    const h = (e) => {
      const meta = e.ctrlKey || e.metaKey;
      if (meta && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); (e.shiftKey ? kbd.current.redo : kbd.current.undo)(); return; }
      if (meta && (e.key === 'y' || e.key === 'Y')) { e.preventDefault(); kbd.current.redo(); return; }
      if (!meta && e.key === 'k') setBrush('keep');
      if (!meta && e.key === 'd') setBrush('junk');
    };
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
      ${datasets.length > 1 ? html`<select class="dataset-sel" value=${selEmpiar || ''}
         onChange=${e => setSelEmpiar(e.target.value)} title="Switch dataset">
        ${datasets.map(d => html`<option key=${d.empiar} value=${d.empiar}>${d.label}${d.has_gt ? '' : ' · no GT'}</option>`)}
      </select>` : html`<div class="pill">dataset <b>EMPIAR-${empiar}</b></div>`}
      <div class="pill">${info.micrographs.length} micrographs</div>
      <div class="pill"><span class=${'dot ' + (streaming ? 'on' : 'off')}></span>${streaming ? 'streaming' : 'idle'}</div>
    </div>

    <div class="main">
      <div class="viewer">
        <div class="toolbar">
          <button class="sm" onClick=${() => setIdx(i => Math.max(0, i - 1))}>◀ prev</button>
          <select value=${stem} onChange=${e => setIdx(info.micrographs.findIndex(m => m.stem === e.target.value))}
             style=${{ width: '230px' }}>
            ${info.micrographs.map((m, i) => html`<option key=${m.stem} value=${m.stem}>${i + 1}. ${m.stem}${m.uploaded ? '  (uploaded)' : ''}</option>`)}
          </select>
          <button class="sm" onClick=${() => setIdx(i => Math.min(info.micrographs.length - 1, i + 1))}>next ▶</button>
          ${info.micrographs.some(m => m.uploaded)
      ? html`<button class="sm junk" onClick=${clearUploads} title="Remove uploaded micrographs">clear uploads</button>`
      : html`<label class=${'sm' + (uploading ? ' primary' : '')} style=${{ cursor: 'pointer' }}
             title="Pick + junk-triage your own micrograph (no ground truth)">
            ${uploading ? `uploading ${uploadPct}%` : 'upload MRC'}
            <input type="file" accept=".mrc" style=${{ display: 'none' }} onChange=${e => uploadMrc(e.target.files)} />
          </label>`}
          ${uploading ? html`<div class="uprog"><div class="upbar" style=${{ width: uploadPct + '%' }}></div></div>` : ''}
          ${uploadMsg ? html`<span class="lbl" style=${{ color: 'var(--keep)' }}>${uploadMsg}</span>` : ''}
          <div style=${{ width: '12px' }}></div>
          <span class="lbl">brush:</span>
          <button class=${'sm junk' + (brush === 'junk' ? ' active' : '')} onClick=${() => setBrush('junk')}>dump <span class="kbd">d</span></button>
          <button class=${'sm keep' + (brush === 'keep' ? ' active' : '')} onClick=${() => setBrush('keep')}>keep <span class="kbd">k</span></button>
          <div style=${{ width: '12px' }}></div>
          <span class="lbl">tool:</span>
          <button class=${'sm' + (tool === 'select' ? ' primary' : '')} onClick=${() => setTool('select')}>select</button>
          <button class=${'sm' + (tool === 'pan' ? ' primary' : '')} onClick=${() => setTool('pan')}>pan</button>
          <div style=${{ width: '12px' }}></div>
          <button class="sm" disabled=${!canUndo} onClick=${undo} title="Ctrl/Cmd+Z">↶ undo</button>
          <button class="sm" disabled=${!canRedo} onClick=${redo} title="Ctrl/Cmd+Shift+Z">↷ redo</button>
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
          <h4>Scoreboard — kept particles vs CryoPPP ground truth</h4>
          <div class="cards">
            <${Metric} big k="Kept-set purity (precision)" v=${pa ? (pa.precision * 100).toFixed(0) + '%' : '—'}
              d=${pa && pb ? { up: pa.precision >= pb.precision, t: (pa.precision - pb.precision >= 0 ? '+' : '') + ((pa.precision - pb.precision) * 100).toFixed(0) + ' pts vs raw picks' } : null} />
            <${Metric} k="Raw-pick purity" v=${pb ? (pb.precision * 100).toFixed(0) + '%' : '—'} />
            <${Metric} k="Junk removed" v=${met ? met.junk_pct.toFixed(0) + '%' : '—'} />
            <${Metric} k="Junk-rejection F1" v=${jr ? jr.junk_f1.toFixed(2) : '—'} />
            <${Metric} k="Kept particles" v=${met ? met.n_kept : '—'} />
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
          <h4>Picker</h4>
          <select value=${picker} onChange=${e => switchPicker(e.target.value)}>
            ${Object.entries(info.picker_menu || { blob: { label: 'Blob LoG', device: 'CPU', ready: true } })
        .filter(([k, m]) => m.ready).map(([k, m]) =>
          html`<option key=${k} value=${k}>${m.label} · ${m.device}</option>`)}
          </select>
          <div class="note">${(info.pickers || []).length > 1
            ? 'Switching picker keeps the same micrograph, so the differences are clear. The junk classifier then triages whichever picker you choose.'
            : 'Only the blob picker is cached for this dataset. Topaz / CryoSegNet are wired but cached only where you see them in the menu.'}</div>
        </div>

        <div class="divider"></div>
        <div class="group">
          <h4>Junk classifier</h4>
          <select value=${clfModel} disabled=${mode === 'learn'} onChange=${e => switchClf(e.target.value)}>
            ${Object.entries(info.clf_options).map(([k, v]) =>
              html`<option key=${k} value=${k}>${v.label}</option>`)}
          </select>
          <div class="note">Flags each candidate as keep or junk (carbon / ice / aggregate) and removes it
            from the kept set, lifting purity while keeping the real particles. LightGBM is the robust
            default; RandomForest, SGD (online), and the CNN are alternatives. Scoreboard is in-sample on
            these micrographs; held-out generalisation is more conservative.</div>
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
      <span>picker: ${(info.picker_menu && info.picker_menu[picker] && info.picker_menu[picker].label) || picker}</span>
      <span>junk classifier: ${(info.clf_options && info.clf_options[clfModel] && info.clf_options[clfModel].label.split(' —')[0]) || clfModel}</span>
      <span>open source · MIT</span>
      <span class="spacer"></span>
      <span class="tag">corrections fed: ${met ? '' : ''}${(f1hist.length)}</span>
    </div>
  </div>`;
}

ReactDOM.createRoot(document.getElementById('root')).render(html`<${App} />`);

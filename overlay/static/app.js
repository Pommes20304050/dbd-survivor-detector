// DBD Survivor Detector Dashboard

const API = '/api';
const POLL = 400;

// State
let chartData = { fps: Array(60).fill(0), gpu: Array(60).fill(0), vram: Array(60).fill(0), temp: Array(60).fill(0) };
let activeMetric = 'fps';
let rendered = { presets: false, profiles: false };

// ─── Helpers ──────────────────────────────────────────────

async function api(path, method = 'GET', data = null) {
    try {
        const opts = { method };
        if (data) {
            opts.headers = { 'Content-Type': 'application/json' };
            opts.body = JSON.stringify(data);
        }
        const res = await fetch(API + path, opts);
        return await res.json();
    } catch (e) { console.error(e); return null; }
}

function fmtTime(sec) {
    if (sec < 60) return `${Math.floor(sec)}s`;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    if (m < 60) return `${m}m ${s}s`;
    return `${Math.floor(m/60)}h ${m%60}m`;
}

function $(id) { return document.getElementById(id); }

// ─── Tabs ─────────────────────────────────────────────────

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const tab = item.dataset.tab;
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        item.classList.add('active');
        document.querySelector(`.tab-${tab}`).classList.add('active');
    });
});

// ─── Chart Metric Switcher ────────────────────────────────

document.querySelectorAll('.chart-tab').forEach(t => {
    t.addEventListener('click', () => {
        document.querySelectorAll('.chart-tab').forEach(x => x.classList.remove('active'));
        t.classList.add('active');
        activeMetric = t.dataset.metric;
        drawChart();
        fetchStatus();   // big value sofort updaten
    });
});

// ─── Buttons ──────────────────────────────────────────────

$('btnStart').addEventListener('click', async () => {
    $('btnStart').disabled = true;
    try { await api('/start', 'POST'); }
    finally { fetchStatus(); }   // updateUI setzt button-state basierend auf state
});

$('btnStop').addEventListener('click', async () => {
    $('btnStop').disabled = true;
    try { await api('/stop', 'POST'); }
    finally { fetchStatus(); }
});

$('btnShutdown').addEventListener('click', async () => {
    if (!confirm('Wirklich alles beenden?')) return;
    await api('/shutdown', 'POST');
});

// ─── Settings ─────────────────────────────────────────────

$('confSlider').addEventListener('input', e => {
    const v = parseFloat(e.target.value);
    $('confValue').textContent = `${Math.round(v * 100)}%`;
    api('/config', 'POST', { conf: v });
});

$('thickSlider').addEventListener('input', e => {
    const v = parseInt(e.target.value);
    $('thickValue').textContent = `${v}px`;
    api('/config', 'POST', { box_thickness: v });
});

$('maxDetSlider').addEventListener('input', e => {
    const v = parseInt(e.target.value);
    $('maxDetValue').textContent = v;
    api('/config', 'POST', { max_detections: v });
});

['toggleLabels', 'toggleConf', 'toggleGlow', 'toggleCrosshair', 'toggleHud'].forEach(id => {
    const keyMap = { toggleLabels: 'show_labels', toggleConf: 'show_conf', toggleGlow: 'glow',
                     toggleCrosshair: 'show_crosshair', toggleHud: 'show_hud_regions' };
    $(id).addEventListener('change', e => api('/config', 'POST', { [keyMap[id]]: e.target.checked }));
});

// Monitor-Select
const monitorSelect = $('monitorSelect');
if (monitorSelect) {
    monitorSelect.addEventListener('change', async (e) => {
        const idx = parseInt(e.target.value);
        if (!isNaN(idx)) {
            await api('/monitor', 'POST', { index: idx });
        }
    });
}

function renderMonitors(monitors, activeIdx) {
    if (!monitorSelect || !monitors || monitors.length === 0) return;
    const sig = monitors.map(m => m.name).join('|');
    if (monitorSelect.dataset.sig === sig) {
        monitorSelect.value = activeIdx;
        return;
    }
    monitorSelect.dataset.sig = sig;
    monitorSelect.innerHTML = '';
    monitors.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.idx;
        opt.textContent = m.name;
        monitorSelect.appendChild(opt);
    });
    monitorSelect.value = activeIdx;
}

document.querySelectorAll('.color-dot').forEach(d => {
    d.addEventListener('click', e => {
        document.querySelectorAll('.color-dot').forEach(x => x.classList.remove('selected'));
        e.target.classList.add('selected');
        api('/config', 'POST', { color: e.target.dataset.color });
    });
});

// ─── Status fetch + UI Update ─────────────────────────────

async function fetchStatus() {
    const s = await api('/status');
    if (!s) return;
    updateUI(s);
}

function set(id, key, val) { const el = $(id); if (el) el[key] = val; }

function updateUI(s) {
    // Defensive defaults
    s = s || {};
    const fps = Number.isFinite(s.fps) ? s.fps : 0;
    const detections = s.detections ?? 0;
    const total = s.total_detections ?? 0;
    const uptime = s.uptime ?? 0;
    const running = !!s.running;
    const gpu = s.gpu || {};

    // Sidebar
    set('sbStatus', 'textContent', running ? 'AKTIV' : 'STANDBY');
    const sb = $('sbStatus'); if (sb) sb.style.color = running ? 'var(--green)' : 'var(--text-dim)';
    set('sbFps', 'textContent', fps.toFixed(1));
    set('sbDet', 'textContent', detections);
    set('sbTotal', 'textContent', total.toLocaleString());
    set('btnStart', 'disabled', running);
    set('btnStop', 'disabled', !running);

    // GPU + CPU Chips im Topbar
    const gpuShort = (gpu.name || 'GPU').replace('NVIDIA GeForce ', '').trim();
    set('gpuName', 'textContent', gpuShort);
    set('gpuPct', 'textContent', `${gpu.util || 0}%`);

    const cpu = s.cpu || {};
    const cpuShort = (cpu.name || 'CPU')
        .split('@')[0]
        .replace('Intel(R) Core(TM) ', '')
        .replace('(R)', '')
        .replace('(TM)', '')
        .replace('CPU', '')
        .trim();
    set('cpuName', 'textContent', cpuShort || 'CPU');
    set('cpuPct', 'textContent', `${(cpu.util || 0).toFixed(0)}%`);

    // Chart data push
    if (running) {
        chartData.fps.shift(); chartData.fps.push(fps);
        chartData.gpu.shift(); chartData.gpu.push(gpu.util || 0);
        chartData.vram.shift(); chartData.vram.push(gpu.vram_pct || 0);
        chartData.temp.shift(); chartData.temp.push(gpu.temp || 0);
    }

    // Big chart value
    const bigVal = { fps: fps.toFixed(1), gpu: (gpu.util || 0) + '%',
                     vram: (gpu.vram_pct || 0) + '%', temp: (gpu.temp || 0) + '°C' };
    const bigLabel = { fps: 'FPS', gpu: 'GPU LOAD', vram: 'VRAM USED', temp: 'TEMPERATURE' };
    set('chartBigValue', 'textContent', bigVal[activeMetric]);
    set('chartBigLabel', 'textContent', bigLabel[activeMetric]);

    // Mini Stats
    set('miniFps', 'textContent', fps.toFixed(1));
    const fb = $('miniFpsBar'); if (fb) fb.style.width = Math.min(100, fps) + '%';
    set('miniGpu', 'textContent', (gpu.util || 0) + '%');
    const gb = $('miniGpuBar'); if (gb) gb.style.width = (gpu.util || 0) + '%';
    set('miniVram', 'textContent', gpu.vram_used ? `${(gpu.vram_used/1024).toFixed(1)} GB` : '—');
    const vb = $('miniVramBar'); if (vb) vb.style.width = (gpu.vram_pct || 0) + '%';
    set('miniTemp', 'textContent', (gpu.temp || 0) + '°C');
    const tb = $('miniTempBar'); if (tb) tb.style.width = Math.min(100, (gpu.temp || 0) / 90 * 100) + '%';
    set('miniPower', 'textContent', (gpu.power || 0).toFixed(0) + ' W');
    const pb = $('miniPowerBar'); if (pb) pb.style.width = Math.min(100, (gpu.power || 0) / 300 * 100) + '%';
    set('miniUptime', 'textContent', fmtTime(uptime));

    // Feed FPS
    set('feedFps',  'textContent', running ? `${fps.toFixed(0)} FPS` : 'INACTIVE');
    set('feedFps2', 'textContent', running ? `${fps.toFixed(0)} FPS` : 'INACTIVE');
    set('infoModel','textContent', s.model || '—');

    // Monitor-Dropdown
    renderMonitors(s.monitors, s.monitor_index);

    // Presets
    if (!rendered.presets && s.presets) {
        renderPresets(s.presets, s.active_preset);
        rendered.presets = true;
    } else if (s.presets) {
        document.querySelectorAll('.preset-card').forEach(c =>
            c.classList.toggle('active', c.dataset.preset === s.active_preset));
    }

    // Profiles
    if (!rendered.profiles && s.profiles) {
        renderProfiles(s.profiles, s.profile);
        rendered.profiles = true;
    } else if (s.profiles) {
        document.querySelectorAll('.profile-card').forEach(c =>
            c.classList.toggle('active', c.dataset.profile === s.profile));
    }

    drawChart();
    drawDonut(s);
}

// ─── Presets ──────────────────────────────────────────────

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

function renderPresets(presets, active) {
    const order = ['maximum', 'standard', 'competitive', 'minimal', 'stream'];
    const grid = $('presetGrid');
    if (!grid) return;
    grid.innerHTML = '';
    order.forEach(key => {
        const p = presets[key]; if (!p) return;
        const card = document.createElement('div');
        card.className = 'preset-card' + (key === active ? ' active' : '');
        card.dataset.preset = key;
        card.appendChild(el('div', 'preset-emoji', p.emoji || '◉'));
        const info = el('div', 'preset-info');
        info.appendChild(el('div', 'preset-name', p.name || ''));
        info.appendChild(el('div', 'preset-desc', p.desc || ''));
        card.appendChild(info);
        card.addEventListener('click', async () => {
            await api('/preset', 'POST', { preset: key });
            document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            setTimeout(() => { rendered.profiles = false; fetchStatus(); }, 150);
        });
        grid.appendChild(card);
    });
}

// ─── Profiles ─────────────────────────────────────────────

function renderProfiles(profiles, active) {
    const order = ['max', 'ultra', 'high', 'balanced', 'fast', 'extreme'];
    const grid = $('profileGrid');
    if (!grid) return;
    grid.innerHTML = '';
    order.forEach(key => {
        const p = profiles[key]; if (!p) return;
        const card = document.createElement('div');
        card.className = 'profile-card' + (key === active ? ' active' : '');
        card.dataset.profile = key;
        const badge = p.use_engine ? ' · TRT' : '';
        card.innerHTML = `
            <div class="profile-name">${p.name.toUpperCase()}${badge}</div>
            <div class="profile-sub">${p.desc}</div>
            <div class="profile-metric"><span class="profile-metric-label">Auflösung</span><span class="profile-metric-value">${p.imgsz}px</span></div>
            <div class="profile-metric"><span class="profile-metric-label">VRAM</span><span class="profile-metric-value">${p.vram_mb} MB</span></div>
            <div class="profile-bar"><div class="profile-bar-fill vram" style="width:${p.vram_pct}%"></div></div>
            <div class="profile-metric"><span class="profile-metric-label">FPS</span><span class="profile-metric-value">${p.fps}</span></div>
            <div class="profile-bar"><div class="profile-bar-fill fps" style="width:${Math.min(100, p.fps/4)}%"></div></div>
            <div class="profile-metric"><span class="profile-metric-label">Qualität</span><span class="profile-metric-value">${p.quality_pct}%</span></div>
            <div class="profile-bar"><div class="profile-bar-fill quality" style="width:${p.quality_pct}%"></div></div>`;
        card.addEventListener('click', async () => {
            await api('/profile', 'POST', { profile: key });
            document.querySelectorAll('.profile-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');
        });
        grid.appendChild(card);
    });
}

// ─── Chart ────────────────────────────────────────────────

function drawChart() {
    const canvas = $('mainChart');
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;  // hidden tab
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const data = chartData[activeMetric];
    const maxMap = { fps: Math.max(100, ...data, 1), gpu: 100, vram: 100, temp: 90 };
    const max = maxMap[activeMetric];

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
        const y = (h / 5) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Y-axis values
    ctx.fillStyle = 'rgba(138,143,154,0.7)';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'left';
    for (let i = 0; i <= 4; i++) {
        const v = max - (max / 4) * i;
        ctx.fillText(Math.round(v), 4, (h / 4) * i + 10);
    }

    // Line with fill
    const step = w / (data.length - 1);
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(255, 140, 0, 0.35)');
    grad.addColorStop(1, 'rgba(255, 140, 0, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, h);
    data.forEach((v, i) => ctx.lineTo(i * step, h - (v / max) * h));
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fill();

    // Line
    ctx.strokeStyle = '#ff8c00';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(255,140,0,0.6)';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    data.forEach((v, i) => {
        const x = i * step, y = h - (v / max) * h;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Current point
    const lx = (data.length - 1) * step;
    const ly = h - (data[data.length - 1] / max) * h;
    ctx.fillStyle = '#ff8c00';
    ctx.beginPath();
    ctx.arc(lx, ly, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();
}

// ─── Donut ────────────────────────────────────────────────

let donutCounts = { high: 0, med: 0, low: 0 };

function drawDonut(s) {
    const total = (s && s.total_detections) || 0;
    donutCounts = {
        high: Math.floor(total * 0.70),
        med:  Math.floor(total * 0.25),
        low:  total - Math.floor(total * 0.70) - Math.floor(total * 0.25),
    };

    const sum = Math.max(1, donutCounts.high + donutCounts.med + donutCounts.low);
    const circ = 2 * Math.PI * 50;  // r=50 → ~314.16

    const highLen = (donutCounts.high / sum) * circ;
    const medLen  = (donutCounts.med  / sum) * circ;
    const lowLen  = (donutCounts.low  / sum) * circ;

    const dh = $('donutHigh'), dm = $('donutMed'), dl = $('donutLow');
    if (dh && dm && dl) {
        // Jedes Segment: dasharray = "segLen gap", dashoffset dreht Start
        // High: Start bei 0
        dh.style.strokeDasharray = `${highLen} ${circ}`;
        dh.style.strokeDashoffset = '0';

        // Med: Start nach High (negatives offset rueckt Start)
        dm.style.strokeDasharray = `${medLen} ${circ}`;
        dm.style.strokeDashoffset = `${-highLen}`;

        // Low: Start nach High+Med
        dl.style.strokeDasharray = `${lowLen} ${circ}`;
        dl.style.strokeDashoffset = `${-(highLen + medLen)}`;

        set('donutTotal', 'textContent', total.toLocaleString());
        set('legHigh',    'textContent', donutCounts.high.toLocaleString());
        set('legMed',     'textContent', donutCounts.med.toLocaleString());
        set('legLow',     'textContent', donutCounts.low.toLocaleString());
    }
}

// ─── Init ─────────────────────────────────────────────────

async function init() {
    const s = await api('/status');
    if (s) {
        $('confSlider').value = s.conf;
        $('confValue').textContent = `${Math.round(s.conf * 100)}%`;
        $('thickSlider').value = s.box_thickness || 3;
        $('thickValue').textContent = `${s.box_thickness || 3}px`;
        $('maxDetSlider').value = s.max_detections || 10;
        $('maxDetValue').textContent = s.max_detections || 10;
        $('toggleLabels').checked = s.show_labels;
        $('toggleConf').checked = s.show_conf;
        $('toggleGlow').checked = s.glow;
        $('toggleCrosshair').checked = s.show_crosshair;
        $('toggleHud').checked = s.show_hud_regions;
        document.querySelectorAll('.color-dot').forEach(d => {
            if (d.dataset.color === s.color) d.classList.add('selected');
        });
        updateUI(s);
    }
    setInterval(fetchStatus, POLL);
    window.addEventListener('resize', drawChart);
}

init();

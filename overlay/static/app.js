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
    });
});

// ─── Buttons ──────────────────────────────────────────────

$('btnStart').addEventListener('click', async () => {
    $('btnStart').disabled = true;
    await api('/start', 'POST');
    fetchStatus();
});

$('btnStop').addEventListener('click', async () => {
    $('btnStop').disabled = true;
    await api('/stop', 'POST');
    fetchStatus();
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

function updateUI(s) {
    // Sidebar
    $('sbStatus').textContent = s.running ? 'AKTIV' : 'STANDBY';
    $('sbStatus').style.color = s.running ? 'var(--green)' : 'var(--text-dim)';
    $('sbFps').textContent = s.fps.toFixed(1);
    $('sbDet').textContent = s.detections;
    $('sbTotal').textContent = s.total_detections.toLocaleString();
    $('btnStart').disabled = s.running;
    $('btnStop').disabled = !s.running;

    // GPU
    const gpu = s.gpu || {};
    $('gpuName').textContent = gpu.name || 'GPU';

    // Chart data push
    if (s.running) {
        chartData.fps.shift(); chartData.fps.push(s.fps);
        chartData.gpu.shift(); chartData.gpu.push(gpu.util || 0);
        chartData.vram.shift(); chartData.vram.push(gpu.vram_pct || 0);
        chartData.temp.shift(); chartData.temp.push(gpu.temp || 0);
    }

    // Big chart value
    const bigVal = { fps: s.fps.toFixed(1), gpu: (gpu.util || 0) + '%',
                     vram: (gpu.vram_pct || 0) + '%', temp: (gpu.temp || 0) + '°C' };
    const bigLabel = { fps: 'FPS', gpu: 'GPU LOAD', vram: 'VRAM USED', temp: 'TEMPERATURE' };
    $('chartBigValue').textContent = bigVal[activeMetric];
    $('chartBigLabel').textContent = bigLabel[activeMetric];

    // Mini Stats
    $('miniFps').textContent = s.fps.toFixed(1);
    $('miniFpsBar').style.width = Math.min(100, s.fps / 100 * 100) + '%';
    $('miniGpu').textContent = (gpu.util || 0) + '%';
    $('miniGpuBar').style.width = (gpu.util || 0) + '%';
    $('miniVram').textContent = gpu.vram_used ? `${(gpu.vram_used/1024).toFixed(1)} GB` : '—';
    $('miniVramBar').style.width = (gpu.vram_pct || 0) + '%';
    $('miniTemp').textContent = (gpu.temp || 0) + '°C';
    $('miniTempBar').style.width = Math.min(100, (gpu.temp || 0) / 90 * 100) + '%';
    $('miniPower').textContent = (gpu.power || 0).toFixed(0) + ' W';
    $('miniPowerBar').style.width = Math.min(100, (gpu.power || 0) / 300 * 100) + '%';
    $('miniUptime').textContent = fmtTime(s.uptime);

    // Feed FPS
    $('feedFps').textContent = s.running ? `${s.fps.toFixed(0)} FPS` : 'INACTIVE';
    if ($('feedFps2')) $('feedFps2').textContent = s.running ? `${s.fps.toFixed(0)} FPS` : 'INACTIVE';

    // Info tab
    if ($('infoModel')) $('infoModel').textContent = s.model || '—';

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

function renderPresets(presets, active) {
    const order = ['standard', 'competitive', 'minimal', 'stream'];
    const grid = $('presetGrid');
    grid.innerHTML = '';
    order.forEach(key => {
        const p = presets[key]; if (!p) return;
        const card = document.createElement('div');
        card.className = 'preset-card' + (key === active ? ' active' : '');
        card.dataset.preset = key;
        card.innerHTML = `
            <div class="preset-emoji">${p.emoji}</div>
            <div class="preset-info">
                <div class="preset-name">${p.name}</div>
                <div class="preset-desc">${p.desc}</div>
            </div>`;
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
    const order = ['ultra', 'high', 'balanced', 'fast', 'extreme'];
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
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
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
    // Estimate confidence distribution from total
    // Approximation (we don't have per-detection confidence split from API)
    const total = s.total_detections;
    // Assume 70% high, 25% med, 5% low (heuristic)
    donutCounts = {
        high: Math.floor(total * 0.70),
        med:  Math.floor(total * 0.25),
        low:  Math.floor(total * 0.05),
    };

    const sum = Math.max(1, donutCounts.high + donutCounts.med + donutCounts.low);
    const circ = 314;

    const highOff = circ - (donutCounts.high / sum) * circ;
    const medOff  = circ - (donutCounts.med / sum) * circ;
    const lowOff  = circ - (donutCounts.low / sum) * circ;

    const $d = id => document.getElementById(id);
    if ($d('donutHigh')) {
        $d('donutHigh').style.strokeDashoffset = highOff;
        $d('donutMed').style.strokeDashoffset = medOff;
        $d('donutMed').setAttribute('transform', `rotate(${(donutCounts.high/sum) * 360} 60 60)`);
        $d('donutLow').style.strokeDashoffset = lowOff;
        $d('donutLow').setAttribute('transform', `rotate(${((donutCounts.high + donutCounts.med)/sum) * 360} 60 60)`);

        $d('donutTotal').textContent = total.toLocaleString();
        $d('legHigh').textContent = donutCounts.high.toLocaleString();
        $d('legMed').textContent = donutCounts.med.toLocaleString();
        $d('legLow').textContent = donutCounts.low.toLocaleString();
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

// DBD Survivor Detector — Neural Dashboard

const API = '/api';
const POLL_INTERVAL = 300;

// ═══ Elements ══════════════════════════════════════════════

const el = {
    statusDot:   document.getElementById('statusDot'),
    statusText:  document.getElementById('statusText'),
    statusSub:   document.getElementById('statusSub'),
    ringProgress: document.getElementById('ringProgress'),
    btnStart:    document.getElementById('btnStart'),
    btnStop:     document.getElementById('btnStop'),
    btnShutdown: document.getElementById('btnShutdown'),

    statFps:        document.getElementById('statFps'),
    statAvgFps:     document.getElementById('statAvgFps'),
    statDetections: document.getElementById('statDetections'),
    statTotal:      document.getElementById('statTotal'),
    statFrames:     document.getElementById('statFrames'),
    statUptime:     document.getElementById('statUptime'),
    fpsTrend:       document.getElementById('fpsTrend'),
    fpsBar:         document.getElementById('fpsBar'),
    detBar:         document.getElementById('detBar'),

    modelName:  document.getElementById('modelName'),
    infoModel:  document.getElementById('infoModel'),

    mirrorOverlay: document.getElementById('mirrorOverlay'),

    presetGrid:   document.getElementById('presetGrid'),
    currentPreset: document.getElementById('currentPreset'),
    profileGrid:  document.getElementById('profileGrid'),

    confSlider:   document.getElementById('confSlider'),
    confValue:    document.getElementById('confValue'),
    thickSlider:  document.getElementById('thickSlider'),
    thickValue:   document.getElementById('thickValue'),
    maxDetSlider: document.getElementById('maxDetSlider'),
    maxDetValue:  document.getElementById('maxDetValue'),

    toggleLabels:    document.getElementById('toggleLabels'),
    toggleConf:      document.getElementById('toggleConf'),
    toggleGlow:      document.getElementById('toggleGlow'),
    toggleCrosshair: document.getElementById('toggleCrosshair'),
    toggleHud:       document.getElementById('toggleHud'),

    colorDots: document.querySelectorAll('.color-dot'),
    chart:     document.getElementById('fpsChart'),

    advHeader: document.getElementById('advHeader'),
    advPanel:  document.getElementById('advPanel'),
};

// ═══ State ═════════════════════════════════════════════════

let chartData = Array(60).fill(0);
let rendered = { presets: false, profiles: false };
let lastFps = 0;

// ═══ API ═══════════════════════════════════════════════════

async function api(path, method = 'GET', data = null) {
    try {
        const opts = { method };
        if (data) {
            opts.headers = { 'Content-Type': 'application/json' };
            opts.body = JSON.stringify(data);
        }
        const res = await fetch(API + path, opts);
        return await res.json();
    } catch (e) {
        console.error('API:', e);
        return null;
    }
}

async function fetchStatus() {
    const s = await api('/status');
    if (!s) return;
    updateUI(s);
}

// ═══ UI Update ════════════════════════════════════════════

function fmtTime(sec) {
    if (sec < 60) return `${Math.floor(sec)}s`;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    if (m < 60) return `${m}m ${s}s`;
    return `${Math.floor(m/60)}h ${m%60}m`;
}

function updateUI(s) {
    // Status
    const running = s.running;
    el.statusDot.classList.toggle('active', running);
    el.statusText.textContent = running ? 'AKTIV' : 'STANDBY';
    el.statusText.classList.toggle('running', running);
    el.statusSub.textContent = running
        ? `Neural network processing at ${s.fps.toFixed(1)} FPS`
        : 'Waiting for start command';
    el.btnStart.disabled = running;
    el.btnStop.disabled = !running;

    // Ring progress (based on confidence avg or fps)
    const ringPct = Math.min(1, s.fps / 100);
    el.ringProgress.style.strokeDashoffset = 163 * (1 - ringPct);

    // Stats
    el.statFps.textContent = s.fps.toFixed(1);
    el.statAvgFps.textContent = `Ø ${s.avg_fps.toFixed(1)}`;
    el.statDetections.textContent = s.detections;
    el.statTotal.textContent = s.total_detections.toLocaleString();
    el.statFrames.textContent = `${s.frame_count.toLocaleString()} Frames`;
    el.statUptime.textContent = fmtTime(s.uptime);

    // Bars
    el.fpsBar.style.width = Math.min(100, (s.fps / 100) * 100) + '%';
    el.detBar.style.width = Math.min(100, (s.detections / 5) * 100) + '%';

    // Trend
    if (s.fps > lastFps * 1.1) el.fpsTrend.textContent = '↑ UP';
    else if (s.fps < lastFps * 0.9) el.fpsTrend.textContent = '↓ DOWN';
    else el.fpsTrend.textContent = '→ STABLE';
    lastFps = s.fps;

    // Model
    el.modelName.textContent = s.model || '—';
    el.infoModel.textContent = s.model || '—';

    // Mirror FPS
    if (el.mirrorOverlay) {
        el.mirrorOverlay.textContent = running ? `${s.fps.toFixed(0)} FPS` : 'INACTIVE';
    }

    // Preset
    if (el.currentPreset) {
        const presetName = s.active_preset === 'custom'
            ? 'Custom'
            : (s.presets && s.presets[s.active_preset]
                ? s.presets[s.active_preset].name
                : '—');
        el.currentPreset.textContent = `Aktiv: ${presetName}`;
    }

    // Presets rendern
    if (!rendered.presets && s.presets) {
        renderPresets(s.presets, s.active_preset);
        rendered.presets = true;
    } else if (s.presets && el.presetGrid) {
        el.presetGrid.querySelectorAll('.preset-card').forEach(card => {
            card.classList.toggle('active', card.dataset.preset === s.active_preset);
        });
    }

    // Profiles rendern
    if (!rendered.profiles && s.profiles) {
        renderProfiles(s.profiles, s.profile, s.trt_available);
        rendered.profiles = true;
    } else if (s.profiles && el.profileGrid) {
        el.profileGrid.querySelectorAll('.profile-card').forEach(card => {
            card.classList.toggle('active', card.dataset.profile === s.profile);
        });
    }

    // Chart
    if (running) {
        chartData.shift();
        chartData.push(s.fps);
    }
    drawChart();
}

// ═══ Presets (4 Quick Modes) ═══════════════════════════════

function renderPresets(presets, active) {
    const order = ['standard', 'competitive', 'minimal', 'stream'];
    el.presetGrid.innerHTML = '';

    order.forEach(key => {
        const p = presets[key];
        if (!p) return;
        const card = document.createElement('div');
        card.className = 'preset-card' + (key === active ? ' active' : '');
        card.dataset.preset = key;

        card.innerHTML = `
            <div class="preset-emoji">${p.emoji || '◉'}</div>
            <div class="preset-name">${p.name.toUpperCase()}</div>
            <div class="preset-desc">${p.desc}</div>
            <div class="preset-hint">KLICK FÜR ONE-CLICK SETUP</div>
        `;

        card.addEventListener('click', async () => {
            await api('/preset', 'POST', { preset: key });
            el.presetGrid.querySelectorAll('.preset-card').forEach(c =>
                c.classList.remove('active'));
            card.classList.add('active');
            // Re-fetch um alle UI-Inputs zu updaten
            setTimeout(() => { rendered.profiles = false; fetchStatus(); }, 200);
        });

        el.presetGrid.appendChild(card);
    });
}

// ═══ Profile Grid ══════════════════════════════════════════

function renderProfiles(profiles, active, trtAvail) {
    const order = ['ultra', 'high', 'balanced', 'fast', 'extreme'];
    el.profileGrid.innerHTML = '';

    order.forEach(key => {
        const p = profiles[key];
        if (!p) return;

        const card = document.createElement('div');
        card.className = 'profile-card' + (key === active ? ' active' : '');
        card.dataset.profile = key;

        const badge = p.use_engine ? ' · TRT' : '';

        card.innerHTML = `
            <div class="profile-name">${p.name.toUpperCase()}${badge}</div>
            <div class="profile-sub">${p.desc}</div>

            <div class="profile-metric">
                <span class="profile-metric-label">Auflösung</span>
                <span class="profile-metric-value">${p.imgsz}px</span>
            </div>

            <div class="profile-metric">
                <span class="profile-metric-label">VRAM</span>
                <span class="profile-metric-value">${p.vram_mb} MB</span>
            </div>
            <div class="profile-bar">
                <div class="profile-bar-fill vram" style="width:${p.vram_pct}%"></div>
            </div>

            <div class="profile-metric">
                <span class="profile-metric-label">FPS</span>
                <span class="profile-metric-value">${p.fps}</span>
            </div>
            <div class="profile-bar">
                <div class="profile-bar-fill fps" style="width:${Math.min(100, p.fps/4)}%"></div>
            </div>

            <div class="profile-metric">
                <span class="profile-metric-label">Qualität</span>
                <span class="profile-metric-value">${p.quality_pct}%</span>
            </div>
            <div class="profile-bar">
                <div class="profile-bar-fill quality" style="width:${p.quality_pct}%"></div>
            </div>
        `;

        card.addEventListener('click', async () => {
            await api('/profile', 'POST', { profile: key });
            el.profileGrid.querySelectorAll('.profile-card').forEach(c =>
                c.classList.remove('active'));
            card.classList.add('active');
        });

        el.profileGrid.appendChild(card);
    });
}

// ═══ Chart ═════════════════════════════════════════════════

function drawChart() {
    const canvas = el.chart;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const max = Math.max(60, ...chartData, 1);

    // Grid
    ctx.strokeStyle = 'rgba(120, 140, 200, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        const y = (h / 4) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    ctx.fillStyle = 'rgba(107, 119, 163, 0.6)';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        ctx.fillText((max - (max / 4) * i).toFixed(0), w - 6, (h / 4) * i + 10);
    }

    const step = w / (chartData.length - 1);

    // Fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(0, 255, 157, 0.4)');
    grad.addColorStop(1, 'rgba(0, 255, 157, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, h);
    chartData.forEach((v, i) => ctx.lineTo(i * step, h - (v / max) * h));
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fill();

    // Line
    ctx.strokeStyle = '#00ff9d';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#00ff9d';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    chartData.forEach((v, i) => {
        const x = i * step;
        const y = h - (v / max) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Dot
    const lx = (chartData.length - 1) * step;
    const ly = h - (chartData[chartData.length - 1] / max) * h;
    ctx.fillStyle = '#00ff9d';
    ctx.beginPath();
    ctx.arc(lx, ly, 4, 0, Math.PI * 2);
    ctx.fill();
}

// ═══ Particles Background ══════════════════════════════════

function initParticles() {
    const c = document.getElementById('particleCanvas');
    const ctx = c.getContext('2d');
    const parts = [];

    function resize() {
        c.width = window.innerWidth;
        c.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < 60; i++) {
        parts.push({
            x: Math.random() * c.width,
            y: Math.random() * c.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            size: Math.random() * 1.5 + 0.5,
        });
    }

    function animate() {
        ctx.clearRect(0, 0, c.width, c.height);
        parts.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = c.width;
            if (p.x > c.width) p.x = 0;
            if (p.y < 0) p.y = c.height;
            if (p.y > c.height) p.y = 0;

            ctx.fillStyle = `rgba(0, 255, 157, ${p.size * 0.3})`;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        });

        // Lines between near particles
        for (let i = 0; i < parts.length; i++) {
            for (let j = i + 1; j < parts.length; j++) {
                const dx = parts[i].x - parts[j].x;
                const dy = parts[i].y - parts[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.strokeStyle = `rgba(0, 255, 157, ${(1 - dist / 120) * 0.08})`;
                    ctx.lineWidth = 0.5;
                    ctx.beginPath();
                    ctx.moveTo(parts[i].x, parts[i].y);
                    ctx.lineTo(parts[j].x, parts[j].y);
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(animate);
    }
    animate();
}

// ═══ Event Handlers ════════════════════════════════════════

el.btnStart.addEventListener('click', async () => {
    el.btnStart.disabled = true;
    await api('/start', 'POST');
    fetchStatus();
});

el.btnStop.addEventListener('click', async () => {
    el.btnStop.disabled = true;
    await api('/stop', 'POST');
    fetchStatus();
});

el.btnShutdown.addEventListener('click', async () => {
    if (!confirm('Alles komplett beenden?')) return;
    await api('/shutdown', 'POST');
});

el.confSlider.addEventListener('input', async (e) => {
    const v = parseFloat(e.target.value);
    el.confValue.textContent = `${Math.round(v * 100)}%`;
    await api('/config', 'POST', { conf: v });
});

if (el.thickSlider) {
    el.thickSlider.addEventListener('input', async (e) => {
        const v = parseInt(e.target.value);
        el.thickValue.textContent = `${v}px`;
        await api('/config', 'POST', { box_thickness: v });
    });
}

if (el.maxDetSlider) {
    el.maxDetSlider.addEventListener('input', async (e) => {
        const v = parseInt(e.target.value);
        el.maxDetValue.textContent = v;
        await api('/config', 'POST', { max_detections: v });
    });
}

el.toggleLabels.addEventListener('change', (e) =>
    api('/config', 'POST', { show_labels: e.target.checked }));
if (el.toggleConf) el.toggleConf.addEventListener('change', (e) =>
    api('/config', 'POST', { show_conf: e.target.checked }));
if (el.toggleGlow) el.toggleGlow.addEventListener('change', (e) =>
    api('/config', 'POST', { glow: e.target.checked }));
el.toggleCrosshair.addEventListener('change', (e) =>
    api('/config', 'POST', { show_crosshair: e.target.checked }));
el.toggleHud.addEventListener('change', (e) =>
    api('/config', 'POST', { show_hud_regions: e.target.checked }));

el.colorDots.forEach(dot => {
    dot.addEventListener('click', async (e) => {
        const color = e.target.dataset.color;
        el.colorDots.forEach(d => d.classList.remove('selected'));
        e.target.classList.add('selected');
        await api('/config', 'POST', { color });
    });
});

// Advanced collapse
if (el.advHeader) {
    el.advHeader.addEventListener('click', () => {
        el.advHeader.classList.toggle('collapsed');
        el.advPanel.classList.toggle('collapsed');
    });
}

// ═══ Init ══════════════════════════════════════════════════

async function init() {
    initParticles();

    const s = await api('/status');
    if (s) {
        el.confSlider.value = s.conf;
        el.confValue.textContent = `${Math.round(s.conf * 100)}%`;
        if (el.thickSlider) {
            el.thickSlider.value = s.box_thickness || 3;
            el.thickValue.textContent = `${s.box_thickness || 3}px`;
        }
        if (el.maxDetSlider) {
            el.maxDetSlider.value = s.max_detections || 10;
            el.maxDetValue.textContent = s.max_detections || 10;
        }
        el.toggleLabels.checked = s.show_labels;
        if (el.toggleConf) el.toggleConf.checked = s.show_conf;
        if (el.toggleGlow) el.toggleGlow.checked = s.glow;
        el.toggleCrosshair.checked = s.show_crosshair;
        el.toggleHud.checked = s.show_hud_regions;

        el.colorDots.forEach(d => {
            if (d.dataset.color === s.color) d.classList.add('selected');
        });

        updateUI(s);
    }

    setInterval(fetchStatus, POLL_INTERVAL);
    window.addEventListener('resize', drawChart);
}

init();

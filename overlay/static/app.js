// DBD Survivor Detector — Dashboard JS

const API = '/api';
const POLL_INTERVAL = 300;

// ─── Elements ─────────────────────────────────────────────

const el = {
    statusDot:   document.getElementById('statusDot'),
    statusText:  document.getElementById('statusText'),
    btnStart:    document.getElementById('btnStart'),
    btnStop:     document.getElementById('btnStop'),
    btnShutdown: document.getElementById('btnShutdown'),

    statFps:         document.getElementById('statFps'),
    statAvgFps:      document.getElementById('statAvgFps'),
    statDetections:  document.getElementById('statDetections'),
    statTotal:       document.getElementById('statTotal'),
    statFrames:      document.getElementById('statFrames'),
    statUptime:      document.getElementById('statUptime'),

    modelName:   document.getElementById('modelName'),
    modelBadge:  document.getElementById('modelBadge'),
    infoModel:   document.getElementById('infoModel'),

    confSlider:      document.getElementById('confSlider'),
    confValue:       document.getElementById('confValue'),
    toggleLabels:    document.getElementById('toggleLabels'),
    toggleCrosshair: document.getElementById('toggleCrosshair'),
    toggleHud:       document.getElementById('toggleHud'),
    colorDots:       document.querySelectorAll('.color-dot'),

    chart: document.getElementById('fpsChart'),
    profileGrid: document.getElementById('profileGrid'),
    mirrorOverlay: document.getElementById('mirrorOverlay'),
};

// ─── State ────────────────────────────────────────────────

let currentState = {};
let chartData = Array(60).fill(0);

// ─── API Calls ────────────────────────────────────────────

async function apiCall(path, method = 'GET', data = null) {
    try {
        const opts = { method };
        if (data) {
            opts.headers = { 'Content-Type': 'application/json' };
            opts.body = JSON.stringify(data);
        }
        const res = await fetch(API + path, opts);
        return await res.json();
    } catch (e) {
        console.error('API Error:', e);
        return null;
    }
}

async function fetchStatus() {
    const status = await apiCall('/status');
    if (!status) return;
    currentState = status;
    updateUI(status);
}

// ─── UI Update ────────────────────────────────────────────

function fmtTime(seconds) {
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    if (m < 60) return `${m}m ${s}s`;
    const h = Math.floor(m / 60);
    return `${h}h ${m % 60}m`;
}

function updateUI(s) {
    // Status
    if (s.running) {
        el.statusDot.classList.add('active');
        el.statusText.textContent = 'LÄUFT';
        el.statusText.classList.add('running');
        el.btnStart.disabled = true;
        el.btnStop.disabled = false;
    } else {
        el.statusDot.classList.remove('active');
        el.statusText.textContent = 'BEREIT';
        el.statusText.classList.remove('running');
        el.btnStart.disabled = false;
        el.btnStop.disabled = true;
    }

    // Stats
    el.statFps.textContent = s.fps.toFixed(1);
    el.statAvgFps.textContent = `Ø ${s.avg_fps.toFixed(1)}`;
    el.statDetections.textContent = s.detections;
    el.statTotal.textContent = s.total_detections.toLocaleString();
    el.statFrames.textContent = `${s.frame_count.toLocaleString()} Frames`;
    el.statUptime.textContent = fmtTime(s.uptime);

    // Model
    el.modelName.textContent = s.model || '—';
    el.infoModel.textContent = s.model || '—';

    // Mirror Status Text
    if (el.mirrorOverlay) {
        el.mirrorOverlay.textContent = s.running ? `${s.fps.toFixed(0)} FPS` : 'Inaktiv';
    }

    // Profile Grid (nur einmal rendern)
    if (el.profileGrid && !el.profileGrid.dataset.rendered && s.profiles) {
        renderProfiles(s.profiles, s.profile, s.trt_available);
        el.profileGrid.dataset.rendered = '1';
    } else if (el.profileGrid && s.profiles) {
        // Aktiven Status aktualisieren
        el.profileGrid.querySelectorAll('.profile-card').forEach(card => {
            card.classList.toggle('active', card.dataset.profile === s.profile);
        });
    }

    // Chart data push
    if (s.running) {
        chartData.shift();
        chartData.push(s.fps);
    }
    drawChart();
}

// ─── Profile Grid ─────────────────────────────────────────

function renderProfiles(profiles, active, trtAvail) {
    const order = ['ultra', 'high', 'balanced', 'fast', 'extreme'];
    el.profileGrid.innerHTML = '';

    order.forEach(key => {
        const p = profiles[key];
        if (!p) return;

        const card = document.createElement('div');
        card.className = 'profile-card' + (key === active ? ' active' : '');
        card.dataset.profile = key;

        const engineBadge = p.use_engine ? ' · TRT' : '';
        const engineNote = p.use_engine && !trtAvail
            ? '<div style="color:#ff9500;font-size:10px;margin-top:4px">! TRT nicht verfügbar</div>'
            : '';

        card.innerHTML = `
            <div class="profile-name">${p.name.toUpperCase()}${engineBadge}</div>
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
                <span class="profile-metric-label">FPS (erwartet)</span>
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
            ${engineNote}
        `;

        card.addEventListener('click', async () => {
            await apiCall('/profile', 'POST', { profile: key });
            el.profileGrid.querySelectorAll('.profile-card').forEach(c =>
                c.classList.remove('active'));
            card.classList.add('active');
        });

        el.profileGrid.appendChild(card);
    });
}


// ─── Chart ────────────────────────────────────────────────

function drawChart() {
    const canvas = el.chart;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(120, 140, 200, 0.08)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        const y = (h / 4) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Max FPS
    const max = Math.max(60, ...chartData, 1);

    // Y Labels
    ctx.fillStyle = 'rgba(122, 132, 168, 0.6)';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
        const val = max - (max / 4) * i;
        ctx.fillText(val.toFixed(0), w - 5, (h / 4) * i + 10);
    }

    // Line
    const step = w / (chartData.length - 1);

    // Fill area
    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, 'rgba(0, 255, 136, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 255, 136, 0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(0, h);
    chartData.forEach((v, i) => {
        const x = i * step;
        const y = h - (v / max) * h;
        ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fill();

    // Line
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    chartData.forEach((v, i) => {
        const x = i * step;
        const y = h - (v / max) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Current point
    const lastX = (chartData.length - 1) * step;
    const lastY = h - (chartData[chartData.length - 1] / max) * h;
    ctx.fillStyle = '#00ff88';
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fill();
}

// ─── Event Handlers ───────────────────────────────────────

el.btnStart.addEventListener('click', async () => {
    el.btnStart.disabled = true;
    await apiCall('/start', 'POST');
    fetchStatus();
});

el.btnStop.addEventListener('click', async () => {
    el.btnStop.disabled = true;
    await apiCall('/stop', 'POST');
    fetchStatus();
});

el.btnShutdown.addEventListener('click', async () => {
    if (!confirm('Alles komplett beenden (Overlay + Server)?')) return;
    await apiCall('/shutdown', 'POST');
    el.statusText.textContent = 'HERUNTERGEFAHREN';
    el.btnStart.disabled = true;
    el.btnStop.disabled = true;
});

el.confSlider.addEventListener('input', async (e) => {
    const val = parseFloat(e.target.value);
    el.confValue.textContent = `${Math.round(val * 100)}%`;
    await apiCall('/config', 'POST', { conf: val });
});

el.toggleLabels.addEventListener('change', async (e) => {
    await apiCall('/config', 'POST', { show_labels: e.target.checked });
});

el.toggleCrosshair.addEventListener('change', async (e) => {
    await apiCall('/config', 'POST', { show_crosshair: e.target.checked });
});

el.toggleHud.addEventListener('change', async (e) => {
    await apiCall('/config', 'POST', { show_hud_regions: e.target.checked });
});

el.colorDots.forEach(dot => {
    dot.addEventListener('click', async (e) => {
        const color = e.target.dataset.color;
        el.colorDots.forEach(d => d.classList.remove('selected'));
        e.target.classList.add('selected');
        await apiCall('/config', 'POST', { color });
    });
});

// ─── Init ─────────────────────────────────────────────────

async function init() {
    const s = await apiCall('/status');
    if (s) {
        currentState = s;
        el.confSlider.value = s.conf;
        el.confValue.textContent = `${Math.round(s.conf * 100)}%`;
        el.toggleLabels.checked = s.show_labels;
        el.toggleCrosshair.checked = s.show_crosshair;
        el.toggleHud.checked = s.show_hud_regions;

        el.colorDots.forEach(d => {
            if (d.dataset.color === s.color) d.classList.add('selected');
        });

        updateUI(s);
    }

    // Polling
    setInterval(fetchStatus, POLL_INTERVAL);

    // Re-draw chart on resize
    window.addEventListener('resize', drawChart);
}

init();

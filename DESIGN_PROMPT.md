# DBD SURVIVOR DETECTOR — AI Design Brief

Kopiere diesen Prompt in eine AI wie **v0.dev**, **Lovable**, **Claude**, **Galileo AI**, **Figma AI** oder **Framer AI** um ein alternatives Design zu generieren.

---

## 🎯 The Prompt

```
Design an insanely impressive cyberpunk-neural dashboard for a real-time AI detection system in a horror game.

PROJECT CONTEXT:
DBD Survivor Detector — A YOLOv8 neural network that detects player characters ("survivors") in the horror game "Dead by Daylight" in real-time.
The system runs locally, draws detection boxes as an overlay on the game screen, and controls everything via a web dashboard.

AESTHETIC:
Cyberpunk + Neural-Net + Military-Grade Tactical UI mixed.
Think: Ghost in the Shell meets Deus Ex meets a Warstation UI from Call of Duty.
Dark theme ONLY (never light mode). Holographic, glassmorphism, subtle motion.

COLOR PALETTE:
- Primary (accent green):  #00ff9d  — neural network indicators
- Cyan (data streams):     #00e5ff  — secondary accents
- Magenta (alerts):        #ff0080  — warnings
- Orange:                  #ff9500  — VRAM / resource usage
- Danger red:              #ff2a5b  — critical errors
- Deep background:         #02040c  — near-black
- Card background:         rgba(14, 20, 42, 0.7) with backdrop-blur

TYPOGRAPHY:
- Titles: 'Orbitron', wide letter-spacing (4px+), font-weight 800-900
- Body:   'Rajdhani', semibold for emphasis
- Data:   'JetBrains Mono', all caps with spacing for stats/metrics

LAYOUT (single-page app):

1. ANIMATED TOPBAR (sticky)
   - Left: Hexagonal rotating logo with "DBD SURVIVOR DETECTOR v3"
   - Right: Model status badge (pulsing dot), connection indicator
   - Animated gradient line at bottom (flows left-to-right, green→cyan→magenta)

2. HERO STATUS SECTION
   - Glowing animated border (gradient cycling)
   - Left: Radial progress ring around status dot, big "STANDBY/ACTIVE" text
   - Right: 3 large buttons (START / STOP / EXIT) with shine animation
   - Horizontal scanline moving across periodically

3. QUICK MODES (4 preset cards)
   - STANDARD, COMPETITIVE, MINIMAL, STREAM
   - Each a hover-lifting card with an icon symbol, name, description
   - Clicking instantly applies all settings
   - Active card has bright green border + glow + checkmark

4. LIVE STATS GRID (4 cards)
   - FPS Live / Detections Visible / Total Count / Session Uptime
   - Each with animated counter, glow effect, progress bar below
   - Trend indicators (↑ UP / → STABLE / ↓ DOWN)

5. LIVE NEURAL FEED (left) + FPS CHART (right) — 2-column
   - Video feed with corner brackets, scanning line animation, "LIVE" badge
   - Line chart with glowing green line, gradient fill, grid background

6. PERFORMANCE PROFILES (5 cards: Ultra/High/Balanced/Fast/Extreme)
   - Each card shows: Resolution, VRAM MB, FPS estimate, Quality %
   - Three progress bars per card (VRAM orange→red, FPS cyan→green, Quality purple→cyan)

7. ADVANCED SETTINGS (collapsible panel)
   - Sliders with glowing thumb (confidence, box thickness, max detections)
   - Toggle switches with neon slide animation
   - Color picker with 6 glowing dot options

8. SYSTEM INFO (minimal 3-column grid)
   - Model architecture, performance metrics, server info

ANIMATIONS (critical for the look):
- Particle network background (connected dots with thin lines)
- Subtle noise/grain overlay
- Scanning line effects
- Pulsing indicators
- Hex rotating logo
- Gradient borders that cycle colors
- Hover lift + glow on cards
- Boot-up animation on load ("INITIALIZING NEURAL NETWORK")
- Shine on primary buttons
- Smooth transitions using cubic-bezier easing

EFFECTS:
- Heavy use of box-shadow with primary color for glow
- backdrop-filter: blur on cards
- Subtle radial gradients behind content for depth
- CSS grid pattern overlay
- Scanline overlay (2-3px horizontal lines, animated)

TECHNICAL CONSTRAINTS:
- Must work without external dependencies (no Chart.js, Tailwind, etc.)
- Vanilla HTML/CSS/JS only
- Responsive (min 700px desktop-focused, breakpoint at 1100px for 2-col)
- Uses Google Fonts (Orbitron, JetBrains Mono, Rajdhani)

DELIVERABLE:
Three files: index.html, style.css, app.js
Mock data where needed (status updates, detection counts).

GOAL: Make someone see this and immediately think "this is the most advanced gaming AI dashboard I've ever seen". It should look like a military HUD from a sci-fi movie.
```

---

## 📐 Variation Prompts

### For a cleaner, Apple-style version:

```
Redesign the dashboard in a clean Swiss-design aesthetic.
Minimal, white space, sharp typography (SF Pro or similar).
Keep all features, but make it feel like an enterprise SaaS dashboard
(think Linear, Notion, or Stripe Dashboard).
Accent color: single deep green #00785a. No cyberpunk effects.
```

### For a Matrix-inspired version:

```
Full green-on-black Matrix aesthetic.
Falling code animation in background (katakana characters).
Monospace fonts everywhere. Terminal/CLI vibe.
Add typing animation on status text.
All borders as ASCII-art boxes.
```

### For a Bloodborne / dark fantasy:

```
Medieval dark fantasy UI. Gothic serifs (Cormorant Garamond).
Blood red accents (#8b0000). Parchment textures.
Ornate gold-decorated frames. Candle-flicker animations.
Latin phrases as section dividers.
```

---

## 🎨 Color Palette Alternatives

Change the primary accent if the cyberpunk green feels too generic:

| Mood | Primary | Secondary |
|------|---------|-----------|
| Military Tactical | `#ff9500` orange | `#ffffff` white |
| Aqua Hacker | `#00ffff` cyan | `#ff00ff` magenta |
| Blood Noir | `#ff0033` red | `#ffcc00` gold |
| Purple Nebula | `#a855f7` purple | `#00ffff` cyan |
| Arctic | `#4fc3f7` ice-blue | `#ffffff` white |

---

## 📦 Export-ready file structure

```
overlay/
├── templates/
│   └── index.html           # Main dashboard
├── static/
│   ├── style.css            # All styling
│   └── app.js               # API calls + animations
```

The backend talks on `localhost:8765` with these endpoints:
- `GET /api/status` — current state
- `POST /api/start` — start detection
- `POST /api/stop` — stop detection
- `POST /api/preset` — apply preset
- `POST /api/profile` — change performance profile
- `POST /api/config` — update any setting
- `GET /video_feed` — MJPEG stream of current frame with boxes

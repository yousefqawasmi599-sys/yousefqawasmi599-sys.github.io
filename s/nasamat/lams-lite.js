// LAMS lite: the hand pointer from LAMS (لمس), cut down to what a web game needs.
// One Euro smoothing, the index fingertip, palm-scaled mapping (same arm movement near or far),
// mirroring, a short grace when the hand drops out for a frame. Frames never leave the device.
const V = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const EURO = { minCutoff: 1.0, beta: 4.0, dCutoff: 1.0 };
const SPAN_PALMS = 4.2, LOST_GRACE_MS = 400, MIN_PALM = 0.03;

const alpha = (hz, dt) => 1 / (1 + 1 / (2 * Math.PI * hz) / dt);
class OneEuro2D {
  reset() { this.has = false; }
  filter(x, y, t) {
    if (!this.has) { this.has = true; Object.assign(this, { x, y, dx: 0, dy: 0, t }); return { x, y }; }
    const dt = Math.max(1e-3, (t - this.t) / 1000); this.t = t;
    const ad = alpha(EURO.dCutoff, dt);
    this.dx += ((x - this.x) / dt - this.dx) * ad;
    this.dy += ((y - this.y) / dt - this.dy) * ad;
    const a = alpha(EURO.minCutoff + EURO.beta * Math.hypot(this.dx, this.dy), dt);
    this.x += (x - this.x) * a; this.y += (y - this.y) * a;
    return { x: this.x, y: this.y };
  }
}

// fingertip -> area pixels, in a box around where the hand was first raised; pushing past an edge slides the box
class Mapper {
  reset() { this.anchor = null; this.palm = 0; }
  map(tip, palm, fw, fh, vw, vh) {
    const tx = tip.x, ty = tip.y * (fh / fw);
    this.palm = this.palm ? this.palm * 0.92 + palm * 0.08 : palm;
    if (!this.anchor) this.anchor = { x: tx, y: ty };
    const sx0 = SPAN_PALMS * this.palm, sy0 = sx0 * (vh / vw);
    let sx = 0.5 - (tx - this.anchor.x) / sx0, sy = 0.5 + (ty - this.anchor.y) / sy0;
    if (sx < 0 || sx > 1) { const c = Math.min(1, Math.max(0, sx)); this.anchor.x -= (sx - c) * sx0; sx = c; }
    if (sy < 0 || sy > 1) { const c = Math.min(1, Math.max(0, sy)); this.anchor.y += (sy - c) * sy0; sy = c; }
    return { x: sx * vw, y: sy * vh };
  }
}

function palmSize(im, fw, fh) {
  const ar = fh / fw, d = (a, b) => Math.hypot(im[a].x - im[b].x, (im[a].y - im[b].y) * ar);
  return Math.max(d(0, 5), d(0, 9), d(0, 17), d(5, 17) * 1.15);
}

let loading = null, lastTs = -1;
/** Load the hand model in the background (no camera). Safe to call many times. */
export function preload() {
  if (!loading) loading = (async () => {
    const { FilesetResolver, HandLandmarker } = await import(`${V}/vision_bundle.mjs`);
    const fs = await FilesetResolver.forVisionTasks(`${V}/wasm`);
    const make = (delegate) => HandLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'VIDEO', numHands: 1,
      minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    let lm; try { lm = await make('GPU'); } catch { lm = await make('CPU'); }
    // warm up once so the first real frame doesn't stall while the GPU compiles
    try { const c = document.createElement('canvas'); c.width = c.height = 64; lm.detectForVideo(c, (lastTs = performance.now())); } catch {}
    return lm;
  })().catch((e) => { loading = null; throw e; });
  return loading;
}

/** Camera + tracker. Call only after the player taps "play with camera". */
export async function startPointer(video) {
  const cam = navigator.mediaDevices.getUserMedia({
    audio: false, video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
  });
  let lm;
  try { lm = await preload(); } catch (e) { cam.then((s) => s.getTracks().forEach((t) => t.stop()), () => {}); throw e; }
  const stream = await cam;
  video.srcObject = stream; video.muted = true; video.playsInline = true;
  await video.play();
  const euro = new OneEuro2D(), map = new Mapper();
  let lastVT = -1, lastSeen = -1, point = null;
  return {
    /** call once per animation frame with the play area size; returns the smoothed point or null */
    sample(now, vw, vh) {
      if (video.readyState >= 2 && video.currentTime !== lastVT) {
        lastVT = video.currentTime;
        const ts = Math.max(now, lastTs + 0.01); lastTs = ts;
        const r = lm.detectForVideo(video, ts);
        const fw = video.videoWidth || 640, fh = video.videoHeight || 480;
        let best = null, bp = MIN_PALM;
        for (const im of r.landmarks || []) { const p = palmSize(im, fw, fh); if (p >= bp) { bp = p; best = im; } }
        if (best) {
          const raw = map.map(best[8], bp, fw, fh, vw, vh);
          const f = euro.filter(raw.x / vw, raw.y / vw, now);
          point = { x: f.x * vw, y: f.y * vw }; lastSeen = now;
        } else if (now - lastSeen > LOST_GRACE_MS) { point = null; euro.reset(); map.reset(); }
      }
      return point;
    },
    stop() { stream.getTracks().forEach((t) => t.stop()); video.srcObject = null; },
  };
}

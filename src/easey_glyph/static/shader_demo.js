// EASEy-GLYPH — Tron Grid: Audio-reactive neon perspective grid.
// WebGL2, 12 audio feature uniforms for live visual effects.

var ShaderDemo = (function () {
  "use strict";

  // Feature key mapping: server feature names → uniform names
  var FEATURE_KEYS = [
    "bass", "mid", "treble", "rms", "beat_phase", "onset_strength",
    "spectral_centroid", "spectral_flux", "spectral_flatness",
    "spectral_rolloff", "spectral_bandwidth", "zero_crossing_rate"
  ];
  var UNIFORM_NAMES = [
    "u_bass", "u_mid", "u_treble", "u_rms", "u_phase", "u_onset",
    "u_centroid", "u_flux", "u_flatness", "u_rolloff", "u_bandwidth", "u_zcr"
  ];
  var FEATURE_LABELS = [
    "Bass", "Mid", "Treble", "RMS", "Phase", "Onset",
    "Centroid", "Flux", "Flatness", "Rolloff", "Bandwidth", "ZCR"
  ];
  var FEATURE_DESCS = [
    "Terrain waves", "Horizon glow", "Minor grid lines", "Line brightness",
    "Hue rotation", "Pulse ring", "Color temperature", "Scroll speed",
    "Line softness", "Fog distance", "Grid scale", "Camera height"
  ];

  // ─── GLSL: shared vertex shader (fullscreen quad) ──────────────

  var VERT_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

  // ─── GLSL: Pass 1 — Tron Grid (ray-plane + 12 audio effects) ──

  var FRAG_GRID = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_scroll;
uniform vec2 u_resolution;

// Audio feature uniforms (0.0 when effect disabled)
uniform float u_bass;
uniform float u_mid;
uniform float u_treble;
uniform float u_rms;
uniform float u_phase;
uniform float u_onset;
uniform float u_centroid;
uniform float u_flux;
uniform float u_flatness;
uniform float u_rolloff;
uniform float u_bandwidth;
uniform float u_zcr;
uniform float u_onset_time;

// HSV-based hue rotation — preserves saturation and value
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}
vec3 hueShift(vec3 c, float shift) {
  vec3 hsv = rgb2hsv(c);
  hsv.x = fract(hsv.x + shift);
  return hsv2rgb(hsv);
}

// Grid line with per-axis AA using fwidth of grid coordinates
// Returns line intensity [0,1], properly anti-aliased
float gridLine(vec2 gridUV, vec2 gridFW, float softness) {
  // Distance to nearest line in each axis (normalized 0..0.5)
  float dX = min(gridUV.x, 1.0 - gridUV.x);
  float dZ = min(gridUV.y, 1.0 - gridUV.y);

  // AA width: half the screen-space derivative, clamped to a minimum
  float aaX = max(0.015 * softness, gridFW.x * 0.6);
  float aaZ = max(0.015 * softness, gridFW.y * 0.6);

  float lx = 1.0 - smoothstep(0.0, aaX, dX);
  float lz = 1.0 - smoothstep(0.0, aaZ, dZ);
  float line = max(lx, lz);

  // Fade when cells are sub-pixel (< ~2px per cell)
  float cellPx = 1.0 / max(gridFW.x, gridFW.y);
  line *= smoothstep(1.0, 2.5, cellPx);

  return line;
}

void main() {
  vec2 uv = v_uv * 2.0 - 1.0;
  float aspect = u_resolution.x / u_resolution.y;
  uv.x *= aspect;

  // Camera: height modulated by ZCR
  vec3 ro = vec3(0.0, 2.5 + u_zcr * 2.0, 0.0);
  vec3 lookAt = vec3(0.0, 0.0, 20.0);

  vec3 fwd = normalize(lookAt - ro);
  vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(right, fwd);
  vec3 rd = normalize(fwd * 1.8 + right * uv.x + up * uv.y);

  vec3 col = vec3(0.0);

  // Ray-plane intersection with optional terrain displacement
  float denom = rd.y;
  if (denom < -0.001) {
    // First pass: flat-plane intersection for terrain height estimate
    float t0 = -ro.y / denom;
    vec3 p0 = ro + rd * t0;

    // Terrain wave displacement (bass) — gentle rolling hills
    float wz = p0.z + u_scroll;
    float waveHeight = u_bass * 1.0 * (
      sin(p0.x * 0.4 + wz * 0.25 + u_time * 0.4) * 0.6 +
      sin(p0.x * 0.7 - wz * 0.4 + u_time * 0.7) * 0.4
    );

    // Second pass: intersect with displaced plane at y=waveHeight
    float t = -(ro.y - waveHeight) / denom;
    if (t < 0.0) t = t0;
    vec3 hitPos = ro + rd * t;

    // World-space XZ with scroll
    vec2 worldXZ = hitPos.xz + vec2(0.0, u_scroll);

    // Grid parameters: scale from bandwidth (subtle breathing)
    float gridSize = 0.8 + u_bandwidth * 0.4;
    float majorSize = gridSize * 5.0;

    // Line softness from flatness
    float softness = 1.0 + u_flatness * 3.0;

    // Grid coordinates + screen-space derivatives for AA
    vec2 minorUV = fract(worldXZ / gridSize);
    vec2 minorFW = fwidth(worldXZ / gridSize);
    float line = gridLine(minorUV, minorFW, softness);

    vec2 majorUV = fract(worldXZ / majorSize);
    vec2 majorFW = fwidth(worldXZ / majorSize);
    float majorLine = gridLine(majorUV, majorFW, softness);

    // Distance fog: draw distance from rolloff
    float fogExp = 0.02 + (1.0 - u_rolloff) * 0.06;
    float fog = exp(-t * fogExp);

    // Base neon colors: slightly brighter cyan
    vec3 lineColor = vec3(0.0, 0.9, 1.0);
    vec3 majorColor = vec3(0.1, 0.95, 1.0);

    // Color temperature: cyan ↔ warm amber (centroid) — more saturated warm
    lineColor = mix(lineColor, vec3(1.0, 0.5, 0.05), u_centroid);
    majorColor = mix(majorColor, vec3(1.0, 0.55, 0.08), u_centroid);

    // Hue rotation from beat_phase
    if (u_phase > 0.001) {
      lineColor = hueShift(lineColor, u_phase);
      majorColor = hueShift(majorColor, u_phase);
    }

    // Minor line brightness from treble — dim at rest, treble reveals them
    float minorBright = 0.12 + u_treble * 0.88;

    // Combine: minor lines + major lines
    col = lineColor * line * minorBright + majorColor * majorLine * 1.2;
    col *= fog;

    // Proximity glow: also scaled by minorBright so treble controls ALL minor visibility
    float proxGlow = smoothstep(30.0, 2.0, t) * (0.4 + u_flux * 0.2);
    col += lineColor * line * proxGlow * minorBright * fog;

    // Overall brightness from RMS (higher floor, smoother scaling)
    col *= 0.65 + u_rms * 0.7;

    // Pulse ring from onset — bright expanding ring
    float ringAge = u_time - u_onset_time;
    if (ringAge >= 0.0 && ringAge < 1.2) {
      float ringRadius = ringAge * 25.0;
      float distFromCenter = length(hitPos.xz - vec2(0.0, 8.0));
      float ringDist = abs(distFromCenter - ringRadius);
      float ringWidth = 0.2 + ringAge * 1.0;
      float ring = 1.0 - smoothstep(0.0, ringWidth, ringDist);
      ring *= max(0.0, 1.0 - ringAge / 1.2);
      // Bright white-cyan flash, distinct from grid lines
      vec3 ringColor = vec3(0.6, 1.0, 1.0);
      col += ringColor * ring * fog * 3.0;
    }
  }

  // Horizon glow: tight band at horizon only (rd.y ≈ 0)
  float horizonBand = exp(-abs(rd.y) * 80.0);
  float horizonIntensity = 0.15 + u_mid * 0.45;
  vec3 horizonCol = mix(vec3(0.0, 0.5, 0.6), vec3(0.6, 0.3, 0.05), u_centroid);
  col += horizonCol * horizonBand * horizonIntensity;

  fragColor = vec4(col, 1.0);
}`;

  // ─── GLSL: Pass 2 — Output (ACES tonemap + vignette) ──────────

  var FRAG_OUTPUT = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_color;

vec3 aces(vec3 x) {
  float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
  vec3 col = texture(u_color, v_uv).rgb;

  col = aces(col);

  // Vignette
  vec2 q = v_uv - 0.5;
  float vig = 1.0 - dot(q, q) * 1.6;
  vig = smoothstep(0.0, 1.0, clamp(vig, 0.0, 1.0));
  col *= vig;

  col = pow(max(col, 0.0), vec3(1.0 / 2.2));
  fragColor = vec4(col, 1.0);
}`;

  // ─── ShaderDemo class ─────────────────────────────────────────

  function ShaderDemo(canvasEl) {
    this.canvas = canvasEl;
    this.gl = null;
    this.running = false;
    this.rafId = null;
    this.startTime = 0;
    this._lastFrameTime = 0;

    this.programs = {};
    this.fbos = {};
    this.quadVAO = null;
    this.uniformLocs = {};

    this._resizeObs = null;
    this._needsResize = true;

    // Audio features
    this.features = {};
    this.rawFeatures = {};
    this.enables = {};
    for (var i = 0; i < FEATURE_KEYS.length; i++) {
      var k = FEATURE_KEYS[i];
      this.features[k] = 0;
      this.rawFeatures[k] = 0;
      this.enables[k] = true;
    }

    this._scrollOffset = 0;
    this._onsetActive = false;
    this._onsetTime = -10;

    this._init();
  }

  // Expose static arrays for panel construction
  ShaderDemo.FEATURE_KEYS = FEATURE_KEYS;
  ShaderDemo.FEATURE_LABELS = FEATURE_LABELS;
  ShaderDemo.FEATURE_DESCS = FEATURE_DESCS;

  // ─── Init ─────────────────────────────────────────────────────

  ShaderDemo.prototype._init = function () {
    var gl = this.canvas.getContext("webgl2", {
      alpha: false, antialias: false, depth: false,
      stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: false
    });
    if (!gl) return;
    this.gl = gl;

    this._hasFloatFBO = !!gl.getExtension("EXT_color_buffer_float");
    if (!this._hasFloatFBO) {
      this._hasFloatFBO = !!gl.getExtension("EXT_color_buffer_half_float");
    }

    this._buildQuad();

    this.programs.grid   = this._buildProgram(VERT_SRC, FRAG_GRID);
    this.programs.output = this._buildProgram(VERT_SRC, FRAG_OUTPUT);

    if (!this.programs.grid || !this.programs.output) {
      console.error("[ShaderDemo] Failed to compile shaders");
      this.gl = null;
      return;
    }

    this._cacheUniforms();
    this._createFBOs();

    var self = this;
    this._resizeObs = new ResizeObserver(function () {
      self._needsResize = true;
    });
    this._resizeObs.observe(this.canvas);
  };

  // ─── Quad geometry ────────────────────────────────────────────

  ShaderDemo.prototype._buildQuad = function () {
    var gl = this.gl;
    var verts = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    var vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.quadVAO = vao;
  };

  // ─── Shader compilation ───────────────────────────────────────

  ShaderDemo.prototype._compileShader = function (type, src) {
    var gl = this.gl;
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("[ShaderDemo] Shader compile error:", gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  };

  ShaderDemo.prototype._buildProgram = function (vertSrc, fragSrc) {
    var gl = this.gl;
    var vs = this._compileShader(gl.VERTEX_SHADER, vertSrc);
    var fs = this._compileShader(gl.FRAGMENT_SHADER, fragSrc);
    if (!vs || !fs) return null;
    var prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, "a_pos");
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error("[ShaderDemo] Program link error:", gl.getProgramInfoLog(prog));
      gl.deleteProgram(prog);
      return null;
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  };

  // ─── Uniform caching ─────────────────────────────────────────

  ShaderDemo.prototype._cacheUniforms = function () {
    var gl = this.gl;
    var locs = this.uniformLocs;

    var gp = this.programs.grid;
    locs.grid = {
      u_time:       gl.getUniformLocation(gp, "u_time"),
      u_scroll:     gl.getUniformLocation(gp, "u_scroll"),
      u_resolution: gl.getUniformLocation(gp, "u_resolution"),
      u_onset_time: gl.getUniformLocation(gp, "u_onset_time")
    };
    // Cache all 12 audio feature uniform locations
    for (var i = 0; i < UNIFORM_NAMES.length; i++) {
      locs.grid[UNIFORM_NAMES[i]] = gl.getUniformLocation(gp, UNIFORM_NAMES[i]);
    }

    var op = this.programs.output;
    locs.output = {
      u_color: gl.getUniformLocation(op, "u_color")
    };
  };

  // ─── FBO management ───────────────────────────────────────────

  ShaderDemo.prototype._createFBO = function (w, h) {
    var gl = this.gl;
    var fb = gl.createFramebuffer();
    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (this._hasFloatFBO) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return { fb: fb, tex: tex, w: w, h: h };
  };

  ShaderDemo.prototype._deleteFBO = function (fbo) {
    if (!fbo) return;
    this.gl.deleteFramebuffer(fbo.fb);
    this.gl.deleteTexture(fbo.tex);
  };

  ShaderDemo.prototype._createFBOs = function () {
    var w = this.canvas.width || 1;
    var h = this.canvas.height || 1;
    this.fbos.main = this._createFBO(w, h);
  };

  ShaderDemo.prototype._resizeFBOs = function () {
    var dpr = Math.min(window.devicePixelRatio || 1, 1.0);
    var w = Math.round(this.canvas.clientWidth * dpr);
    var h = Math.round(this.canvas.clientHeight * dpr);
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    if (this.canvas.width === w && this.canvas.height === h) return;
    this.canvas.width = w;
    this.canvas.height = h;

    this._deleteFBO(this.fbos.main);
    this.fbos.main = this._createFBO(w, h);
  };

  // ─── Draw quad ────────────────────────────────────────────────

  ShaderDemo.prototype._drawQuad = function () {
    var gl = this.gl;
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  };

  // ─── Feature smoothing ────────────────────────────────────────

  ShaderDemo.prototype._smoothFeatures = function (dt) {
    for (var i = 0; i < FEATURE_KEYS.length; i++) {
      var key = FEATURE_KEYS[i];
      var raw = this.rawFeatures[key];
      var cur = this.features[key];

      // Phase is a continuous ramp, pass through directly
      if (key === "beat_phase") {
        this.features[key] = raw;
        continue;
      }

      // Per-feature asymmetric smoothing: fast attack, slower release
      var attack = 12.0;
      var release = 5.0;
      if (key === "onset_strength") {
        attack = 25.0;
        release = 3.0;
      } else if (key === "spectral_flux" || key === "spectral_rolloff" ||
                 key === "spectral_bandwidth") {
        // Faster response for effects that feel sluggish when smoothed
        attack = 18.0;
        release = 10.0;
      }

      var rate = raw > cur ? attack : release;
      this.features[key] = cur + (raw - cur) * Math.min(1.0, rate * dt);
    }
  };

  // ─── Render loop ──────────────────────────────────────────────

  ShaderDemo.prototype._render = function (timestamp) {
    if (!this.running) return;
    var gl = this.gl;
    if (!gl) return;

    if (this._needsResize) {
      this._resizeFBOs();
      this._needsResize = false;
    }

    var t = (timestamp - this.startTime) * 0.001;
    var dt = this._lastFrameTime > 0 ? (timestamp - this._lastFrameTime) * 0.001 : 0.016;
    this._lastFrameTime = timestamp;
    if (dt > 0.1) dt = 0.016; // clamp large gaps

    var w = this.canvas.width;
    var h = this.canvas.height;

    // Smooth audio features
    this._smoothFeatures(dt);

    // Onset edge-trigger for pulse ring (use raw value, gate by enable)
    // Higher reset threshold so it re-arms in fast music (psytrance etc.)
    // Minimum 0.25s between triggers to avoid overlapping rings
    var rawOnset = this.rawFeatures.onset_strength;
    var timeSinceLast = t - this._onsetTime;
    if (rawOnset > 0.25 && !this._onsetActive && this.enables.onset_strength && timeSinceLast > 0.25) {
      this._onsetActive = true;
      this._onsetTime = t;
    }
    if (rawOnset < 0.15) {
      this._onsetActive = false;
    }

    // Scroll accumulation: speed = base + flux contribution
    var fluxVal = this.features.spectral_flux * (this.enables.spectral_flux ? 1.0 : 0.0);
    var speed = 2.0 + fluxVal * 6.0;
    this._scrollOffset += speed * dt;

    // --- Pass 1: Grid → FBO ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.main.fb);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.programs.grid);
    var gl_ = this.uniformLocs.grid;
    gl.uniform1f(gl_.u_time, t);
    gl.uniform1f(gl_.u_scroll, this._scrollOffset);
    gl.uniform2f(gl_.u_resolution, w, h);
    gl.uniform1f(gl_.u_onset_time, this._onsetTime);

    // Set all 12 audio feature uniforms (multiplied by enable flag)
    for (var i = 0; i < FEATURE_KEYS.length; i++) {
      var loc = gl_[UNIFORM_NAMES[i]];
      if (loc === null) continue;
      var val = this.features[FEATURE_KEYS[i]] * (this.enables[FEATURE_KEYS[i]] ? 1.0 : 0.0);
      gl.uniform1f(loc, val);
    }

    this._drawQuad();

    // --- Pass 2: Output → screen ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.programs.output);
    var ol = this.uniformLocs.output;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.fbos.main.tex);
    gl.uniform1i(ol.u_color, 0);
    this._drawQuad();

    this.rafId = requestAnimationFrame(this._render.bind(this));
  };

  // ─── Public API ───────────────────────────────────────────────

  ShaderDemo.prototype.isSupported = function () {
    return !!this.gl;
  };

  ShaderDemo.prototype.start = function () {
    if (!this.gl || this.running) return;
    this.running = true;
    this.startTime = performance.now();
    this._lastFrameTime = 0;
    this._scrollOffset = 0;
    this._needsResize = true;
    this.rafId = requestAnimationFrame(this._render.bind(this));
  };

  ShaderDemo.prototype.stop = function () {
    this.running = false;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  };

  ShaderDemo.prototype.updateFeatures = function (obj) {
    if (!obj) return;
    for (var i = 0; i < FEATURE_KEYS.length; i++) {
      var key = FEATURE_KEYS[i];
      if (obj[key] !== undefined) {
        this.rawFeatures[key] = obj[key];
      }
    }
  };

  ShaderDemo.prototype.setEnable = function (key, val) {
    if (this.enables.hasOwnProperty(key)) {
      this.enables[key] = !!val;
    }
  };

  ShaderDemo.prototype.getEnables = function () {
    var copy = {};
    for (var k in this.enables) copy[k] = this.enables[k];
    return copy;
  };

  ShaderDemo.prototype.destroy = function () {
    this.stop();
    if (this._resizeObs) {
      this._resizeObs.disconnect();
      this._resizeObs = null;
    }
    if (this.gl) {
      var gl = this.gl;
      this._deleteFBO(this.fbos.main);
      for (var name in this.programs) {
        if (this.programs[name]) gl.deleteProgram(this.programs[name]);
      }
      if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
      this.gl = null;
    }
  };

  ShaderDemo.isWebGL2Supported = function () {
    try {
      var c = document.createElement("canvas");
      var g = c.getContext("webgl2");
      var ok = !!g;
      if (g) { var ext = g.getExtension("WEBGL_lose_context"); if (ext) ext.loseContext(); }
      return ok;
    } catch (e) { return false; }
  };

  return ShaderDemo;
})();

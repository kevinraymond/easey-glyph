// EASEy-GLYPH — Tron Grid: Neon perspective grid scrolling toward the camera.
// Minimal WebGL2, no audio reactivity. Cyan lines on black, classic aesthetic.

var ShaderDemo = (function () {
  "use strict";

  // ─── GLSL: shared vertex shader (fullscreen quad) ──────────────

  var VERT_SRC = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

  // ─── GLSL: Pass 1 — Tron Grid (ray-plane intersection) ────────

  var FRAG_GRID = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;

void main() {
  vec2 uv = v_uv * 2.0 - 1.0;
  float aspect = u_resolution.x / u_resolution.y;
  uv.x *= aspect;

  // Camera: above the plane, looking forward-down
  vec3 ro = vec3(0.0, 2.5, 0.0);
  vec3 lookAt = vec3(0.0, 0.0, 20.0);

  vec3 fwd = normalize(lookAt - ro);
  vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(right, fwd);
  vec3 rd = normalize(fwd * 1.8 + right * uv.x + up * uv.y);

  vec3 col = vec3(0.0);

  // Ray-plane intersection: y=0
  float denom = rd.y;
  if (denom < -0.001) {
    float t = -ro.y / denom;
    vec3 hitPos = ro + rd * t;

    // World-space XZ with scroll
    float speed = 3.0;
    vec2 worldXZ = hitPos.xz + vec2(0.0, u_time * speed);

    // Grid lines via fract — two scales
    float gridSize = 1.0;
    vec2 gf = fract(worldXZ / gridSize);
    // Distance to nearest grid line in each axis
    float lineX = min(gf.x, 1.0 - gf.x);
    float lineZ = min(gf.y, 1.0 - gf.y);
    float lineDist = min(lineX, lineZ);

    // Anti-aliased line width that thins with distance
    float pixelScale = t * 0.003;
    float lineWidth = max(0.02, pixelScale);
    float line = 1.0 - smoothstep(0.0, lineWidth, lineDist);

    // Major grid every 5 units (brighter)
    float majorSize = 5.0;
    vec2 gfMajor = fract(worldXZ / majorSize);
    float majorX = min(gfMajor.x, 1.0 - gfMajor.x);
    float majorZ = min(gfMajor.y, 1.0 - gfMajor.y);
    float majorDist = min(majorX, majorZ);
    float majorWidth = max(0.004, pixelScale * 0.25);
    float majorLine = 1.0 - smoothstep(0.0, majorWidth, majorDist / majorSize);

    // Distance fog (fade to black)
    float fog = exp(-t * 0.04);

    // Neon color: cyan/teal
    vec3 lineColor = vec3(0.0, 0.85, 0.9);
    vec3 majorColor = vec3(0.1, 0.95, 1.0);

    // Combine: minor lines dimmer, major lines brighter
    col = lineColor * line * 0.5 + majorColor * majorLine * 1.2;
    col *= fog;

    // Proximity glow: brighten lines close to camera
    float proxGlow = smoothstep(30.0, 2.0, t) * 0.4;
    col += lineColor * line * proxGlow * fog;
  }

  // Horizon glow: subtle gradient where plane meets sky
  float horizonMask = smoothstep(-0.02, 0.04, rd.y);
  float horizonGlow = (1.0 - horizonMask) * 0.15;
  col += vec3(0.0, 0.4, 0.5) * horizonGlow;

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

    this._init();
  }

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
      u_resolution: gl.getUniformLocation(gp, "u_resolution")
    };

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
    var w = this.canvas.width;
    var h = this.canvas.height;

    // --- Pass 1: Grid → FBO ---
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos.main.fb);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.programs.grid);
    var gl_ = this.uniformLocs.grid;
    gl.uniform1f(gl_.u_time, t);
    gl.uniform2f(gl_.u_resolution, w, h);
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

  // No-op stub: app.js may call this but grid has no audio reactivity
  ShaderDemo.prototype.updateFeatures = function () {};

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

// EASEy-GLYPH — WebSocket client, canvas, controls, audio mapping, tooltips

(function () {
  "use strict";

  // ─── Constants ──────────────────────────────────────────────────

  var AUDIO_FEATURES = [
    { value: "none", label: "--" },
    { value: "bass", label: "Bass" },
    { value: "mid", label: "Mid" },
    { value: "treble", label: "Treble" },
    { value: "rms", label: "Energy" },
    { value: "beat_phase", label: "Phase" },
    { value: "onset_strength", label: "Onset" },
    { value: "spectral_centroid", label: "Bright" },
    { value: "spectral_flux", label: "Flux" },
    { value: "spectral_flatness", label: "Flat" },
    { value: "spectral_rolloff", label: "Roll" },
    { value: "spectral_bandwidth", label: "Band" },
    { value: "zero_crossing_rate", label: "ZCR" },
  ];

  var CURVE_OPTIONS = [
    { value: "linear", label: "Linear" },
    { value: "ease_in", label: "Ease In" },
    { value: "ease_out", label: "Ease Out" },
    { value: "ease_in_out", label: "Ease In/Out" },
    { value: "exponential", label: "Exponential" },
    { value: "logarithmic", label: "Logarithmic" },
    { value: "threshold", label: "Threshold" },
  ];

  // Which slider keys get audio mapping in the Mappings card
  var MAPPABLE_PARAMS = [
    { key: "fg_brightness", label: "Brightness" },
    { key: "opacity", label: "Opacity" },
    { key: "alpha_curve", label: "Alpha Curve" },
    { key: "render_scale", label: "Scale" },
    { key: "saturation", label: "Saturation" },
    { key: "contrast", label: "Contrast" },
    { key: "gamma", label: "Gamma" },
    { key: "posterize", label: "Posterize" },
    { key: "sharpen", label: "Sharpen" },
    { key: "grain", label: "Grain" },
    { key: "scanlines", label: "Scanlines" },
    { key: "superres", label: "SuperRes" },
    { key: "edge_enhance", label: "Edge On/Off" },
    { key: "edge_intensity", label: "Edge Intensity" },
    { key: "persistence", label: "Persistence" },
    { key: "cfg_scale", label: "CFG Scale" },
  ];

  var MAPPABLE_KEYS = new Set(MAPPABLE_PARAMS.map(function (p) { return p.key; }));

  // ─── Themes ────────────────────────────────────────────────────

  var THEMES = [
    {
      id: "midnight", label: "Midnight", tag: "default",
      accent: "#4488ff", beat: "#ff5577",
      vars: {
        "--accent": "#4488ff", "--accent-dim": "#2266cc", "--accent-rgb": "68, 136, 255",
        "--beat": "#ff5577", "--beat-rgb": "255, 85, 119",
        "--btn-active-text": "#000"
      }
    },
    {
      id: "neon", label: "Neon", tag: "deutan/protan safe",
      accent: "#00aaff", beat: "#ff8800",
      vars: {
        "--accent": "#00aaff", "--accent-dim": "#0077bb", "--accent-rgb": "0, 170, 255",
        "--beat": "#ff8800", "--beat-rgb": "255, 136, 0",
        "--btn-active-text": "#000"
      }
    },
    {
      id: "vapor", label: "Vapor", tag: "tritan safe",
      accent: "#ff44cc", beat: "#00dddd",
      vars: {
        "--accent": "#ff44cc", "--accent-dim": "#bb3399", "--accent-rgb": "255, 68, 204",
        "--beat": "#00dddd", "--beat-rgb": "0, 221, 221",
        "--btn-active-text": "#000"
      }
    },
    {
      id: "amber", label: "Amber", tag: "deutan/protan safe",
      accent: "#ffaa00", beat: "#7744ff",
      vars: {
        "--accent": "#ffaa00", "--accent-dim": "#bb7700", "--accent-rgb": "255, 170, 0",
        "--beat": "#7744ff", "--beat-rgb": "119, 68, 255",
        "--btn-active-text": "#000"
      }
    },
    {
      id: "highcontrast", label: "Hi-Con", tag: "low vision",
      accent: "#ffffff", beat: "#ff2222",
      vars: {
        "--accent": "#ffffff", "--accent-dim": "#aaaaaa", "--accent-rgb": "255, 255, 255",
        "--beat": "#ff2222", "--beat-rgb": "255, 34, 34",
        "--btn-active-text": "#000",
        "--border": "#555", "--slider-track": "#555"
      }
    },
    {
      id: "mono", label: "Mono", tag: "universal",
      accent: "#cccccc", beat: "#ffffff",
      vars: {
        "--accent": "#cccccc", "--accent-dim": "#888888", "--accent-rgb": "204, 204, 204",
        "--beat": "#ffffff", "--beat-rgb": "255, 255, 255",
        "--btn-active-text": "#000"
      }
    }
  ];

  var STORAGE_THEME = "easey-glyph-theme";

  function applyTheme(id) {
    var theme = null;
    for (var i = 0; i < THEMES.length; i++) {
      if (THEMES[i].id === id) { theme = THEMES[i]; break; }
    }
    if (!theme) theme = THEMES[0];

    var root = document.documentElement;
    // Reset theme vars to defaults (midnight) first
    var defaults = THEMES[0].vars;
    for (var k in defaults) root.style.removeProperty(k);
    // Apply selected theme vars
    for (var k in theme.vars) root.style.setProperty(k, theme.vars[k]);

    // Update swatch active states
    var swatches = document.querySelectorAll(".theme-swatch");
    swatches.forEach(function (sw) {
      sw.classList.toggle("active", sw.dataset.theme === theme.id);
    });

    // Re-render correlation matrix with new theme colors
    if (_lastAnalysisResult) renderAnalysisResults(_lastAnalysisResult);

    try { localStorage.setItem(STORAGE_THEME, theme.id); } catch (e) {}
  }

  function initSettings() {
    var modal = document.getElementById("settings-modal");
    var btnOpen = document.getElementById("btn-settings");
    var btnClose = document.getElementById("settings-close");
    var backdrop = modal.querySelector(".settings-backdrop");
    var grid = document.getElementById("theme-grid");

    // Build theme swatches
    THEMES.forEach(function (theme) {
      var btn = document.createElement("button");
      btn.className = "theme-swatch";
      btn.dataset.theme = theme.id;
      btn.innerHTML =
        '<div class="theme-swatch-colors">' +
          '<span class="theme-swatch-dot" style="background:' + theme.accent + '"></span>' +
          '<span class="theme-swatch-dot" style="background:' + theme.beat + '"></span>' +
        '</div>' +
        '<span class="theme-swatch-label">' + theme.label + '</span>' +
        '<span class="theme-swatch-tag">' + theme.tag + '</span>';
      btn.addEventListener("click", function () {
        applyTheme(theme.id);
      });
      grid.appendChild(btn);
    });

    function openModal() { modal.classList.remove("hidden"); }
    function closeModal() { modal.classList.add("hidden"); }

    btnOpen.addEventListener("click", openModal);
    btnClose.addEventListener("click", closeModal);
    backdrop.addEventListener("click", closeModal);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !modal.classList.contains("hidden")) {
        closeModal();
      }
    });

    // Restore saved theme
    var saved = null;
    try { saved = localStorage.getItem(STORAGE_THEME); } catch (e) {}
    applyTheme(saved || "midnight");
  }

  // ─── State ──────────────────────────────────────────────────────

  var ws = null;
  var reconnectTimer = null;
  var previewFrameCount = 0;
  var lastPreviewFormat = "jpeg";
  var activePresetName = "Default";
  var presetDirty = false;
  var firstFrameReceived = false;

  // Multi-client sync: prevent feedback loops when applying peer updates
  var _syncingFromServer = false;
  // Track which sliders are being dragged to avoid fighting with peer updates
  var _draggingKeys = new Set();

  var canvas = document.getElementById("canvas");
  var ctx = canvas.getContext("2d");
  var elLoadingOverlay = document.getElementById("loading-overlay");
  var elBtnRegen = document.getElementById("btn-regenerate");
  var regenPending = false;

  // Per-key mapping UI references for centralized Mappings card
  var mappingUI = {};

  // Cached DOM refs for updateMetrics (avoid per-frame getElementById)
  var elBpm = document.getElementById("bpm-display");
  var elBeatDot = document.getElementById("beat-dot");
  var elStatusRms = document.getElementById("status-rms");
  var elStatusPool = document.getElementById("status-pool");
  var elStatusMode = document.getElementById("status-mode");
  var elStatusFps = document.getElementById("status-fps");
  var elStatusBpm = document.getElementById("status-bpm");
  var elStatusBeatDot = document.getElementById("status-beat-dot");
  var elMeters = {};
  var elMeterVals = {};
  ["bass","mid","treble","rms","beat_phase","onset_strength","spectral_centroid","spectral_flux","spectral_flatness","spectral_rolloff","spectral_bandwidth","zero_crossing_rate"].forEach(function(k) {
    elMeters[k] = document.getElementById("meter-" + k);
    elMeterVals[k] = document.getElementById("mval-" + k);
  });
  var elSrToggle = document.querySelector('.toggle-ctrl[data-key="superres"]');
  var elScale2xRow = document.querySelector('.toggle-ctrl[data-key="pixel_upscale"]');
  var elScale2xCtrlRow = elScale2xRow ? elScale2xRow.closest(".ctrl-row") : null;
  var valEls = {};

  // Badge elements
  var elBadgePerf = document.getElementById("badge-performance");
  var elBadgePresets = document.getElementById("badge-presets");
  var elBadgeAudio = document.getElementById("badge-audio");
  var elBadgeEffects = document.getElementById("badge-effects");
  var elBadgeMappings = document.getElementById("badge-mappings");
  var elBadgeGen = document.getElementById("badge-generation");
  var elBadgeOutput = document.getElementById("badge-output");
  var elBadgeMidi = document.getElementById("badge-midi");

  // ─── Feature Analysis State ───────────────────────────────────

  var ANALYSIS_FEATURE_KEYS = ["bass","mid","treble","rms","beat_phase","onset_strength","spectral_centroid","spectral_flux","spectral_flatness","spectral_rolloff","spectral_bandwidth","zero_crossing_rate"];
  var ANALYSIS_FEATURE_SHORT = ["Bass","Mid","Treb","RMS","Phase","Onset","Bright","Flux","Flat","Roll","Band","ZCR"];
  var analysisState = "idle";  // "idle" | "recording" | "done"
  var _lastAnalysisResult = null;
  var analysisSamples = [];
  var analysisDuration = 60;
  var analysisFrameCount = 0;
  var analysisAutoStop = false;
  var analysisAutoStopTimeout = 5;
  var analysisSilenceStart = null;

  var elBadgeAnalysis = document.getElementById("badge-analysis");
  var elAnalysisDurationSlider = document.getElementById("analysis-duration-slider");
  var elAnalysisDurationVal = document.getElementById("val-analysis-duration");
  var elBtnRecord = document.getElementById("btn-analysis-record");
  var elBtnStop = document.getElementById("btn-analysis-stop");
  var elBtnClear = document.getElementById("btn-analysis-clear");
  var elProgressWrap = document.getElementById("analysis-progress-wrap");
  var elProgressFill = document.getElementById("analysis-progress-fill");
  var elProgressText = document.getElementById("analysis-progress-text");
  var elLogContainer = document.getElementById("analysis-log-container");
  var elLog = document.getElementById("analysis-log");
  var elResults = document.getElementById("analysis-results");
  var elHeadlineValue = document.getElementById("analysis-headline-value");
  var elHeadlineSamples = document.getElementById("analysis-headline-samples");
  var elMatrix = document.getElementById("analysis-matrix");

  // ─── WebSocket ──────────────────────────────────────────────────

  function connect() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws");
    ws.binaryType = "arraybuffer";

    ws.onopen = function () {
      document.getElementById("status-ws").className = "ws-dot connected";
      clearTimeout(reconnectTimer);
      if (elLoadingOverlay && !firstFrameReceived) {
        elLoadingOverlay.querySelector(".loading-text").textContent = "Generating grids\u2026";
      }
      // Re-establish camera mode on reconnect
      if (cameraActive) {
        send({ type: "camera_start" });
      }
    };

    ws.onclose = function () {
      document.getElementById("status-ws").className = "ws-dot disconnected";
      firstFrameReceived = false;
      if (elLoadingOverlay) {
        elLoadingOverlay.classList.remove("hidden");
        elLoadingOverlay.querySelector(".loading-text").textContent = "Reconnecting\u2026";
      }
      reconnectTimer = setTimeout(connect, 2000);
    };

    ws.onerror = function () { ws.close(); };

    ws.onmessage = function (evt) {
      if (evt.data instanceof ArrayBuffer) {
        renderFrame(evt.data);
        previewFrameCount++;
      } else if (typeof evt.data === "string") {
        try {
          var parsed = JSON.parse(evt.data);
          if (parsed.type === "init_state") {
            applyInitState(parsed);
          } else if (parsed.type === "state_update") {
            applyStateUpdate(parsed);
          } else if (parsed.type === "sender_result") {
            handleSenderResult(parsed);
          } else if (parsed.type === "midi_ports") {
            handleMidiPorts(parsed);
          } else if (parsed.bpm !== undefined) {
            // Dismiss loading overlay once pool has generated grids
            if (!firstFrameReceived && parsed.pool_ready) {
              firstFrameReceived = true;
              if (elLoadingOverlay) elLoadingOverlay.classList.add("hidden");
            }
            updateMetrics(parsed);
          }
        } catch (e) { console.error("WS message error:", e); }
      }
    };
  }

  // ─── Canvas ─────────────────────────────────────────────────────

  function renderFrame(buffer) {
    var mime = lastPreviewFormat === "webp" ? "image/webp" : "image/jpeg";
    var blob = new Blob([buffer], { type: mime });
    createImageBitmap(blob).then(function (bmp) {
      if (canvas.width !== bmp.width) canvas.width = bmp.width;
      if (canvas.height !== bmp.height) canvas.height = bmp.height;
      if (lastPreviewFormat === "webp") {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      ctx.drawImage(bmp, 0, 0);
      bmp.close();

      // A/B labels — DOM overlay (updated only when names change)
      var abLabelsEl = document.getElementById("ab-labels");
      if (abLabelsEl) {
        if (abModeActive && abPresetAName && abPresetBName) {
          abLabelsEl.classList.remove("hidden");
          var labelA = document.getElementById("ab-label-a");
          var labelB = document.getElementById("ab-label-b");
          if (labelA && labelA.textContent !== "A: " + abPresetAName)
            labelA.textContent = "A: " + abPresetAName;
          if (labelB && labelB.textContent !== "B: " + abPresetBName)
            labelB.textContent = "B: " + abPresetBName;
        } else {
          abLabelsEl.classList.add("hidden");
        }
      }
    });
  }

  var canvasWrap = document.getElementById("canvas-wrap");
  canvas.addEventListener("click", function () {
    if (document.fullscreenElement) document.exitFullscreen();
    else canvasWrap.requestFullscreen().catch(function () {});
  });

  // ─── Metrics ────────────────────────────────────────────────────

  function updateMetrics(meta) {
    // BPM & beat
    elBpm.textContent = meta.bpm > 0 ? String(Math.round(meta.bpm)).padStart(3, " ") + " BPM" : " -- BPM";

    if (meta.is_beat) {
      elBeatDot.classList.add("active");
      setTimeout(function () { elBeatDot.classList.remove("active"); }, 120);
      if (elStatusBeatDot) {
        elStatusBeatDot.classList.add("active");
        setTimeout(function () { elStatusBeatDot.classList.remove("active"); }, 120);
      }
    }

    // Mirror BPM to status bar
    if (elStatusBpm) {
      elStatusBpm.textContent = meta.bpm > 0 ? Math.round(meta.bpm) + " BPM" : "";
    }

    // Level meters
    if (meta.features) {
      for (var fk in elMeters) {
        var el = elMeters[fk];
        if (el) {
          var pct = (meta.features[fk] || 0) * 100;
          el.style.height = pct.toFixed(0) + "%";
          var valEl = elMeterVals[fk];
          if (valEl) valEl.textContent = pct.toFixed(0);
        }
      }
    }

    // Status bar
    elStatusRms.textContent = "RMS: " + (meta.rms * 100).toFixed(0) + "%";
    if (meta.gen_mode === "realtime") {
      elStatusPool.textContent = "Realtime";
      elStatusPool.classList.remove("filling");
    } else {
      var poolTarget = meta.pool_target || 64;
      var filling = meta.pool < poolTarget * 0.5;
      elStatusPool.textContent = "Pool: " + meta.pool + "/" + poolTarget;
      elStatusPool.classList.toggle("filling", filling);
      if (regenPending && !filling) {
        regenPending = false;
        elBtnRegen.classList.remove("loading");
      }
    }
    elStatusMode.textContent = meta.mode;

    // Update mapping output bars in Mappings card
    if (meta.features) {
      for (var key in mappingUI) {
        var ui = mappingUI[key];
        var source = ui.sourceSelect.value;
        if (source !== "none" && meta.features[source] !== undefined) {
          var featureVal = meta.features[source];
          var mappedVal = meta.mapped && meta.mapped[key] !== undefined ? meta.mapped[key] : null;
          // Update source label with live %
          ui.sourceLabel.textContent = getFeatureLabel(source) + " (" + (featureVal * 100).toFixed(0) + "%)";
          // Update output bar
          if (mappedVal !== null) {
            var slider = document.querySelector('.slider[data-key="' + key + '"]');
            var pct = 0;
            if (slider) {
              var sMin = parseFloat(slider.min), sMax = parseFloat(slider.max);
              if (sMax > sMin) pct = ((typeof mappedVal === "boolean" ? (mappedVal ? 1 : 0) : mappedVal) - sMin) / (sMax - sMin) * 100;
            }
            ui.outputFill.style.width = Math.min(100, Math.max(0, pct)).toFixed(0) + "%";
            ui.outputVal.textContent = formatMappedVal(key, mappedVal);
          }
        } else {
          ui.sourceLabel.textContent = "--";
          ui.outputFill.style.width = "0%";
          ui.outputVal.textContent = "";
        }
      }
    }

    // Update mapped value displays on Effects card sliders
    if (meta.mapped) {
      for (var mkey in meta.mapped) {
        if (!valEls[mkey]) valEls[mkey] = document.getElementById("val-" + mkey);
        var valEl = valEls[mkey];
        if (valEl) {
          valEl.textContent = formatMappedVal(mkey, meta.mapped[mkey]);
          valEl.classList.add("mapped-val");
        }
        // Mark slider row as mapped
        var mSlider = document.querySelector('.slider[data-key="' + mkey + '"]');
        if (mSlider) {
          var mRow = mSlider.closest(".ctrl-row");
          if (mRow) mRow.classList.add("mapped");
        }
      }
      // Sync superres checkbox
      if (meta.mapped.superres !== undefined && elSrToggle) {
        elSrToggle.checked = !!meta.mapped.superres;
      }
    }

    // Sync UI buttons/toggles with server state
    syncServerState(meta);

    // Track preview format
    if (meta.preview_format) {
      lastPreviewFormat = meta.preview_format;
      if (meta.preview_format === "webp") {
        canvas.classList.add("alpha-preview");
      } else {
        canvas.classList.remove("alpha-preview");
      }
    }

    // Track A/B mode from frame meta
    if (meta.ab_mode !== undefined) {
      abModeActive = !!meta.ab_mode;
      var previewCard = document.querySelector(".card--preview");
      if (previewCard) previewCard.classList.toggle("ab-expanded", abModeActive);
    }

    // Update camera stats
    var camStatsEl = document.getElementById("camera-stats");
    if (camStatsEl) {
      if (meta.camera_stats) {
        camStatsEl.textContent = "R:" + meta.camera_stats.recv + " E:" + meta.camera_stats.encoded + " G:" + meta.camera_stats.generated;
        camStatsEl.classList.remove("hidden");
      } else {
        camStatsEl.classList.add("hidden");
      }
    }

    // Update output status
    updateOutputStatus(meta);

    // Update MIDI status
    if (meta.midi_active !== undefined) {
      updateMidiStatus(meta);
    }

    // Disable Scale2x when SuperRes active
    if (elScale2xCtrlRow) {
      var srActive = meta.superres || (meta.mapped && meta.mapped.superres);
      if (srActive) {
        elScale2xCtrlRow.classList.add("disabled");
      } else {
        elScale2xCtrlRow.classList.remove("disabled");
      }
    }

    // Feature analysis recording
    if (analysisState === "recording" && meta.features) {
      var sample = new Float64Array(12);
      for (var ai = 0; ai < ANALYSIS_FEATURE_KEYS.length; ai++) {
        sample[ai] = meta.features[ANALYSIS_FEATURE_KEYS[ai]] || 0;
      }
      analysisSamples.push(sample);
      analysisFrameCount++;

      var targetSamples = analysisDuration * 60;
      var pct = Math.min(100, analysisSamples.length / targetSamples * 100);
      elProgressFill.style.width = pct.toFixed(1) + "%";
      elProgressText.textContent = analysisSamples.length + " samples";

      // Log every 20th frame
      if (analysisFrameCount % 20 === 0) {
        var parts1 = [], parts2 = [];
        for (var li = 0; li < 12; li++) {
          var s = ANALYSIS_FEATURE_SHORT[li] + ":" + sample[li].toFixed(2);
          if (li < 6) parts1.push(s); else parts2.push(s);
        }
        var line1 = document.createElement("div");
        line1.textContent = parts1.join("  ");
        var line2 = document.createElement("div");
        line2.textContent = "  " + parts2.join("  ");
        line2.style.marginBottom = "2px";
        elLog.appendChild(line1);
        elLog.appendChild(line2);
        if (elLog.children.length > 400) {
          elLog.removeChild(elLog.firstChild);
          elLog.removeChild(elLog.firstChild);
        }
        elLogContainer.scrollTop = elLogContainer.scrollHeight;
      }

      // Silence detection for auto-stop
      var silenceText = "";
      if (analysisAutoStop) {
        var rms = meta.features.rms || 0;
        if (rms < 0.02) {
          if (analysisSilenceStart === null) analysisSilenceStart = Date.now();
          var silenceSecs = (Date.now() - analysisSilenceStart) / 1000;
          silenceText = " (silence " + Math.floor(silenceSecs) + "s)";
          if (silenceSecs >= analysisAutoStopTimeout) {
            analysisFinish();
            return;
          }
        } else {
          analysisSilenceStart = null;
        }
      }

      // Update badge
      var elapsed = Math.floor(analysisSamples.length / 60);
      elBadgeAnalysis.textContent = "Recording " + formatDuration(elapsed) + silenceText;

      // Auto-stop on duration cap
      if (analysisSamples.length >= targetSamples) {
        analysisFinish();
      }
    }

    // Update badges
    updateBadges(meta);
  }

  function formatMappedVal(key, val) {
    if (val === "" || val === undefined) return "--";
    if (key === "superres") return val ? "ON" : "off";
    if (key === "edge_enhance") return val ? "ON" : "off";
    if (key === "posterize" || key === "scanlines") {
      return val === 0 ? "off" : String(val);
    }
    if (key === "cfg_scale") return val <= 0 ? "off" : val.toFixed(1);
    if (key === "render_scale") return String(val);
    if (Number.isInteger(val)) return String(val);
    return val.toFixed(1);
  }

  var FEATURE_LABELS = {};
  AUDIO_FEATURES.forEach(function (f) { FEATURE_LABELS[f.value] = f.label; });
  function getFeatureLabel(source) { return FEATURE_LABELS[source] || source; }

  // Preview FPS counter
  setInterval(function () {
    elStatusFps.textContent = previewFrameCount + " fps";
    previewFrameCount = 0;
  }, 1000);

  // ─── Badges ─────────────────────────────────────────────────────

  function updateBadges(meta) {
    // Performance badge
    if (elBadgePerf) {
      var morph = meta.morph || "ambient";
      var gen = meta.gen_mode || "pool";
      var aspect = "1:1";
      var activeAspect = document.querySelector('[data-aspect].active');
      if (activeAspect) aspect = activeAspect.dataset.aspect;
      elBadgePerf.textContent = capitalize(morph) + " | " + capitalize(gen) + " | " + aspect;
    }

    // Presets badge
    if (elBadgePresets) {
      elBadgePresets.textContent = activePresetName;
      if (presetDirty) {
        elBadgePresets.classList.add("dirty");
      } else {
        elBadgePresets.classList.remove("dirty");
      }
    }
    // Preset reset button
    var resetBtn = document.getElementById("preset-reset-btn");
    if (resetBtn) {
      if (presetDirty) { resetBtn.classList.remove("hidden"); } else { resetBtn.classList.add("hidden"); }
    }
    // Dirty class on active preset button
    var activeBtn = document.querySelector(".preset-btn.active");
    if (activeBtn) {
      if (presetDirty) { activeBtn.classList.add("dirty"); } else { activeBtn.classList.remove("dirty"); }
    }

    // Audio badge
    if (elBadgeAudio) {
      var mode = meta.mode || "idle";
      var bpm = meta.bpm > 0 ? String(Math.round(meta.bpm)).padStart(3, " ") + " BPM" : "";
      if (mode === "live") {
        elBadgeAudio.textContent = "Live" + (bpm ? " " + bpm : "");
      } else if (mode === "file") {
        elBadgeAudio.textContent = "File" + (bpm ? " " + bpm : "");
      } else {
        elBadgeAudio.textContent = "Idle";
      }
    }

    // Generation badge
    if (elBadgeGen) {
      var gParts = [];
      var stepsEl = document.querySelector('.slider[data-key="steps"]');
      var solverSel = document.querySelector('.select-ctrl[data-key="solver"]');
      var schedSel = document.querySelector('.select-ctrl[data-key="time_schedule"]');
      var gSteps = stepsEl ? stepsEl.value : "8";
      var gSolver = solverSel ? solverSel.value : "euler";
      var gSched = schedSel ? schedSel.value : "uniform";
      gParts.push(gSteps + " " + capitalize(gSolver));
      if (gSched !== "uniform") gParts.push(capitalize(gSched));
      if (meta.cfg_scale > 0) gParts.push("CFG " + meta.cfg_scale.toFixed(1));
      elBadgeGen.textContent = gParts.join(" | ");
    }

    // Effects badge
    if (elBadgeEffects) {
      var eParts = [];
      if (meta.superres || (meta.mapped && meta.mapped.superres)) {
        eParts.push("SuperRes ON");
      } else {
        var scaleEl = document.querySelector('.slider[data-key="render_scale"]');
        var interpEl = document.querySelector('.select-ctrl[data-key="interp"]');
        var scale = scaleEl ? scaleEl.value : "4";
        var interp = interpEl ? interpEl.value : "bilinear";
        eParts.push(scale + "x " + capitalize(interp));
      }
      elBadgeEffects.textContent = eParts.join(" | ");
    }

    // Mappings badge
    if (elBadgeMappings) {
      var count = 0;
      for (var k in mappingUI) {
        if (mappingUI[k].sourceSelect.value !== "none" && mappingUI[k].enableCheckbox.checked) count++;
      }
      elBadgeMappings.textContent = count + " active";
    }

    // Output badge
    if (elBadgeOutput) {
      if (meta.output_active) {
        var oParts = [];
        if (meta.output_size) oParts.push(meta.output_size[0] + "x" + meta.output_size[1]);
        var outFps = meta.output_fps != null ? meta.output_fps : meta.fps_smooth;
        oParts.push(Math.round(outFps) + "fps");
        if (meta.output_senders && meta.output_senders.length > 0) {
          oParts.push(meta.output_senders.join(", "));
        }
        elBadgeOutput.textContent = oParts.join(" | ");
      } else {
        elBadgeOutput.textContent = "Idle";
      }
    }

    // MIDI badge
    if (elBadgeMidi) {
      if (meta.midi_active) {
        var mCount = 0;
        if (meta.midi_mappings) mCount += Object.keys(meta.midi_mappings).length;
        if (meta.midi_triggers) mCount += Object.keys(meta.midi_triggers).length;
        elBadgeMidi.textContent = "Connected | " + mCount + " mapped";
      } else {
        elBadgeMidi.textContent = "No MIDI";
      }
    }
  }

  function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ""; }

  // ─── Server State Sync ─────────────────────────────────────────

  var _lastMorph = null;
  var _lastGenMode = null;
  var _lastSuperres = null;
  var _srAvailChecked = false;

  function syncServerState(meta) {
    // Morph mode buttons
    if (meta.morph && meta.morph !== _lastMorph) {
      _lastMorph = meta.morph;
      document.querySelectorAll("[data-morph]").forEach(function (b) {
        if (b.dataset.morph === meta.morph) b.classList.add("active");
        else b.classList.remove("active");
      });
    }

    // Gen mode buttons + pool actions
    if (meta.gen_mode && meta.gen_mode !== _lastGenMode) {
      _lastGenMode = meta.gen_mode;
      document.querySelectorAll("[data-gen]").forEach(function (b) {
        if (b.dataset.gen === meta.gen_mode) b.classList.add("active");
        else b.classList.remove("active");
      });
      var poolActions = document.getElementById("pool-actions");
      if (poolActions) poolActions.classList.toggle("disabled", meta.gen_mode === "realtime");
    }

    // SuperRes availability (one-time check)
    if (!_srAvailChecked && meta.has_superres !== undefined) {
      _srAvailChecked = true;
      if (!meta.has_superres && elSrToggle) {
        elSrToggle.disabled = true;
        var row = elSrToggle.closest(".ctrl-row");
        row.classList.add("disabled");
        elSrToggle.title = "Requires --superres checkpoint";
        var hint = document.createElement("span");
        hint.className = "sender-status sender-unavail";
        hint.textContent = "No model loaded";
        row.appendChild(hint);
      }
    }

    // SuperRes checkbox
    if (meta.superres !== undefined && meta.superres !== _lastSuperres) {
      _lastSuperres = meta.superres;
      if (elSrToggle && !(meta.mapped && meta.mapped.superres !== undefined)) {
        elSrToggle.checked = meta.superres;
      }
    }
  }

  // ─── Controls: Sliders ──────────────────────────────────────────

  function formatSliderVal(key, v) {
    if (key === "posterize" || key === "scanlines") return v === 0 ? "off" : String(v);
    if (key === "feedback_strength") return v <= 0 ? "off" : v.toFixed(2);
    if (key === "persistence") return v <= 0 ? "off" : v.toFixed(2);
    if (key === "cfg_scale") return v <= 0 ? "off" : v.toFixed(1);
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(1);
  }

  document.querySelectorAll(".slider").forEach(function (slider) {
    var key = slider.dataset.key;
    var valEl = document.getElementById("val-" + key);

    // Track drag state for sync conflict avoidance
    slider.addEventListener("pointerdown", function () { _draggingKeys.add(key); });
    slider.addEventListener("pointerup", function () { _draggingKeys.delete(key); });
    slider.addEventListener("pointercancel", function () { _draggingKeys.delete(key); });

    slider.addEventListener("input", function () {
      if (_syncingFromServer) return;
      var v = parseFloat(slider.value);
      if (valEl && !valEl.classList.contains("mapped-val")) {
        valEl.textContent = formatSliderVal(key, v);
      }
      var sendValue = v;
      if (key === "posterize" || key === "scanlines") sendValue = v === 0 ? 0 : v;
      // UI: 0=off, 0.95=max feedback → Server: 1.0=off, 0.05=max feedback
      if (key === "feedback_strength") sendValue = 1.0 - v;
      send({ type: "param", key: key, value: sendValue });
      markPresetDirty();
    });
  });

  document.querySelectorAll(".select-ctrl").forEach(function (sel) {
    sel.addEventListener("change", function () {
      if (_syncingFromServer) return;
      send({ type: "param", key: sel.dataset.key, value: sel.value });
      markPresetDirty();
    });
  });

  document.querySelectorAll(".toggle-ctrl").forEach(function (chk) {
    chk.addEventListener("change", function () {
      if (_syncingFromServer) return;
      send({ type: "param", key: chk.dataset.key, value: chk.checked });
      markPresetDirty();
    });
  });

  function markPresetDirty() {
    presetDirty = true;
  }

  // ─── Mode buttons ───────────────────────────────────────────────

  var fileUploadLabel = document.getElementById("file-upload-label");
  var audioClearBtn = document.getElementById("audio-clear");

  function updateFileUploadVisibility() {
    var fileActive = document.getElementById("btn-file").classList.contains("active");
    if (fileActive) {
      fileUploadLabel.classList.remove("inactive");
    } else {
      fileUploadLabel.classList.add("inactive");
    }
  }

  document.querySelectorAll(".mode-btn").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      if (_syncingFromServer) return;
      e.stopPropagation(); // Don't trigger card collapse
      document.querySelectorAll(".mode-btn").forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      send({ type: "mode", mode: btn.dataset.mode });
      updateFileUploadVisibility();
    });
  });

  // ─── Action buttons ─────────────────────────────────────────────

  elBtnRegen.addEventListener("click", function () {
    send({ type: "regenerate" });
    regenPending = true;
    elBtnRegen.classList.add("loading");
  });
  document.getElementById("btn-reseed").addEventListener("click", function () { send({ type: "reseed" }); });

  // ─── Audio file upload ──────────────────────────────────────────

  var fileInput = document.getElementById("file-input");
  fileInput.addEventListener("change", function () {
    if (fileInput.files.length > 0) uploadFile(fileInput.files[0]);
  });

  fileUploadLabel.addEventListener("click", function (e) { e.stopPropagation(); });

  audioClearBtn.addEventListener("click", function () {
    var player = document.getElementById("audio-player");
    player.pause();
    player.src = "";
    document.getElementById("audio-filename").textContent = "";
    var container = document.getElementById("audio-container");
    container.classList.add("hidden");
    audioClearBtn.classList.add("hidden");
    fileInput.value = "";
    document.querySelectorAll(".mode-btn").forEach(function (b) { b.classList.remove("active"); });
    document.getElementById("btn-idle").classList.add("active");
    send({ type: "mode", mode: "idle" });
    updateFileUploadVisibility();
  });

  var dragCounter = 0;
  var dropOverlay = document.getElementById("drop-overlay");

  document.addEventListener("dragenter", function (e) { e.preventDefault(); dragCounter++; dropOverlay.classList.remove("hidden"); });
  document.addEventListener("dragleave", function (e) { e.preventDefault(); dragCounter--; if (dragCounter <= 0) { dropOverlay.classList.add("hidden"); dragCounter = 0; } });
  document.addEventListener("dragover", function (e) { e.preventDefault(); });
  document.addEventListener("drop", function (e) {
    e.preventDefault(); dragCounter = 0; dropOverlay.classList.add("hidden");
    if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
  });

  function uploadFile(file) {
    var container = document.getElementById("audio-container");
    container.classList.remove("hidden");
    container.classList.add("skeleton");

    var formData = new FormData();
    formData.append("file", file);
    fetch("/api/upload", { method: "POST", body: formData })
      .then(function (resp) { return resp.json(); })
      .then(function (data) {
        container.classList.remove("skeleton");
        document.getElementById("audio-filename").textContent = data.filename;
        var player = document.getElementById("audio-player");
        player.src = data.url;
        send({ type: "mode", mode: "file" });
        document.querySelectorAll(".mode-btn").forEach(function (b) { b.classList.remove("active"); });
        document.getElementById("btn-file").classList.add("active");
        updateFileUploadVisibility();
        audioClearBtn.classList.remove("hidden");
      })
      .catch(function (err) {
        console.error("Upload failed:", err);
        container.classList.remove("skeleton");
        container.classList.add("hidden");
      });
  }

  // Audio time sync
  var audioPlayer = document.getElementById("audio-player");
  audioPlayer.addEventListener("timeupdate", function () {
    send({ type: "audio_time", t: audioPlayer.currentTime });
  });

  function audioTimeSync() {
    if (!audioPlayer.paused && audioPlayer.duration) {
      send({ type: "audio_time", t: audioPlayer.currentTime });
    }
    requestAnimationFrame(audioTimeSync);
  }
  requestAnimationFrame(audioTimeSync);

  // ─── Centralized Mappings Card ──────────────────────────────────

  function initMappingsCard() {
    var list = document.getElementById("mapping-list");

    MAPPABLE_PARAMS.forEach(function (param) {
      var key = param.key;
      var slider = document.querySelector('.slider[data-key="' + key + '"]');
      var sMin = slider ? slider.min : "0";
      var sMax = slider ? slider.max : "1";
      var sStep = slider ? slider.step : "0.01";

      // Compact row
      var row = document.createElement("div");
      row.className = "mapping-row dim";
      row.dataset.key = key;

      var enableCb = document.createElement("input");
      enableCb.type = "checkbox";
      enableCb.className = "mapping-enable";
      enableCb.checked = false;

      var nameSpan = document.createElement("span");
      nameSpan.className = "mapping-name";
      nameSpan.textContent = param.label;

      var sourceLabel = document.createElement("span");
      sourceLabel.className = "mapping-source-label";
      sourceLabel.textContent = "--";

      var arrow = document.createElement("span");
      arrow.className = "mapping-arrow";
      arrow.textContent = "\u2192";

      var outputBar = document.createElement("div");
      outputBar.className = "mapping-output-bar";
      var outputFill = document.createElement("div");
      outputFill.className = "mapping-output-fill";
      var outputVal = document.createElement("span");
      outputVal.className = "mapping-output-val";
      outputBar.appendChild(outputFill);
      outputBar.appendChild(outputVal);

      var expandBtn = document.createElement("span");
      expandBtn.className = "mapping-expand";
      expandBtn.textContent = "\u25B6";

      row.appendChild(enableCb);
      row.appendChild(nameSpan);
      row.appendChild(sourceLabel);
      row.appendChild(arrow);
      row.appendChild(outputBar);
      row.appendChild(expandBtn);

      // Detail row (expandable)
      var detail = document.createElement("div");
      detail.className = "mapping-detail";
      detail.dataset.key = key;

      // Source dropdown
      var srcLabel = document.createElement("label");
      srcLabel.textContent = "Source";
      var sourceSelect = document.createElement("select");
      AUDIO_FEATURES.forEach(function (f) {
        var opt = document.createElement("option");
        opt.value = f.value;
        opt.textContent = f.label;
        sourceSelect.appendChild(opt);
      });

      // Curve dropdown
      var curveLabel = document.createElement("label");
      curveLabel.textContent = "Curve";
      var curveSelect = document.createElement("select");
      CURVE_OPTIONS.forEach(function (c) {
        var opt = document.createElement("option");
        opt.value = c.value;
        opt.textContent = c.label;
        curveSelect.appendChild(opt);
      });

      // Lo slider
      var loLabel = document.createElement("label");
      loLabel.textContent = "Lo";
      var loSlider = document.createElement("input");
      loSlider.type = "range";
      loSlider.className = "map-slider";
      loSlider.min = sMin;
      loSlider.max = sMax;
      loSlider.step = sStep;
      loSlider.value = sMin;

      // Hi slider
      var hiLabel = document.createElement("label");
      hiLabel.textContent = "Hi";
      var hiSlider = document.createElement("input");
      hiSlider.type = "range";
      hiSlider.className = "map-slider";
      hiSlider.min = sMin;
      hiSlider.max = sMax;
      hiSlider.step = sStep;
      hiSlider.value = sMax;

      // Range display
      var valsDisplay = document.createElement("span");
      valsDisplay.className = "map-vals";
      valsDisplay.textContent = formatSliderVal(key, parseFloat(sMin)) + " \u2192 " + formatSliderVal(key, parseFloat(sMax));

      // Invert button
      var invertBtn = document.createElement("button");
      invertBtn.className = "map-invert";
      invertBtn.textContent = "Inv";
      invertBtn.title = "Invert mapping";

      // Input threshold sliders (Floor/Ceil)
      var floorLabel = document.createElement("label");
      floorLabel.textContent = "Floor";
      var floorSlider = document.createElement("input");
      floorSlider.type = "range";
      floorSlider.className = "map-slider";
      floorSlider.min = "0";
      floorSlider.max = "1";
      floorSlider.step = "0.01";
      floorSlider.value = "0";

      var ceilLabel = document.createElement("label");
      ceilLabel.textContent = "Ceil";
      var ceilSlider = document.createElement("input");
      ceilSlider.type = "range";
      ceilSlider.className = "map-slider";
      ceilSlider.min = "0";
      ceilSlider.max = "1";
      ceilSlider.step = "0.01";
      ceilSlider.value = "1";

      var threshDisplay = document.createElement("span");
      threshDisplay.className = "map-thresh-display";
      threshDisplay.textContent = "0% \u2192 100%";

      // Row 1: Source + Curve dropdowns (4-col grid)
      var row1 = document.createElement("div");
      row1.className = "map-detail-row";
      row1.appendChild(srcLabel);
      row1.appendChild(sourceSelect);
      row1.appendChild(curveLabel);
      row1.appendChild(curveSelect);

      // Slider grid: Lo/Hi + Floor/Ceil in one grid for aligned columns
      // 6 columns: label slider label slider display button
      var sliderGrid = document.createElement("div");
      sliderGrid.className = "map-detail-grid";
      sliderGrid.appendChild(loLabel);
      sliderGrid.appendChild(loSlider);
      sliderGrid.appendChild(hiLabel);
      sliderGrid.appendChild(hiSlider);
      sliderGrid.appendChild(valsDisplay);
      sliderGrid.appendChild(invertBtn);
      sliderGrid.appendChild(floorLabel);
      sliderGrid.appendChild(floorSlider);
      sliderGrid.appendChild(ceilLabel);
      sliderGrid.appendChild(ceilSlider);
      sliderGrid.appendChild(threshDisplay);

      detail.appendChild(row1);
      detail.appendChild(sliderGrid);

      list.appendChild(row);
      list.appendChild(detail);

      // Store refs
      mappingUI[key] = {
        row: row,
        detail: detail,
        enableCheckbox: enableCb,
        sourceSelect: sourceSelect,
        curveSelect: curveSelect,
        sourceLabel: sourceLabel,
        outputFill: outputFill,
        outputVal: outputVal,
        expandBtn: expandBtn,
        loSlider: loSlider,
        hiSlider: hiSlider,
        valsDisplay: valsDisplay,
        invertBtn: invertBtn,
        floorSlider: floorSlider,
        ceilSlider: ceilSlider,
        threshDisplay: threshDisplay,
        slider: slider,
      };

      // --- Event handlers ---

      expandBtn.addEventListener("click", function () {
        var isOpen = detail.classList.contains("open");
        if (isOpen) {
          detail.classList.remove("open");
          expandBtn.classList.remove("open");
        } else {
          detail.classList.add("open");
          expandBtn.classList.add("open");
        }
      });

      enableCb.addEventListener("change", function () {
        if (_syncingFromServer) return;
        if (!enableCb.checked) {
          clearMapping(key);
        } else if (sourceSelect.value !== "none") {
          activateMapping(key);
        } else {
          // Open detail to pick source
          detail.classList.add("open");
          expandBtn.classList.add("open");
        }
      });

      sourceSelect.addEventListener("change", function () {
        if (_syncingFromServer) return;
        if (sourceSelect.value === "none") {
          clearMapping(key);
        } else {
          enableCb.checked = true;
          activateMapping(key);
        }
      });

      curveSelect.addEventListener("change", function () {
        if (_syncingFromServer) return;
        if (sourceSelect.value !== "none") sendCurrentMapping(key);
      });

      loSlider.addEventListener("input", function () {
        if (_syncingFromServer) return;
        valsDisplay.textContent = formatSliderVal(key, parseFloat(loSlider.value)) + " \u2192 " + formatSliderVal(key, parseFloat(hiSlider.value));
        if (sourceSelect.value !== "none") sendCurrentMapping(key);
      });

      hiSlider.addEventListener("input", function () {
        if (_syncingFromServer) return;
        valsDisplay.textContent = formatSliderVal(key, parseFloat(loSlider.value)) + " \u2192 " + formatSliderVal(key, parseFloat(hiSlider.value));
        if (sourceSelect.value !== "none") sendCurrentMapping(key);
      });

      invertBtn.addEventListener("click", function () {
        if (_syncingFromServer) return;
        invertBtn.classList.toggle("active");
        if (sourceSelect.value !== "none") sendCurrentMapping(key);
      });

      floorSlider.addEventListener("input", function () {
        if (_syncingFromServer) return;
        var f = parseFloat(floorSlider.value);
        var c = parseFloat(ceilSlider.value);
        if (f > c) ceilSlider.value = floorSlider.value;
        threshDisplay.textContent = Math.round(parseFloat(floorSlider.value) * 100) + "% \u2192 " + Math.round(parseFloat(ceilSlider.value) * 100) + "%";
        if (sourceSelect.value !== "none") sendCurrentMapping(key);
      });

      ceilSlider.addEventListener("input", function () {
        if (_syncingFromServer) return;
        var f = parseFloat(floorSlider.value);
        var c = parseFloat(ceilSlider.value);
        if (c < f) floorSlider.value = ceilSlider.value;
        threshDisplay.textContent = Math.round(parseFloat(floorSlider.value) * 100) + "% \u2192 " + Math.round(parseFloat(ceilSlider.value) * 100) + "%";
        if (sourceSelect.value !== "none") sendCurrentMapping(key);
      });
    });
  }

  function activateMapping(key) {
    var ui = mappingUI[key];
    ui.row.classList.remove("dim");
    ui.enableCheckbox.checked = true;
    sendCurrentMapping(key);
    // Mark slider row in Effects as mapped
    if (ui.slider) {
      var ctrlRow = ui.slider.closest(".ctrl-row");
      if (ctrlRow) ctrlRow.classList.add("mapped");
    }
  }

  function clearMapping(key) {
    var ui = mappingUI[key];
    ui.sourceSelect.value = "none";
    ui.enableCheckbox.checked = false;
    ui.row.classList.add("dim");
    ui.sourceLabel.textContent = "--";
    ui.outputFill.style.width = "0%";
    ui.outputVal.textContent = "";
    ui.invertBtn.classList.remove("active");
    ui.curveSelect.value = "linear";
    ui.floorSlider.value = "0";
    ui.ceilSlider.value = "1";
    ui.threshDisplay.textContent = "0% \u2192 100%";
    var detail = ui.detail;
    detail.classList.remove("open");
    ui.expandBtn.classList.remove("open");

    // Restore normal val display from slider
    var valEl = document.getElementById("val-" + key);
    if (valEl && ui.slider) {
      valEl.classList.remove("mapped-val");
      valEl.textContent = formatSliderVal(key, parseFloat(ui.slider.value));
    }
    // Unmark slider row
    if (ui.slider) {
      var ctrlRow = ui.slider.closest(".ctrl-row");
      if (ctrlRow) ctrlRow.classList.remove("mapped");
    }

    send({ type: "mapping", key: key, source: "none", min: 0, max: 0 });
    markPresetDirty();
  }

  function sendCurrentMapping(key) {
    var ui = mappingUI[key];
    send({
      type: "mapping",
      key: key,
      source: ui.sourceSelect.value,
      min: parseFloat(ui.loSlider.value),
      max: parseFloat(ui.hiSlider.value),
      invert: ui.invertBtn.classList.contains("active"),
      curve: ui.curveSelect.value,
      lo_thresh: parseFloat(ui.floorSlider.value),
      hi_thresh: parseFloat(ui.ceilSlider.value),
    });
    markPresetDirty();
  }

  // ─── Help / Tooltips ────────────────────────────────────────────

  function initHelp() {
    document.querySelectorAll("[data-tip]").forEach(function (label) {
      var tip = label.getAttribute("data-tip");
      if (!tip) return;

      var helpText = document.createElement("div");
      helpText.className = "help-text";
      helpText.textContent = tip;

      var row = label.closest(".ctrl-row");
      if (!row) return;

      row.after(helpText);
    });

    var helpBtn = document.getElementById("btn-help");
    if (helpBtn) {
      helpBtn.addEventListener("click", function () {
        document.body.classList.toggle("show-help");
        helpBtn.classList.toggle("active");
      });
    }

    document.addEventListener("keydown", function (e) {
      if (e.key === "h" && !e.ctrlKey && !e.metaKey &&
          e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT" && e.target.tagName !== "TEXTAREA") {
        document.body.classList.toggle("show-help");
        var btn = document.getElementById("btn-help");
        if (btn) btn.classList.toggle("active");
      }
    });
  }

  // ─── Grid Morph ───────────────────────────────────────────────

  document.querySelectorAll("[data-morph]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      if (_syncingFromServer) return;
      document.querySelectorAll("[data-morph]").forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      send({ type: "morph", mode: btn.dataset.morph });
      markPresetDirty();
    });
  });

  // ─── Gen Mode ──────────────────────────────────────────────────

  document.querySelectorAll("[data-gen]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      if (_syncingFromServer) return;
      document.querySelectorAll("[data-gen]").forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      var mode = btn.dataset.gen;
      send({ type: "gen_mode", mode: mode });
      var poolActions = document.getElementById("pool-actions");
      if (poolActions) poolActions.classList.toggle("disabled", mode === "realtime");
    });
  });

  // ─── Aspect Ratio ──────────────────────────────────────────────

  document.querySelectorAll("[data-aspect]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      if (_syncingFromServer) return;
      document.querySelectorAll("[data-aspect]").forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      send({ type: "aspect", aspect: btn.dataset.aspect });
    });
  });

  // ─── Presets ──────────────────────────────────────────────────

  var PRESET_DEFAULTS = {
    params: {
      fg_brightness: 1.0, render_scale: 4, sharpen: 0, saturation: 1.0,
      contrast: 1.0, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0,
      opacity: 1.0, alpha_curve: 1.0, edge_intensity: 1.0, feedback_strength: 1.0, persistence: 0,
      onset_threshold: 1.4, img2img_strength: 0.5, cfg_scale: 0,
    },
    toggles: { pixel_upscale: false, superres: false, alpha_preview: false },
    selects: { edge_enhance: "off", cfg_audio: "random" },
    morph: "ambient",
  };

  var PRESETS = [
    {
      name: "Default",
      tip: "All defaults, no mappings, no effects",
      params: {},
      toggles: {},
      selects: {},
      mappings: {},
    },
    {
      name: "Bass Pulse",
      tip: "Simple kick-driven punch — contrast and gamma on bass",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 0, saturation: 1.0, contrast: 1.0, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0, opacity: 1.0 },
      toggles: { pixel_upscale: false },
      selects: {},
      morph: "beat",
      mappings: {
        contrast: { source: "bass", min: 0.95, max: 1.5, curve: "ease_out" },
        gamma:    { source: "bass", min: 0.9, max: 1.15, curve: "ease_out" },
      },
    },
    {
      name: "Classic Reactive",
      tip: "Core audio features — onset punch, mid color, treble detail, phase breathing, bass texture, RMS presence",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 0, saturation: 1.0, contrast: 1.0, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0, opacity: 1.0 },
      toggles: { pixel_upscale: false },
      selects: {},
      morph: "beat",
      mappings: {
        contrast:   { source: "onset_strength",  min: 0.95, max: 1.5, curve: "ease_out",
                      lo_thresh: 0.1 },
        saturation: { source: "mid",             min: 0.7,  max: 1.5 },
        sharpen:    { source: "treble",          min: 0,    max: 100, curve: "ease_out",
                      lo_thresh: 0.15, hi_thresh: 0.75 },
        gamma:      { source: "beat_phase",      min: 0.85, max: 1.2, curve: "ease_in_out" },
        grain:      { source: "bass",            min: 0,    max: 4,   curve: "ease_in",
                      lo_thresh: 0.3,  hi_thresh: 0.9 },
        opacity:    { source: "rms",             min: 0.75, max: 1.0, curve: "ease_out",
                      lo_thresh: 0.05 },
      },
    },
    {
      name: "Dense Spectrum",
      tip: "Six spectral features — flux contrast, flatness color, ZCR sharpness, centroid gamma, bandwidth grain, rolloff opacity",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 0, saturation: 1.0, contrast: 1.0, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0, opacity: 1.0 },
      toggles: { pixel_upscale: false },
      selects: {},
      morph: "beat",
      mappings: {
        contrast:   { source: "spectral_flux",      min: 0.9,  max: 1.5, curve: "ease_out",
                      lo_thresh: 0.15, hi_thresh: 0.70 },
        saturation: { source: "spectral_flatness",   min: 0.4,  max: 1.5, invert: true,
                      lo_thresh: 0.05, hi_thresh: 0.55 },
        sharpen:    { source: "zero_crossing_rate",  min: 0,    max: 120,
                      lo_thresh: 0.10, hi_thresh: 0.65 },
        gamma:      { source: "spectral_centroid",   min: 0.8,  max: 1.25, curve: "ease_in_out",
                      lo_thresh: 0.15, hi_thresh: 0.75 },
        grain:      { source: "spectral_bandwidth",  min: 0,    max: 4,   curve: "ease_in",
                      lo_thresh: 0.20, hi_thresh: 0.80 },
        opacity:    { source: "spectral_rolloff",    min: 0.7,  max: 1.0, curve: "ease_in_out",
                      lo_thresh: 0.10, hi_thresh: 0.80 },
      },
    },
    {
      name: "Retro CRT",
      tip: "Scanlines tighten on loud, posterize crushes tonal, grain on flux, bass contrast",
      params: { fg_brightness: 1.0, render_scale: 3, sharpen: 0, saturation: 0.8, contrast: 1.3, gamma: 0.9, posterize: 4, grain: 3, scanlines: 3, opacity: 1.0 },
      toggles: { pixel_upscale: true },
      selects: {},
      morph: "beat",
      mappings: {
        grain:     { source: "spectral_flux",     min: 0, max: 6,   curve: "ease_out",
                     lo_thresh: 0.10, hi_thresh: 0.70 },
        scanlines: { source: "rms",              min: 2, max: 6,   invert: true,
                     lo_thresh: 0.05, hi_thresh: 0.80 },
        posterize: { source: "spectral_flatness", min: 2, max: 6,   curve: "ease_in",
                     lo_thresh: 0.10, hi_thresh: 0.60 },
        contrast:  { source: "bass",             min: 1.1, max: 1.5, curve: "ease_out",
                     lo_thresh: 0.20, hi_thresh: 0.80 },
      },
    },
    {
      name: "Dark Noir",
      tip: "Desaturated, high contrast — transient gamma flash, bass crunch, treble edge",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 40, saturation: 0.15, contrast: 1.6, gamma: 0.8, posterize: 0, grain: 4, scanlines: 0, opacity: 1.0 },
      toggles: { pixel_upscale: false },
      selects: { edge_enhance: "all" },
      morph: "beat",
      mappings: {
        contrast: { source: "bass",            min: 1.3, max: 1.9, curve: "ease_out",
                    lo_thresh: 0.15, hi_thresh: 0.80 },
        sharpen:  { source: "treble",          min: 20,  max: 90,
                    lo_thresh: 0.10, hi_thresh: 0.70 },
        gamma:    { source: "onset_strength",  min: 0.65, max: 0.95, curve: "ease_out",
                    lo_thresh: 0.15 },
      },
    },
    {
      name: "Pixel Pop",
      tip: "Bass = chunky pixels, mid = vivid, onset = contrast pop",
      params: { fg_brightness: 1.0, render_scale: 2, sharpen: 0, saturation: 1.8, contrast: 1.3, gamma: 1.1, posterize: 0, grain: 0, scanlines: 0, opacity: 1.0 },
      toggles: { pixel_upscale: true },
      selects: {},
      morph: "beat",
      mappings: {
        render_scale: { source: "bass",            min: 2,   max: 6,   invert: true },
        saturation:   { source: "mid",             min: 1.0, max: 2.2, curve: "ease_out",
                        lo_thresh: 0.10, hi_thresh: 0.80 },
        contrast:     { source: "onset_strength",  min: 1.1, max: 1.5, curve: "ease_out",
                        lo_thresh: 0.20 },
      },
    },
    {
      name: "Dream Haze",
      tip: "Soft and dreamy — gentle breathing, energy presence, spectral color",
      params: { fg_brightness: 1.1, render_scale: 8, sharpen: 0, saturation: 1.2, contrast: 0.85, gamma: 1.3, posterize: 0, grain: 0, scanlines: 0, opacity: 0.9 },
      toggles: { pixel_upscale: false },
      selects: {},
      morph: "ambient",
      mappings: {
        gamma:      { source: "beat_phase",        min: 0.95, max: 1.35, curve: "ease_in_out" },
        saturation: { source: "spectral_centroid", min: 0.9,  max: 1.5,  curve: "ease_out",
                      lo_thresh: 0.10, hi_thresh: 0.80 },
        opacity:    { source: "rms",              min: 0.7,  max: 0.95, curve: "ease_out",
                      lo_thresh: 0.05 },
      },
    },
    {
      name: "Flux Storm",
      tip: "Chaotic — flux drives sharpen, onset contrast, bass grain, flatness bit-crush",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 60, saturation: 1.4, contrast: 1.2, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0, opacity: 1.0 },
      toggles: { pixel_upscale: false },
      selects: { edge_enhance: "all" },
      morph: "beat",
      mappings: {
        sharpen:   { source: "spectral_flux",     min: 0,   max: 180, curve: "ease_out",
                     lo_thresh: 0.10, hi_thresh: 0.75 },
        contrast:  { source: "onset_strength",    min: 1.0, max: 1.7, curve: "ease_out",
                     lo_thresh: 0.15 },
        posterize: { source: "spectral_flatness", min: 0,   max: 5,   curve: "exponential",
                     lo_thresh: 0.20, hi_thresh: 0.70 },
        grain:     { source: "bass",             min: 0,   max: 8,   curve: "ease_in",
                     lo_thresh: 0.30, hi_thresh: 0.90 },
      },
    },
    {
      name: "Texture Ride",
      tip: "Grain from noisiness, gamma from rolloff, bandwidth vivid, bass contrast",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 30, saturation: 1.1, contrast: 1.1, gamma: 1.0, posterize: 0, grain: 6, scanlines: 0, opacity: 0.95 },
      toggles: { pixel_upscale: false },
      selects: {},
      morph: "beat",
      mappings: {
        grain:      { source: "spectral_flatness",  min: 0,    max: 8,   curve: "ease_out",
                      lo_thresh: 0.05, hi_thresh: 0.60 },
        gamma:      { source: "spectral_rolloff",   min: 0.8,  max: 1.3, curve: "ease_in_out",
                      lo_thresh: 0.15, hi_thresh: 0.70 },
        saturation: { source: "spectral_bandwidth", min: 0.7,  max: 1.6,
                      lo_thresh: 0.10, hi_thresh: 0.70 },
        contrast:   { source: "bass",              min: 0.95, max: 1.3, curve: "ease_out",
                      lo_thresh: 0.20, hi_thresh: 0.80 },
      },
    },
    {
      name: "Transient Edge",
      tip: "ZCR sharpens on transients, bandwidth drives saturation, onset contrast, rolloff gamma",
      params: { fg_brightness: 1.0, render_scale: 3, sharpen: 80, saturation: 1.3, contrast: 1.2, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0, opacity: 1.0 },
      toggles: { pixel_upscale: false },
      selects: { edge_enhance: "all" },
      morph: "beat",
      mappings: {
        sharpen:    { source: "zero_crossing_rate", min: 0,    max: 160, curve: "ease_out",
                      lo_thresh: 0.10, hi_thresh: 0.70 },
        saturation: { source: "spectral_bandwidth", min: 0.7,  max: 1.8,
                      lo_thresh: 0.10, hi_thresh: 0.70 },
        contrast:   { source: "onset_strength",     min: 1.0,  max: 1.5, curve: "ease_out",
                      lo_thresh: 0.15 },
        gamma:      { source: "spectral_rolloff",   min: 0.85, max: 1.15, invert: true, curve: "ease_in_out",
                      lo_thresh: 0.15, hi_thresh: 0.75 },
      },
    },
    {
      name: "SuperRes Snap",
      tip: "CNN super-resolution snaps on with kicks — chunky when quiet, detailed on bass",
      params: { fg_brightness: 1.0, render_scale: 2, sharpen: 0, saturation: 1.0, contrast: 1.0, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0, opacity: 1.0 },
      toggles: { pixel_upscale: false, superres: false },
      selects: {},
      morph: "beat",
      mappings: {
        superres:   { source: "bass",            min: 0, max: 1,   curve: "ease_in",
                      lo_thresh: 0.25 },
        contrast:   { source: "onset_strength",  min: 0.95, max: 1.4, curve: "ease_out",
                      lo_thresh: 0.15 },
        saturation: { source: "rms",            min: 0.8,  max: 1.3, curve: "ease_out",
                      lo_thresh: 0.05 },
      },
    },
    {
      name: "Phase Breath",
      tip: "Coordinated breathing — gamma and color lift toward next beat, onset punch, phase persistence",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 0, saturation: 1.0, contrast: 1.0, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0, opacity: 0.95 },
      toggles: { pixel_upscale: false },
      selects: {},
      morph: "beat",
      mappings: {
        gamma:       { source: "beat_phase",      min: 0.8,  max: 1.25, curve: "ease_in_out" },
        saturation:  { source: "beat_phase",      min: 0.7,  max: 1.4,  curve: "ease_in_out" },
        contrast:    { source: "onset_strength",  min: 0.95, max: 1.4,  curve: "ease_out",
                       lo_thresh: 0.15 },
        persistence: { source: "beat_phase",      min: 0.1,  max: 0.6 },
      },
    },
    {
      name: "Lo-Fi Glitch",
      tip: "Posterize on hits, scanlines on kicks, grain on flux, rolloff gamma",
      params: { fg_brightness: 1.0, render_scale: 3, sharpen: 0, saturation: 1.0, contrast: 1.0, gamma: 1.0, posterize: 3, grain: 2, scanlines: 2, opacity: 1.0 },
      toggles: { pixel_upscale: true },
      selects: {},
      morph: "beat",
      mappings: {
        posterize: { source: "onset_strength",  min: 0, max: 4,    curve: "ease_out",
                     lo_thresh: 0.15 },
        scanlines: { source: "bass",           min: 0, max: 3,
                     lo_thresh: 0.20, hi_thresh: 0.80 },
        grain:     { source: "spectral_flux",  min: 0, max: 5,    curve: "ease_out",
                     lo_thresh: 0.10, hi_thresh: 0.70 },
        gamma:     { source: "spectral_rolloff", min: 0.7, max: 1.15, curve: "ease_in_out",
                     lo_thresh: 0.15, hi_thresh: 0.70 },
      },
    },
    {
      name: "Feedback Drift",
      tip: "Gentle evolution with feedback — dreamy continuity, spectral color, phase opacity",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 0, saturation: 1.15, contrast: 1.0, gamma: 1.1, posterize: 0, grain: 0, scanlines: 0, opacity: 0.9, feedback_strength: 0.35, persistence: 0.6 },
      toggles: { pixel_upscale: false },
      selects: {},
      morph: "beat",
      mappings: {
        saturation: { source: "spectral_centroid", min: 0.85, max: 1.4, curve: "ease_out",
                      lo_thresh: 0.10, hi_thresh: 0.75 },
        gamma:      { source: "rms",              min: 0.95, max: 1.2, curve: "ease_out",
                      lo_thresh: 0.05 },
        opacity:    { source: "beat_phase",       min: 0.75, max: 0.95, curve: "ease_in_out" },
      },
    },
    {
      name: "Feedback Storm",
      tip: "Aggressive feedback — bass crunch, flux sharpen, onset grain bursts",
      params: { fg_brightness: 1.0, render_scale: 4, sharpen: 50, saturation: 1.3, contrast: 1.2, gamma: 1.0, posterize: 0, grain: 0, scanlines: 0, opacity: 1.0, feedback_strength: 0.7, persistence: 0.3 },
      toggles: { pixel_upscale: false },
      selects: { edge_enhance: "all" },
      morph: "beat",
      mappings: {
        contrast: { source: "bass",            min: 1.0, max: 1.7, curve: "ease_out",
                    lo_thresh: 0.15, hi_thresh: 0.80 },
        sharpen:  { source: "spectral_flux",   min: 0,   max: 130, curve: "ease_out",
                    lo_thresh: 0.10, hi_thresh: 0.70 },
        grain:    { source: "onset_strength",  min: 0,   max: 6,   curve: "ease_in",
                    lo_thresh: 0.25 },
      },
    },
    {
      name: "Ghost Layer",
      tip: "Alpha transparency for VJ layering — kick opacity, spectral color, phase breathing",
      params: { fg_brightness: 1.1, render_scale: 4, sharpen: 0, saturation: 1.2, contrast: 1.15, gamma: 1.1, posterize: 0, grain: 0, scanlines: 0, opacity: 0.35, alpha_curve: 0.6 },
      toggles: { pixel_upscale: false, alpha_preview: true },
      selects: {},
      morph: "beat",
      mappings: {
        opacity:    { source: "bass",              min: 0.15, max: 0.9, curve: "ease_out",
                      lo_thresh: 0.05 },
        saturation: { source: "spectral_centroid", min: 0.7,  max: 1.6, curve: "ease_out",
                      lo_thresh: 0.10, hi_thresh: 0.75 },
        gamma:      { source: "beat_phase",        min: 0.85, max: 1.2, curve: "ease_in_out" },
        contrast:   { source: "onset_strength",    min: 1.0,  max: 1.4, curve: "ease_out",
                      lo_thresh: 0.20 },
      },
    },
    {
      name: "Darkpsy Acid",
      tip: "Nine uncorrelated voices — kick crunch, phase pump, acid color, flux sharpen, noise grain, onset crush, rolloff presence, treble edges, centroid edge depth",
      params: { fg_brightness: 1.0, render_scale: 3, sharpen: 30, saturation: 0.9, contrast: 1.3, gamma: 0.85, posterize: 0, grain: 2, scanlines: 0, opacity: 1.0, edge_intensity: 0.8 },
      toggles: { pixel_upscale: false },
      selects: { edge_enhance: "all" },
      morph: "beat",
      mappings: {
        contrast:       { source: "bass",              min: 1.1,  max: 1.8,  curve: "ease_out",
                          lo_thresh: 0.15, hi_thresh: 0.85 },
        gamma:          { source: "beat_phase",        min: 0.7,  max: 1.0,  curve: "ease_in_out" },
        saturation:     { source: "mid",               min: 0.5,  max: 1.6,  curve: "ease_out",
                          lo_thresh: 0.10, hi_thresh: 0.75 },
        sharpen:        { source: "spectral_flux",     min: 0,    max: 150,  curve: "ease_out",
                          lo_thresh: 0.10, hi_thresh: 0.70 },
        grain:          { source: "spectral_flatness", min: 0,    max: 7,    curve: "ease_out",
                          lo_thresh: 0.05, hi_thresh: 0.55 },
        posterize:      { source: "onset_strength",    min: 0,    max: 4,    curve: "ease_in",
                          lo_thresh: 0.20 },
        opacity:        { source: "spectral_rolloff",   min: 0.65, max: 1.0,  curve: "ease_out",
                          lo_thresh: 0.15, hi_thresh: 0.65 },
        edge_enhance:   { source: "treble",            min: 0,    max: 1,    curve: "ease_in",
                          lo_thresh: 0.25, hi_thresh: 0.80 },
        edge_intensity: { source: "spectral_centroid", min: 0.3,  max: 1.0,  curve: "ease_out",
                          lo_thresh: 0.10, hi_thresh: 0.70 },
      },
    },
  ];

  // ─── A/B Comparison ──────────────────────────────────────────

  var abModeActive = false;
  var abPresetAName = "";
  var abPresetBName = "";

  function resolvePreset(preset) {
    /**
     * Build a resolved preset snapshot object (params, toggles, selects, morph, mappings)
     * by merging PRESET_DEFAULTS with the preset overrides. Does NOT send any WS messages
     * or touch the DOM — pure data resolution for A/B mode.
     */
    var params = {};
    for (var dk in PRESET_DEFAULTS.params) params[dk] = PRESET_DEFAULTS.params[dk];
    if (preset.params) { for (var pk in preset.params) params[pk] = preset.params[pk]; }

    var toggles = {};
    for (var dtk in PRESET_DEFAULTS.toggles) toggles[dtk] = PRESET_DEFAULTS.toggles[dtk];
    if (preset.toggles) { for (var tk in preset.toggles) toggles[tk] = preset.toggles[tk]; }

    var selects = {};
    for (var dsk in PRESET_DEFAULTS.selects) selects[dsk] = PRESET_DEFAULTS.selects[dsk];
    if (preset.selects) { for (var sk in preset.selects) selects[sk] = preset.selects[sk]; }

    var morph = preset.morph || PRESET_DEFAULTS.morph;

    // Build mappings in server format
    var mappings = {};
    if (preset.mappings) {
      for (var mkey in preset.mappings) {
        var m = preset.mappings[mkey];
        mappings[mkey] = {
          source: m.source,
          min: m.min,
          max: m.max,
          invert: !!m.invert,
          curve: m.curve || "linear",
          lo_thresh: m.lo_thresh !== undefined ? m.lo_thresh : 0,
          hi_thresh: m.hi_thresh !== undefined ? m.hi_thresh : 1,
        };
      }
    }

    return {
      name: preset.name,
      params: params,
      toggles: toggles,
      selects: selects,
      morph: morph,
      mappings: mappings,
    };
  }

  function findPresetByName(name) {
    for (var i = 0; i < PRESETS.length; i++) {
      if (PRESETS[i].name === name) return PRESETS[i];
    }
    return PRESETS[0];
  }

  function abSendEnable() {
    var selA = document.getElementById("ab-select-a");
    var selB = document.getElementById("ab-select-b");
    var presetA = resolvePreset(findPresetByName(selA.value));
    var presetB = resolvePreset(findPresetByName(selB.value));
    abPresetAName = presetA.name;
    abPresetBName = presetB.name;
    send({ type: "ab_enable", preset_a: presetA, preset_b: presetB });
  }

  function abSendSide(side) {
    var selId = side === "a" ? "ab-select-a" : "ab-select-b";
    var sel = document.getElementById(selId);
    var preset = resolvePreset(findPresetByName(sel.value));
    if (side === "a") abPresetAName = preset.name;
    else abPresetBName = preset.name;
    send({ type: "ab_set_preset", side: side, preset: preset });
  }

  function initABCompare() {
    var btnToggle = document.getElementById("btn-ab-toggle");
    var selA = document.getElementById("ab-select-a");
    var selB = document.getElementById("ab-select-b");

    // Populate A/B selects from PRESETS
    PRESETS.forEach(function (preset) {
      var optA = document.createElement("option");
      optA.value = preset.name;
      optA.textContent = preset.name;
      selA.appendChild(optA);

      var optB = document.createElement("option");
      optB.value = preset.name;
      optB.textContent = preset.name;
      selB.appendChild(optB);
    });

    // Default: A = first preset, B = second (or first if only one)
    if (PRESETS.length > 1) selB.value = PRESETS[1].name;

    btnToggle.addEventListener("click", function () {
      abModeActive = !abModeActive;
      btnToggle.classList.toggle("active", abModeActive);
      document.querySelector(".card--preview").classList.toggle("ab-expanded", abModeActive);
      selA.disabled = !abModeActive;
      selB.disabled = !abModeActive;
      if (abModeActive) {
        abSendEnable();
      } else {
        send({ type: "ab_disable" });
        abPresetAName = "";
        abPresetBName = "";
      }
    });

    selA.addEventListener("change", function () {
      if (!abModeActive) return;
      abSendSide("a");
    });
    selB.addEventListener("change", function () {
      if (!abModeActive) return;
      abSendSide("b");
    });
  }

  function applyPreset(preset) {
    // 1. Clear all existing mappings
    for (var key in mappingUI) {
      clearMapping(key);
    }

    // 2. Merge defaults with preset overrides
    var params = {};
    for (var dk in PRESET_DEFAULTS.params) params[dk] = PRESET_DEFAULTS.params[dk];
    if (preset.params) { for (var pk in preset.params) params[pk] = preset.params[pk]; }

    var toggles = {};
    for (var dtk in PRESET_DEFAULTS.toggles) toggles[dtk] = PRESET_DEFAULTS.toggles[dtk];
    if (preset.toggles) { for (var tk in preset.toggles) toggles[tk] = preset.toggles[tk]; }

    var selects = {};
    for (var dsk in PRESET_DEFAULTS.selects) selects[dsk] = PRESET_DEFAULTS.selects[dsk];
    if (preset.selects) { for (var sk in preset.selects) selects[sk] = preset.selects[sk]; }

    var morph = preset.morph || PRESET_DEFAULTS.morph;

    // 3. Set slider params
    for (var pkey in params) {
      var serverVal = params[pkey];
      // Feedback: preset stores server value (1.0=off), UI is inverted (0=off)
      var uiVal = pkey === "feedback_strength" ? 1.0 - serverVal : serverVal;
      var slider = document.querySelector('.slider[data-key="' + pkey + '"]');
      if (slider) {
        slider.value = uiVal;
        var valEl = document.getElementById("val-" + pkey);
        if (valEl) valEl.textContent = formatSliderVal(pkey, uiVal);
      }
      send({ type: "param", key: pkey, value: serverVal });
    }

    // 4. Set toggles
    for (var tkey in toggles) {
      var chk = document.querySelector('.toggle-ctrl[data-key="' + tkey + '"]');
      if (chk) chk.checked = toggles[tkey];
      send({ type: "param", key: tkey, value: toggles[tkey] });
    }

    // 5. Set selects
    for (var skey in selects) {
      var sel = document.querySelector('.select-ctrl[data-key="' + skey + '"]');
      if (sel) sel.value = selects[skey];
      send({ type: "param", key: skey, value: selects[skey] });
    }

    // 6. Set morph mode
    document.querySelectorAll("[data-morph]").forEach(function (b) { b.classList.remove("active"); });
    var morphBtn = document.querySelector('[data-morph="' + morph + '"]');
    if (morphBtn) morphBtn.classList.add("active");
    send({ type: "morph", mode: morph });

    // 7. Set audio mappings via centralized Mappings card
    if (preset.mappings) {
      for (var mkey in preset.mappings) {
        var m = preset.mappings[mkey];
        var ui = mappingUI[mkey];
        if (!ui) continue;

        ui.sourceSelect.value = m.source;
        ui.enableCheckbox.checked = true;
        ui.row.classList.remove("dim");
        ui.loSlider.value = m.min;
        ui.hiSlider.value = m.max;
        if (m.invert) { ui.invertBtn.classList.add("active"); } else { ui.invertBtn.classList.remove("active"); }
        if (m.curve) { ui.curveSelect.value = m.curve; } else { ui.curveSelect.value = "linear"; }
        ui.valsDisplay.textContent = formatSliderVal(mkey, m.min) + " \u2192 " + formatSliderVal(mkey, m.max);

        var loThresh = (m.lo_thresh !== undefined) ? m.lo_thresh : 0;
        var hiThresh = (m.hi_thresh !== undefined) ? m.hi_thresh : 1;
        ui.floorSlider.value = loThresh;
        ui.ceilSlider.value = hiThresh;
        ui.threshDisplay.textContent = Math.round(loThresh * 100) + "% \u2192 " + Math.round(hiThresh * 100) + "%";

        sendCurrentMapping(mkey);
      }
    }

    // Track active preset
    activePresetName = preset.name;
    presetDirty = false;
    send({ type: "preset", name: preset.name });
  }

  function initPresets() {
    var grid = document.getElementById("preset-grid");
    var sel = document.getElementById("preset-select");
    PRESETS.forEach(function (preset, i) {
      var btn = document.createElement("button");
      btn.className = "preset-btn";
      if (i === 0) btn.classList.add("active");
      btn.textContent = preset.name;
      btn.title = preset.tip;
      btn.addEventListener("click", function () {
        applyPreset(preset);
        document.querySelectorAll(".preset-btn").forEach(function (b) { b.classList.remove("active"); b.classList.remove("dirty"); });
        btn.classList.add("active");
        if (sel) sel.value = preset.name;
      });
      grid.appendChild(btn);
      // Also populate mobile select
      if (sel) {
        var opt = document.createElement("option");
        opt.value = preset.name;
        opt.textContent = preset.name;
        sel.appendChild(opt);
      }
    });
    // Mobile select change handler
    if (sel) {
      sel.addEventListener("change", function () {
        var name = sel.value;
        for (var i = 0; i < PRESETS.length; i++) {
          if (PRESETS[i].name === name) {
            applyPreset(PRESETS[i]);
            document.querySelectorAll(".preset-btn").forEach(function (b) { b.classList.remove("active"); b.classList.remove("dirty"); });
            document.querySelectorAll(".preset-btn").forEach(function (b) {
              if (b.textContent === name) b.classList.add("active");
            });
            break;
          }
        }
      });
    }

    // Reset button re-applies the current active preset
    var resetBtn = document.getElementById("preset-reset-btn");
    if (resetBtn) {
      resetBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        for (var i = 0; i < PRESETS.length; i++) {
          if (PRESETS[i].name === activePresetName) {
            applyPreset(PRESETS[i]);
            break;
          }
        }
      });
    }
  }

  // ─── Source Input (Image / Camera) ─────────────────────────────

  var cameraActive = false;
  var cameraStream = null;
  var cameraRafId = null;
  var cameraCanvas = null;
  var cameraCtx = null;
  var cameraLastSend = 0;
  var videoFileActive = false;
  var videoObjectURL = null;

  function initSourceInput() {
    var input = document.getElementById("img2img-input");
    var uploadLabel = document.getElementById("img2img-upload-label");
    var labelText = uploadLabel.querySelector(".file-upload-text");
    var clearBtn = document.getElementById("img2img-clear");
    var preview = document.getElementById("img2img-preview");
    var thumb = document.getElementById("img2img-thumb");
    var cameraBtn = document.getElementById("btn-camera");
    var cameraVideo = document.getElementById("camera-video");
    var cameraDeviceRow = document.getElementById("camera-device-row");
    var cameraDeviceSelect = document.getElementById("camera-device-select");

    // Offscreen canvas for frame capture
    cameraCanvas = document.createElement("canvas");
    cameraCanvas.width = 256;
    cameraCanvas.height = 256;
    cameraCtx = cameraCanvas.getContext("2d");

    uploadLabel.classList.remove("inactive");

    input.addEventListener("change", function () {
      if (input.files.length > 0) {
        var file = input.files[0];
        if (file.type.startsWith("video/") || file.type === "image/gif") {
          startVideoFile(file);
        } else {
          stopCamera();
          uploadImg2Img(file);
        }
      }
    });

    clearBtn.addEventListener("click", function () {
      stopCamera();
      send({ type: "img2img_clear" });
      preview.classList.add("hidden");
      clearBtn.classList.add("hidden");
      labelText.textContent = "Load Source";
      thumb.src = "";
      thumb.style.display = "";
      input.value = "";
    });

    cameraBtn.addEventListener("click", function () {
      if (cameraActive) {
        stopCamera();
        send({ type: "camera_stop" });
        preview.classList.add("hidden");
        clearBtn.classList.add("hidden");
        labelText.textContent = "Load Source";
      } else {
        startCamera();
      }
    });

    cameraDeviceSelect.addEventListener("change", function () {
      if (cameraActive) {
        // Restart with selected device
        stopCameraStream();
        startCameraStream(cameraDeviceSelect.value);
      }
    });

    function startCamera(deviceId) {
      var constraints = { video: { width: { ideal: 256 }, height: { ideal: 256 } }, audio: false };
      if (deviceId) constraints.video.deviceId = { exact: deviceId };

      navigator.mediaDevices.getUserMedia(constraints)
        .then(function (stream) {
          cameraStream = stream;
          cameraActive = true;
          cameraBtn.classList.add("active");

          // Show video preview, hide image thumb
          thumb.style.display = "none";
          cameraVideo.classList.add("active");
          cameraVideo.srcObject = stream;
          preview.classList.remove("hidden");
          preview.classList.remove("skeleton");
          clearBtn.classList.remove("hidden");
          labelText.textContent = "Load Source";

          send({ type: "camera_start" });

          // Enumerate devices for selector
          navigator.mediaDevices.enumerateDevices().then(function (devices) {
            var videoDevices = devices.filter(function (d) { return d.kind === "videoinput"; });
            if (videoDevices.length > 1) {
              cameraDeviceSelect.innerHTML = "";
              videoDevices.forEach(function (d, i) {
                var opt = document.createElement("option");
                opt.value = d.deviceId;
                opt.textContent = d.label || ("Camera " + (i + 1));
                cameraDeviceSelect.appendChild(opt);
              });
              // Select current device
              var activeTrack = stream.getVideoTracks()[0];
              var activeSettings = activeTrack && activeTrack.getSettings();
              if (activeSettings && activeSettings.deviceId) {
                cameraDeviceSelect.value = activeSettings.deviceId;
              }
              cameraDeviceRow.classList.remove("hidden");
            } else {
              cameraDeviceRow.classList.add("hidden");
            }
          });

          // Start frame capture loop
          cameraLastSend = 0;
          cameraRafId = requestAnimationFrame(captureLoop);
        })
        .catch(function (err) {
          console.error("Camera access failed:", err);
          cameraBtn.classList.remove("active");
        });
    }

    function startCameraStream(deviceId) {
      startCamera(deviceId);
    }

    function captureLoop(timestamp) {
      if (!cameraActive) return;
      cameraRafId = requestAnimationFrame(captureLoop);

      // Throttle to ~5fps
      if (timestamp - cameraLastSend < 200) return;
      // Skip if WS not connected or serious backpressure (>50KB queued)
      if (!ws || ws.readyState !== WebSocket.OPEN || ws.bufferedAmount > 50000) return;

      var video = cameraVideo;
      if (video.readyState < video.HAVE_CURRENT_DATA) return;

      // Set timestamp synchronously BEFORE async toBlob to prevent burst sends
      cameraLastSend = timestamp;

      cameraCtx.drawImage(video, 0, 0, 256, 256);
      cameraCanvas.toBlob(function (blob) {
        if (blob && ws && ws.readyState === WebSocket.OPEN && cameraActive) {
          ws.send(blob);
        }
      }, "image/jpeg", 0.6);
    }

    function uploadImg2Img(file) {
      preview.classList.remove("hidden");
      preview.classList.add("skeleton");
      clearBtn.classList.add("hidden");

      // Show image thumb, hide video
      thumb.style.display = "";
      cameraVideo.classList.remove("active");

      var reader = new FileReader();
      reader.onload = function (e) { thumb.src = e.target.result; };
      reader.readAsDataURL(file);

      var formData = new FormData();
      formData.append("file", file);
      fetch("/api/upload_img2img", { method: "POST", body: formData })
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
          preview.classList.remove("skeleton");
          clearBtn.classList.remove("hidden");
          labelText.textContent = "Replace";
        })
        .catch(function (err) {
          console.error("Img2img upload failed:", err);
          preview.classList.remove("skeleton");
          preview.classList.add("hidden");
        });
    }

    function startVideoFile(file) {
      stopCamera();

      // Create object URL for the video file
      videoObjectURL = URL.createObjectURL(file);

      // Set up the existing cameraVideo element for file playback
      cameraVideo.srcObject = null;
      cameraVideo.src = videoObjectURL;
      cameraVideo.loop = true;
      cameraVideo.muted = true;
      cameraVideo.playsInline = true;
      cameraVideo.play();

      // Show video preview
      thumb.style.display = "none";
      cameraVideo.classList.add("active");
      preview.classList.remove("hidden");
      preview.classList.remove("skeleton");
      clearBtn.classList.remove("hidden");
      labelText.textContent = "Replace";

      // Reuse camera pipeline on server
      cameraActive = true;
      videoFileActive = true;
      send({ type: "camera_start" });

      // Start frame capture (same loop as camera)
      cameraLastSend = 0;
      cameraRafId = requestAnimationFrame(captureLoop);
    }
  }

  function stopCameraStream() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(function (t) { t.stop(); });
      cameraStream = null;
    }
    var video = document.getElementById("camera-video");
    if (video) video.srcObject = null;
  }

  function stopCamera() {
    if (cameraRafId) {
      cancelAnimationFrame(cameraRafId);
      cameraRafId = null;
    }
    stopCameraStream();
    // Clean up video file playback
    if (videoObjectURL) {
      URL.revokeObjectURL(videoObjectURL);
      videoObjectURL = null;
    }
    var video = document.getElementById("camera-video");
    if (video) {
      video.classList.remove("active");
      if (videoFileActive) {
        video.pause();
        video.removeAttribute("src");
        video.load();
      }
    }
    videoFileActive = false;
    cameraActive = false;
    var btn = document.getElementById("btn-camera");
    if (btn) btn.classList.remove("active");
    var deviceRow = document.getElementById("camera-device-row");
    if (deviceRow) deviceRow.classList.add("hidden");
  }

  // ─── Output Section ──────────────────────────────────────────────

  var senderUI = {};

  function initOutput() {
    var resSelect = document.getElementById("output-res-select");
    var customRow = document.getElementById("output-custom-row");
    var wInput = document.getElementById("output-w");
    var hInput = document.getElementById("output-h");
    var customApply = document.getElementById("output-custom-apply");
    var fpsSlider = document.getElementById("output-fps-slider");
    var fpsVal = document.getElementById("val-target_fps");

    resSelect.addEventListener("change", function () {
      if (resSelect.value === "custom") {
        customRow.classList.remove("hidden");
      } else {
        customRow.classList.add("hidden");
        var parts = resSelect.value.split("x");
        var w = parseInt(parts[0]), h = parseInt(parts[1]);
        wInput.value = w;
        hInput.value = h;
        send({ type: "output_size", width: w, height: h });
      }
    });

    customApply.addEventListener("click", function () {
      var w = parseInt(wInput.value) || 1920;
      var h = parseInt(hInput.value) || 1080;
      send({ type: "output_size", width: w, height: h });
    });

    // Output aspect ratio buttons
    document.querySelectorAll("[data-oaspect]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll("[data-oaspect]").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        send({ type: "output_aspect", aspect: btn.dataset.oaspect });
        resSelect.value = "custom";
        customRow.classList.remove("hidden");
      });
    });

    fpsSlider.addEventListener("input", function () {
      if (_syncingFromServer) return;
      var v = parseInt(fpsSlider.value);
      fpsVal.textContent = String(v);
      send({ type: "target_fps", fps: v });
    });

    var previewSelect = document.getElementById("preview-fraction-select");
    previewSelect.addEventListener("change", function () {
      if (_syncingFromServer) return;
      send({ type: "preview_fraction", value: parseFloat(previewSelect.value) });
    });

    // Sender start/stop
    ["ndi", "syphon", "spout"].forEach(function (name) {
      var startBtn = document.getElementById("btn-" + name + "-start");
      var stopBtn = document.getElementById("btn-" + name + "-stop");
      var statusEl = document.getElementById(name + "-status");
      senderUI[name] = { startBtn: startBtn, stopBtn: stopBtn, statusEl: statusEl };

      startBtn.addEventListener("click", function () {
        startBtn.disabled = true;
        statusEl.textContent = "Starting...";
        statusEl.className = "sender-status";
        send({ type: "start_sender", sender: name });
      });
      stopBtn.addEventListener("click", function () {
        send({ type: "stop_sender", sender: name });
        stopBtn.classList.add("hidden");
        startBtn.classList.remove("hidden");
        startBtn.disabled = false;
        statusEl.textContent = "";
      });
    });

    // Recording
    var recordDir = document.getElementById("record-dir");
    var recStart = document.getElementById("btn-record-start");
    var recStop = document.getElementById("btn-record-stop");
    recStart.addEventListener("click", function () {
      send({ type: "start_recording", dir: recordDir.value || "output/recording" });
      recStart.classList.add("hidden");
      recStop.classList.remove("hidden");
      recordDir.disabled = true;
    });
    recStop.addEventListener("click", function () {
      send({ type: "stop_recording" });
      recStop.classList.add("hidden");
      recStart.classList.remove("hidden");
      recordDir.disabled = false;
    });
  }

  function handleSenderResult(msg) {
    var ui = senderUI[msg.sender];
    if (!ui) return;
    if (msg.ok) {
      ui.startBtn.classList.add("hidden");
      ui.stopBtn.classList.remove("hidden");
      ui.statusEl.textContent = "Active";
      ui.statusEl.className = "sender-status sender-active";
    } else {
      ui.startBtn.disabled = false;
      ui.statusEl.textContent = "Not available";
      ui.statusEl.className = "sender-status sender-error";
    }
  }

  var elStatusOutput = document.getElementById("status-output");
  var _senderAvailChecked = false;
  function updateOutputStatus(meta) {
    if (!_senderAvailChecked && meta.has_ndi !== undefined) {
      _senderAvailChecked = true;
      var avail = { ndi: meta.has_ndi, syphon: meta.has_syphon, spout: meta.has_spout };
      for (var name in avail) {
        var ui = senderUI[name];
        if (ui && !avail[name]) {
          ui.startBtn.disabled = true;
          ui.startBtn.title = name.charAt(0).toUpperCase() + name.slice(1) + " not installed";
          ui.statusEl.textContent = "Not installed";
          ui.statusEl.className = "sender-status sender-unavail";
        }
      }
    }

    if (meta.output_senders !== undefined) {
      for (var sname in senderUI) {
        var sui = senderUI[sname];
        var isActive = false;
        for (var i = 0; i < meta.output_senders.length; i++) {
          if (meta.output_senders[i].toLowerCase().indexOf(sname) !== -1) { isActive = true; break; }
        }
        if (isActive) {
          sui.startBtn.disabled = false;
          sui.startBtn.classList.add("hidden");
          sui.stopBtn.classList.remove("hidden");
          sui.statusEl.textContent = "Active";
          sui.statusEl.className = "sender-status sender-active";
        } else if (!sui.startBtn.disabled) {
          sui.startBtn.classList.remove("hidden");
          sui.stopBtn.classList.add("hidden");
          if (sui.statusEl.classList.contains("sender-active")) {
            sui.statusEl.textContent = "";
            sui.statusEl.className = "sender-status";
          }
        }
      }
    }

    var recStart = document.getElementById("btn-record-start");
    var recStop = document.getElementById("btn-record-stop");
    var recordDir = document.getElementById("record-dir");
    if (meta.output_recording) {
      recStart.classList.add("hidden");
      recStop.classList.remove("hidden");
      recordDir.disabled = true;
    } else if (!recStop.classList.contains("hidden")) {
      recStop.classList.add("hidden");
      recStart.classList.remove("hidden");
      recordDir.disabled = false;
    }

    if (meta.output_active) {
      var outFps = meta.output_fps != null ? meta.output_fps : meta.fps_smooth;
      var parts = [meta.output_size[0] + "x" + meta.output_size[1] + " @ " + Math.round(outFps) + "fps"];
      if (meta.output_senders && meta.output_senders.length > 0) {
        parts.push(meta.output_senders.join(", "));
      }
      if (meta.output_recording) {
        parts.push("Rec: " + meta.output_rec_frames);
      }
      elStatusOutput.textContent = "Out: " + parts.join(" | ");
      elStatusOutput.classList.remove("hidden");
    } else {
      elStatusOutput.classList.add("hidden");
    }
  }

  // ─── MIDI ──────────────────────────────────────────────────────

  var MIDI_PARAMS = [
    { key: "fg_brightness", label: "Brightness" },
    { key: "render_scale", label: "Scale" },
    { key: "sharpen", label: "Sharpen" },
    { key: "saturation", label: "Saturation" },
    { key: "contrast", label: "Contrast" },
    { key: "gamma", label: "Gamma" },
    { key: "posterize", label: "Posterize" },
    { key: "grain", label: "Grain" },
    { key: "scanlines", label: "Scanlines" },
    { key: "opacity", label: "Opacity" },
    { key: "feedback_strength", label: "Feedback" },
    { key: "persistence", label: "Persistence" },
    { key: "steps", label: "ODE Steps" },
    { key: "onset_threshold", label: "Onset Thr" },
  ];

  var MIDI_TRIGGERS = [
    { key: "regenerate_pool", label: "Regenerate Pool" },
    { key: "next_grid", label: "Next Grid" },
    { key: "morph_frozen", label: "Morph: Frozen" },
    { key: "morph_ambient", label: "Morph: Ambient" },
    { key: "morph_beat", label: "Morph: Beat" },
    { key: "gen_pool", label: "Gen: Pool" },
    { key: "gen_realtime", label: "Gen: Realtime" },
    { key: "toggle_superres", label: "Toggle SuperRes" },
  ];

  var midiParamUI = {};
  var midiTriggerUI = {};
  var midiLearningKey = null;
  var midiLearningBtn = null;

  function initMIDI() {
    var paramList = document.getElementById("midi-param-list");
    var triggerList = document.getElementById("midi-trigger-list");

    var ph = document.createElement("h3");
    ph.textContent = "Parameters";
    paramList.appendChild(ph);

    MIDI_PARAMS.forEach(function (p) {
      var row = createMidiMapItem(p.key, p.label, false);
      midiParamUI[p.key] = row;
      paramList.appendChild(row.el);
    });

    var th = document.createElement("h3");
    th.textContent = "Triggers";
    triggerList.appendChild(th);

    MIDI_TRIGGERS.forEach(function (t) {
      var row = createMidiMapItem(t.key, t.label, true);
      midiTriggerUI[t.key] = row;
      triggerList.appendChild(row.el);
    });

    var portSelect = document.getElementById("midi-port-select");
    portSelect.addEventListener("change", function () {
      send({ type: "midi_select_port", port: portSelect.value || null });
    });

    document.getElementById("midi-refresh-btn").addEventListener("click", function () {
      send({ type: "midi_list_ports" });
    });

    setTimeout(function () { send({ type: "midi_list_ports" }); }, 500);
  }

  function createMidiMapItem(key, label, isTrigger) {
    var el = document.createElement("div");
    el.className = "midi-map-item";

    var labelEl = document.createElement("span");
    labelEl.className = "midi-map-label";
    labelEl.textContent = label;

    var ccDisplay = document.createElement("span");
    ccDisplay.className = "midi-cc-display";
    ccDisplay.textContent = "--";

    var learnBtn = document.createElement("button");
    learnBtn.className = "midi-learn-btn";
    learnBtn.textContent = "Learn";
    learnBtn.addEventListener("click", function () {
      if (learnBtn.classList.contains("learning")) {
        learnBtn.classList.remove("learning");
        learnBtn.textContent = "Learn";
        midiLearningKey = null;
        midiLearningBtn = null;
        send({ type: "midi_cancel_learn" });
      } else {
        if (midiLearningBtn) {
          midiLearningBtn.classList.remove("learning");
          midiLearningBtn.textContent = "Learn";
        }
        learnBtn.classList.add("learning");
        learnBtn.textContent = "...";
        midiLearningKey = key;
        midiLearningBtn = learnBtn;
        send({ type: isTrigger ? "midi_learn_trigger" : "midi_learn", key: key });
      }
    });

    var clearBtn = document.createElement("button");
    clearBtn.className = "midi-clear-btn";
    clearBtn.textContent = "\u00d7";
    clearBtn.title = "Clear MIDI mapping";
    clearBtn.style.visibility = "hidden";
    clearBtn.addEventListener("click", function () {
      send({ type: "midi_clear", key: key });
      ccDisplay.textContent = "--";
      clearBtn.style.visibility = "hidden";
    });

    var valDisplay = document.createElement("span");
    valDisplay.className = "midi-val-display";
    valDisplay.textContent = "";

    el.appendChild(labelEl);
    el.appendChild(ccDisplay);
    el.appendChild(valDisplay);
    el.appendChild(learnBtn);
    el.appendChild(clearBtn);

    return { el: el, ccDisplay: ccDisplay, valDisplay: valDisplay, learnBtn: learnBtn, clearBtn: clearBtn };
  }

  function updateMidiStatus(meta) {
    var statusText = document.getElementById("midi-status-text");
    if (meta.midi_active) {
      statusText.textContent = "Connected";
      statusText.className = "midi-status-text connected";
    } else {
      statusText.textContent = "Disconnected";
      statusText.className = "midi-status-text";
    }

    var lastCcEl = document.getElementById("midi-last-cc");
    if (meta.midi_last_cc) {
      var mtype = meta.midi_last_cc[0] === "note" ? "N" : "CC";
      lastCcEl.textContent = mtype + meta.midi_last_cc[1] + " ch" + (meta.midi_last_cc[2] + 1) + "=" + meta.midi_last_cc[3];
    }

    if (meta.midi_last_cc) {
      var dot = document.getElementById("midi-activity-dot");
      dot.classList.add("active");
      clearTimeout(dot._timer);
      dot._timer = setTimeout(function () { dot.classList.remove("active"); }, 100);
    }

    if (meta.midi_mappings) {
      for (var key in midiParamUI) {
        var ui = midiParamUI[key];
        var m = meta.midi_mappings[key];
        if (m) {
          var prefix = m.type === "note" ? "N" : "CC";
          ui.ccDisplay.textContent = prefix + m.cc;
          ui.clearBtn.style.visibility = "";
          if (meta.midi_vals && meta.midi_vals[key] !== undefined) {
            ui.valDisplay.textContent = formatMappedVal(key, meta.midi_vals[key]);
          }
        } else {
          ui.ccDisplay.textContent = "--";
          ui.valDisplay.textContent = "";
          ui.clearBtn.style.visibility = "hidden";
        }
      }
    }

    if (meta.midi_triggers) {
      for (var key in midiTriggerUI) {
        var ui = midiTriggerUI[key];
        var m = meta.midi_triggers[key];
        if (m) {
          var prefix = m.type === "note" ? "N" : "CC";
          ui.ccDisplay.textContent = prefix + m.cc;
          ui.clearBtn.style.visibility = "";
        } else {
          ui.ccDisplay.textContent = "--";
          ui.clearBtn.style.visibility = "hidden";
        }
      }
    }

    if (meta.midi_triggers_fired) {
      for (var i = 0; i < meta.midi_triggers_fired.length; i++) {
        var firedKey = meta.midi_triggers_fired[i];
        var tui = midiTriggerUI[firedKey];
        if (tui) {
          tui.el.classList.add("midi-fired");
          clearTimeout(tui._fireTimer);
          tui._fireTimer = setTimeout((function (el) {
            return function () { el.classList.remove("midi-fired"); };
          })(tui.el), 200);
        }
      }
    }

    if (meta.midi_vals) {
      for (var key in meta.midi_vals) {
        if (!valEls[key]) valEls[key] = document.getElementById("val-" + key);
        var valEl = valEls[key];
        if (valEl) {
          valEl.textContent = formatMappedVal(key, meta.midi_vals[key]);
          valEl.classList.add("mapped-val");
        }
        var slider = document.querySelector('.slider[data-key="' + key + '"]');
        if (slider) {
          slider.value = meta.midi_vals[key];
          slider.classList.add("midi-active");
        }
        if (key === "superres" && elSrToggle) {
          elSrToggle.checked = !!meta.midi_vals[key];
        }
      }
    }
    // Clean up stale mapped-val from params no longer MIDI-driven
    for (var key in midiParamUI) {
      if (meta.midi_vals && meta.midi_vals[key] !== undefined) continue;
      if (meta.mapped && meta.mapped[key] !== undefined) continue;
      var valEl = valEls[key] || (valEls[key] = document.getElementById("val-" + key));
      var slider = document.querySelector('.slider[data-key="' + key + '"]');
      if (slider) slider.classList.remove("midi-active");
      if (valEl && valEl.classList.contains("mapped-val")) {
        valEl.classList.remove("mapped-val");
        if (slider) valEl.textContent = formatSliderVal(key, parseFloat(slider.value));
      }
    }

    if (meta.midi_learn === null && midiLearningBtn) {
      midiLearningBtn.classList.remove("learning");
      midiLearningBtn.textContent = "Learn";
      midiLearningKey = null;
      midiLearningBtn = null;
    }
  }

  function handleMidiPorts(msg) {
    var select = document.getElementById("midi-port-select");
    var current = msg.current || "";
    select.innerHTML = '<option value="">-- No MIDI --</option>';
    if (msg.error) {
      var errOpt = document.createElement("option");
      errOpt.disabled = true;
      errOpt.textContent = msg.error;
      select.appendChild(errOpt);
    }
    (msg.ports || []).forEach(function (port) {
      var opt = document.createElement("option");
      opt.value = port;
      opt.textContent = port;
      if (port === current) opt.selected = true;
      select.appendChild(opt);
    });
  }

  // ─── Multi-Client State Sync ───────────────────────────────────

  // Aspect ratio lookup: preview_size → aspect string
  var ASPECT_SIZES = {
    "1:1": function (w, h) { return w === h; },
    "4:3": function (w, h) { return Math.abs(w / h - 4 / 3) < 0.02; },
    "16:9": function (w, h) { return Math.abs(w / h - 16 / 9) < 0.02; },
    "9:16": function (w, h) { return Math.abs(w / h - 9 / 16) < 0.02; },
    "16:10": function (w, h) { return Math.abs(w / h - 16 / 10) < 0.02; },
    "21:9": function (w, h) { return Math.abs(w / h - 21 / 9) < 0.02; },
  };

  function applyInitState(msg) {
    _syncingFromServer = true;
    try {
      // Dismiss loading overlay immediately if pool is ready
      if (msg.pool_ready && !firstFrameReceived) {
        firstFrameReceived = true;
        if (elLoadingOverlay) elLoadingOverlay.classList.add("hidden");
      }
      if (msg.params) applyParamSync(msg.params);
      if (msg.morph_mode) {
        document.querySelectorAll("[data-morph]").forEach(function (b) {
          b.classList.toggle("active", b.dataset.morph === msg.morph_mode);
        });
        _lastMorph = msg.morph_mode;
      }
      if (msg.gen_mode) {
        document.querySelectorAll("[data-gen]").forEach(function (b) {
          b.classList.toggle("active", b.dataset.gen === msg.gen_mode);
        });
        var poolActions = document.getElementById("pool-actions");
        if (poolActions) poolActions.classList.toggle("disabled", msg.gen_mode === "realtime");
        _lastGenMode = msg.gen_mode;
      }
      if (msg.audio_mode) {
        document.querySelectorAll(".mode-btn").forEach(function (b) {
          b.classList.toggle("active", b.dataset.mode === msg.audio_mode);
        });
        updateFileUploadVisibility();
        if (msg.audio_mode === "file" && msg.audio_filename) {
          var ac = document.getElementById("audio-container");
          ac.classList.remove("hidden");
          ac.classList.remove("skeleton");
          document.getElementById("audio-filename").textContent = msg.audio_filename;
          var ap = document.getElementById("audio-player");
          ap.src = "/audio/" + encodeURIComponent(msg.audio_filename);
          document.getElementById("audio-clear").classList.remove("hidden");
        }
      }
      if (msg.mappings) applySyncMappings(msg.mappings);
      if (msg.target_fps !== undefined) {
        var fpsSlider = document.getElementById("output-fps-slider");
        var fpsVal = document.getElementById("val-target_fps");
        if (fpsSlider) fpsSlider.value = msg.target_fps;
        if (fpsVal) fpsVal.textContent = String(Math.round(msg.target_fps));
      }
      if (msg.preview_fraction !== undefined) {
        var previewSelect = document.getElementById("preview-fraction-select");
        if (previewSelect) previewSelect.value = String(msg.preview_fraction);
      }
      if (msg.preview_size) {
        var pw = msg.preview_size[0], ph = msg.preview_size[1];
        document.querySelectorAll("[data-aspect]").forEach(function (b) {
          var fn = ASPECT_SIZES[b.dataset.aspect];
          b.classList.toggle("active", fn ? fn(pw, ph) : false);
        });
      }
      if (msg.active_preset) {
        activePresetName = msg.active_preset;
        presetDirty = false;
        document.querySelectorAll(".preset-btn").forEach(function (b) {
          b.classList.toggle("active", b.textContent === msg.active_preset);
          b.classList.remove("dirty");
        });
        var psel = document.getElementById("preset-select");
        if (psel) psel.value = msg.active_preset;
      }
      syncABState(msg);
    } finally {
      _syncingFromServer = false;
    }
  }

  function applyStateUpdate(msg) {
    _syncingFromServer = true;
    try {
      if (msg.params) applyParamSync(msg.params);
      if (msg.morph_mode) {
        document.querySelectorAll("[data-morph]").forEach(function (b) {
          b.classList.toggle("active", b.dataset.morph === msg.morph_mode);
        });
        _lastMorph = msg.morph_mode;
      }
      if (msg.gen_mode) {
        document.querySelectorAll("[data-gen]").forEach(function (b) {
          b.classList.toggle("active", b.dataset.gen === msg.gen_mode);
        });
        var poolActions = document.getElementById("pool-actions");
        if (poolActions) poolActions.classList.toggle("disabled", msg.gen_mode === "realtime");
        _lastGenMode = msg.gen_mode;
      }
      if (msg.audio_mode) {
        document.querySelectorAll(".mode-btn").forEach(function (b) {
          b.classList.toggle("active", b.dataset.mode === msg.audio_mode);
        });
        updateFileUploadVisibility();
      }
      if (msg.mappings) applySyncMappings(msg.mappings);
      if (msg.target_fps !== undefined) {
        var fpsSlider = document.getElementById("output-fps-slider");
        var fpsVal = document.getElementById("val-target_fps");
        if (fpsSlider) fpsSlider.value = msg.target_fps;
        if (fpsVal) fpsVal.textContent = String(Math.round(msg.target_fps));
      }
      if (msg.preview_fraction !== undefined) {
        var previewSelect = document.getElementById("preview-fraction-select");
        if (previewSelect) previewSelect.value = String(msg.preview_fraction);
      }
      if (msg.preview_size) {
        var pw = msg.preview_size[0], ph = msg.preview_size[1];
        document.querySelectorAll("[data-aspect]").forEach(function (b) {
          var fn = ASPECT_SIZES[b.dataset.aspect];
          b.classList.toggle("active", fn ? fn(pw, ph) : false);
        });
      }
      if (msg.active_preset) {
        activePresetName = msg.active_preset;
        presetDirty = false;
        document.querySelectorAll(".preset-btn").forEach(function (b) {
          b.classList.toggle("active", b.textContent === msg.active_preset);
          b.classList.remove("dirty");
        });
        var psel = document.getElementById("preset-select");
        if (psel) psel.value = msg.active_preset;
      }
      syncABState(msg);
    } finally {
      _syncingFromServer = false;
    }
  }

  function syncABState(msg) {
    if (msg.ab_enabled !== undefined) {
      abModeActive = !!msg.ab_enabled;
      var btnToggle = document.getElementById("btn-ab-toggle");
      var selA = document.getElementById("ab-select-a");
      var selB = document.getElementById("ab-select-b");
      if (btnToggle) btnToggle.classList.toggle("active", abModeActive);
      var previewCard = document.querySelector(".card--preview");
      if (previewCard) previewCard.classList.toggle("ab-expanded", abModeActive);
      if (selA) selA.disabled = !abModeActive;
      if (selB) selB.disabled = !abModeActive;
    }
    if (msg.ab_preset_a) {
      abPresetAName = msg.ab_preset_a;
      var selA2 = document.getElementById("ab-select-a");
      if (selA2) selA2.value = msg.ab_preset_a;
    }
    if (msg.ab_preset_b) {
      abPresetBName = msg.ab_preset_b;
      var selB2 = document.getElementById("ab-select-b");
      if (selB2) selB2.value = msg.ab_preset_b;
    }
  }

  function applyParamSync(params) {
    for (var key in params) {
      var val = params[key];
      // Sliders
      var slider = document.querySelector('.slider[data-key="' + key + '"]');
      if (slider) {
        if (_draggingKeys.has(key)) continue;
        // feedback_strength: server value is inverted from UI
        var uiVal = key === "feedback_strength" ? 1.0 - val : val;
        slider.value = uiVal;
        var valEl = document.getElementById("val-" + key);
        if (valEl && !valEl.classList.contains("mapped-val")) {
          valEl.textContent = formatSliderVal(key, typeof uiVal === "number" ? uiVal : parseFloat(uiVal));
        }
        continue;
      }
      // Toggles
      var chk = document.querySelector('.toggle-ctrl[data-key="' + key + '"]');
      if (chk) {
        chk.checked = !!val;
        if (key === "superres") _lastSuperres = !!val;
        continue;
      }
      // Selects
      var sel = document.querySelector('.select-ctrl[data-key="' + key + '"]');
      if (sel) {
        sel.value = val;
        continue;
      }
    }
  }

  function applySyncMappings(mappings) {
    for (var key in mappings) {
      var ui = mappingUI[key];
      if (!ui) continue;
      var m = mappings[key];
      if (m === null || m === undefined) {
        // Clear this mapping without sending to server
        ui.sourceSelect.value = "none";
        ui.enableCheckbox.checked = false;
        ui.row.classList.add("dim");
        ui.sourceLabel.textContent = "--";
        ui.outputFill.style.width = "0%";
        ui.outputVal.textContent = "";
        ui.invertBtn.classList.remove("active");
        ui.curveSelect.value = "linear";
        ui.floorSlider.value = "0";
        ui.ceilSlider.value = "1";
        ui.threshDisplay.textContent = "0% \u2192 100%";
        ui.detail.classList.remove("open");
        ui.expandBtn.classList.remove("open");
        var valEl = document.getElementById("val-" + key);
        if (valEl && ui.slider) {
          valEl.classList.remove("mapped-val");
          valEl.textContent = formatSliderVal(key, parseFloat(ui.slider.value));
        }
        if (ui.slider) {
          var ctrlRow = ui.slider.closest(".ctrl-row");
          if (ctrlRow) ctrlRow.classList.remove("mapped");
        }
      } else {
        // Apply mapping state
        ui.sourceSelect.value = m.source || "none";
        if (m.source && m.source !== "none") {
          ui.enableCheckbox.checked = true;
          ui.row.classList.remove("dim");
          ui.loSlider.value = m.min !== undefined ? m.min : ui.loSlider.min;
          ui.hiSlider.value = m.max !== undefined ? m.max : ui.hiSlider.max;
          if (m.invert) { ui.invertBtn.classList.add("active"); } else { ui.invertBtn.classList.remove("active"); }
          ui.curveSelect.value = m.curve || "linear";
          ui.valsDisplay.textContent = formatSliderVal(key, parseFloat(ui.loSlider.value)) + " \u2192 " + formatSliderVal(key, parseFloat(ui.hiSlider.value));
          var loThresh = m.lo_thresh !== undefined ? m.lo_thresh : 0;
          var hiThresh = m.hi_thresh !== undefined ? m.hi_thresh : 1;
          ui.floorSlider.value = loThresh;
          ui.ceilSlider.value = hiThresh;
          ui.threshDisplay.textContent = Math.round(loThresh * 100) + "% \u2192 " + Math.round(hiThresh * 100) + "%";
          if (ui.slider) {
            var ctrlRow = ui.slider.closest(".ctrl-row");
            if (ctrlRow) ctrlRow.classList.add("mapped");
          }
        } else {
          ui.enableCheckbox.checked = false;
          ui.row.classList.add("dim");
        }
      }
    }
  }

  // ─── Helpers ────────────────────────────────────────────────────

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }

  // ─── Feature Analysis ──────────────────────────────────────────

  function formatDuration(secs) {
    secs = Math.floor(secs);
    if (secs < 60) return secs + "s";
    var m = Math.floor(secs / 60);
    var s = secs % 60;
    return m + "m " + s + "s";
  }

  function analysisStart() {
    analysisState = "recording";
    analysisSamples = [];
    analysisFrameCount = 0;
    analysisDuration = parseInt(elAnalysisDurationSlider.value) || 60;
    analysisAutoStop = document.getElementById("analysis-autostop").checked;
    analysisAutoStopTimeout = parseInt(document.getElementById("analysis-autostop-timeout").value) || 5;
    analysisSilenceStart = null;

    elBtnRecord.disabled = true;
    elBtnStop.disabled = false;
    elBtnClear.disabled = true;
    elAnalysisDurationSlider.disabled = true;

    elProgressFill.style.width = "0%";
    elProgressText.textContent = "0 samples";
    elProgressWrap.classList.remove("hidden");

    elLog.innerHTML = "";
    elLogContainer.classList.remove("hidden");

    elResults.classList.add("hidden");
    elBadgeAnalysis.textContent = "Recording 0s";
  }

  function analysisStop() {
    if (analysisState !== "recording") return;
    analysisFinish();
  }

  function analysisFinish() {
    analysisState = "done";
    elBtnRecord.disabled = false;
    elBtnStop.disabled = true;
    elBtnClear.disabled = false;
    elAnalysisDurationSlider.disabled = false;

    if (analysisSamples.length < 2) {
      elHeadlineValue.textContent = "--";
      elHeadlineSamples.textContent = "Not enough data (need at least 2 samples)";
      elResults.classList.remove("hidden");
      elBadgeAnalysis.textContent = "Done (0)";
      return;
    }

    var result = computeCorrelationMatrix(analysisSamples);
    _lastAnalysisResult = result;
    renderAnalysisResults(result);
    elBadgeAnalysis.textContent = "Done (" + analysisSamples.length + ")";
  }

  function analysisClear() {
    analysisState = "idle";
    analysisSamples = [];
    _lastAnalysisResult = null;
    analysisFrameCount = 0;

    elBtnRecord.disabled = false;
    elBtnStop.disabled = true;
    elBtnClear.disabled = true;
    elAnalysisDurationSlider.disabled = false;

    elProgressWrap.classList.add("hidden");
    elLogContainer.classList.add("hidden");
    elResults.classList.add("hidden");
    elBadgeAnalysis.textContent = "Idle";
  }

  function computeCorrelationMatrix(samples) {
    var n = samples.length;
    var k = 12;

    // Compute means
    var means = new Float64Array(k);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < k; j++) means[j] += samples[i][j];
    }
    for (var j = 0; j < k; j++) means[j] /= n;

    // Compute covariances and std devs
    var cov = [];
    for (var a = 0; a < k; a++) {
      cov[a] = new Float64Array(k);
    }
    var std = new Float64Array(k);

    for (var i = 0; i < n; i++) {
      for (var a = 0; a < k; a++) {
        var da = samples[i][a] - means[a];
        for (var b = a; b < k; b++) {
          cov[a][b] += da * (samples[i][b] - means[b]);
        }
      }
    }

    for (var a = 0; a < k; a++) {
      for (var b = a; b < k; b++) {
        cov[a][b] /= n;
        if (b !== a) cov[b][a] = cov[a][b];
      }
      std[a] = Math.sqrt(cov[a][a]);
    }

    // Pearson r
    var r = [];
    for (var a = 0; a < k; a++) {
      r[a] = new Float64Array(k);
      for (var b = 0; b < k; b++) {
        if (a === b) { r[a][b] = 1.0; continue; }
        var denom = std[a] * std[b];
        if (denom < 1e-12) { r[a][b] = 0; continue; }
        r[a][b] = Math.max(-1, Math.min(1, cov[a][b] / denom));
      }
    }

    // Mean off-diagonal |r|
    var sumAbs = 0;
    var count = 0;
    for (var a = 0; a < k; a++) {
      for (var b = 0; b < k; b++) {
        if (a !== b) { sumAbs += Math.abs(r[a][b]); count++; }
      }
    }
    var meanAbsR = count > 0 ? sumAbs / count : 0;

    return { r: r, meanAbsR: meanAbsR, n: n };
  }

  function renderAnalysisResults(result) {
    // Headline
    elHeadlineValue.textContent = result.meanAbsR.toFixed(3);
    elHeadlineSamples.textContent = result.n + " samples";

    // Matrix table
    var k = 12;
    var html = "<thead><tr><th></th>";
    for (var i = 0; i < k; i++) html += "<th>" + ANALYSIS_FEATURE_SHORT[i] + "</th>";
    html += "</tr></thead><tbody>";

    for (var a = 0; a < k; a++) {
      html += "<tr><th>" + ANALYSIS_FEATURE_SHORT[a] + "</th>";
      for (var b = 0; b < k; b++) {
        var val = result.r[a][b];
        if (a === b) {
          html += '<td class="diag">1.00</td>';
        } else {
          var absVal = Math.abs(val);
          var alpha = Math.pow(absVal, 0.7) * 0.7;
          var rgbVar = val >= 0 ? "--accent-rgb" : "--beat-rgb";
          var rgb = getComputedStyle(document.documentElement).getPropertyValue(rgbVar).trim();
          var bg = "rgba(" + rgb + "," + alpha.toFixed(3) + ")";
          html += '<td style="background:' + bg + '">' + val.toFixed(2) + "</td>";
        }
      }
      html += "</tr>";
    }
    html += "</tbody>";
    elMatrix.innerHTML = html;

    elResults.classList.remove("hidden");
  }

  function initAnalysis() {
    elAnalysisDurationSlider.addEventListener("input", function () {
      elAnalysisDurationVal.textContent = formatDuration(parseInt(elAnalysisDurationSlider.value));
    });

    elBtnRecord.addEventListener("click", function () { analysisStart(); });
    elBtnStop.addEventListener("click", function () { analysisStop(); });
    elBtnClear.addEventListener("click", function () { analysisClear(); });
  }

  // ─── Card Layout: Collapse + Drag Reorder + Persistence ────────

  var _cardDragging = false;
  var STORAGE_ORDER = "easey-glyph-card-order";
  var STORAGE_COLLAPSED = "easey-glyph-card-collapsed";

  function saveCollapseState() {
    var state = {};
    document.querySelectorAll(".card[data-card]").forEach(function (card) {
      state[card.dataset.card] = card.classList.contains("collapsed");
    });
    try { localStorage.setItem(STORAGE_COLLAPSED, JSON.stringify(state)); } catch (e) {}
  }

  function saveCardOrder() {
    var order = [];
    document.querySelectorAll(".card-grid .card[data-card]").forEach(function (card) {
      order.push(card.dataset.card);
    });
    try { localStorage.setItem(STORAGE_ORDER, JSON.stringify(order)); } catch (e) {}
  }

  function restoreLayout() {
    // Restore card order
    try {
      var orderJSON = localStorage.getItem(STORAGE_ORDER);
      if (orderJSON) {
        var order = JSON.parse(orderJSON);
        var grid = document.querySelector(".card-grid");
        if (grid && Array.isArray(order)) {
          // Build a map of data-card → element
          var cardMap = {};
          grid.querySelectorAll(".card[data-card]").forEach(function (card) {
            cardMap[card.dataset.card] = card;
          });
          // Re-append in saved order (preview always first)
          var preview = cardMap["preview"];
          if (preview) grid.appendChild(preview);
          order.forEach(function (name) {
            if (name !== "preview" && cardMap[name]) {
              grid.appendChild(cardMap[name]);
            }
          });
        }
      }
    } catch (e) {}

    // Restore collapse state
    try {
      var collapsedJSON = localStorage.getItem(STORAGE_COLLAPSED);
      if (collapsedJSON) {
        var state = JSON.parse(collapsedJSON);
        Object.keys(state).forEach(function (name) {
          var card = document.querySelector('[data-card="' + name + '"]');
          if (card) {
            if (state[name]) {
              card.classList.add("collapsed");
            } else {
              card.classList.remove("collapsed");
            }
          }
        });
        return true; // had saved state
      }
    } catch (e) {}
    return false; // no saved state
  }

  // Collapse toggle on card headers
  document.querySelectorAll(".card-header").forEach(function (header) {
    header.addEventListener("click", function (e) {
      // Don't collapse when clicking interactive elements inside header
      var tag = e.target.tagName;
      if (tag === "BUTTON" || tag === "INPUT" || tag === "SELECT" || tag === "LABEL") return;
      if (e.target.closest(".card-header-actions")) return;
      if (e.target.closest(".mode-switch")) return;
      // Don't collapse during drag
      if (_cardDragging) return;

      var card = header.closest(".card");
      if (card) {
        var body = card.querySelector('.card-body');
        if (body) body.style.overflow = '';  // reset to CSS hidden for animation
        card.classList.toggle("collapsed");
        if (!card.classList.contains("collapsed") && body) {
          body.addEventListener('transitionend', function handler(ev) {
            if (ev.propertyName === 'max-height') {
              body.style.overflow = 'visible';
              body.removeEventListener('transitionend', handler);
            }
          });
        }
        saveCollapseState();
      }
    });
  });

  // SortableJS initialization
  function initSortable() {
    var grid = document.querySelector(".card-grid");
    if (!grid || typeof Sortable === "undefined") return;
    Sortable.create(grid, {
      handle: ".card-header",
      filter: ".card--preview",
      preventOnFilter: false,
      animation: 150,
      delay: 100,
      delayOnTouchOnly: true,
      ghostClass: "sortable-ghost",
      chosenClass: "sortable-chosen",
      dragClass: "sortable-drag",
      onStart: function () { _cardDragging = true; },
      onEnd: function () {
        // Delay clearing the flag so the click event from mouseup doesn't toggle collapse
        setTimeout(function () { _cardDragging = false; }, 50);
        saveCardOrder();
      },
    });
  }

  // ─── Mobile Default Collapse ────────────────────────────────────

  function applyMobileDefaults() {
    var isMobile = window.innerWidth < 768 || (window.matchMedia && window.matchMedia("(pointer: coarse)").matches && window.innerWidth < 1024);
    document.body.classList.toggle("is-mobile", isMobile);
    if (isMobile) {
      // Collapse Audio, Effects, Mappings by default on mobile
      ["audio", "effects", "mappings"].forEach(function (name) {
        var card = document.querySelector('[data-card="' + name + '"]');
        if (card && !card.classList.contains("collapsed")) {
          card.classList.add("collapsed");
        }
      });
    }
  }

  // Re-evaluate mobile class on resize/orientation change
  window.addEventListener("resize", function () {
    var isMobile = window.innerWidth < 768 || (window.matchMedia && window.matchMedia("(pointer: coarse)").matches && window.innerWidth < 1024);
    document.body.classList.toggle("is-mobile", isMobile);
  });

  // Preview toggle (minimize/expand preview on mobile)
  (function () {
    var btn = document.getElementById("preview-toggle");
    var card = document.querySelector(".card--preview");
    if (btn && card) {
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        card.classList.toggle("preview-expanded");
        btn.textContent = card.classList.contains("preview-expanded") ? "\u25B2" : "\u25BC";
      });
    }
  })();

  // ─── Init ───────────────────────────────────────────────────────

  initSettings();
  initMappingsCard();
  initSourceInput();
  initOutput();
  initMIDI();
  initAnalysis();
  initHelp();
  initPresets();
  initABCompare();
  var hadSavedLayout = restoreLayout();
  if (!hadSavedLayout) applyMobileDefaults();
  else {
    // Still need to set the is-mobile class
    var isMobile = window.innerWidth < 768 || (window.matchMedia && window.matchMedia("(pointer: coarse)").matches && window.innerWidth < 1024);
    document.body.classList.toggle("is-mobile", isMobile);
  }
  // Allow tooltips to escape expanded card bodies (overflow:hidden is needed during animation)
  document.querySelectorAll('.card:not(.collapsed) .card-body').forEach(function(body) {
    body.style.overflow = 'visible';
  });
  initSortable();
  connect();
})();

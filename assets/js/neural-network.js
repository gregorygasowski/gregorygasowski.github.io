/*
  Live neural network training animation.
  Designed and created by Gregory Gasowski (https://gregorygasowski.github.io).
  Architecture: 2 inputs -> 10 hidden layers x 5 ReLU neurons -> 1 linear output.
  Each cycle trains on real data and animates the four steps:
    1. Forward propagation   z = sum(a_i * w_i) + b,  a = ReLU(z) = max(0, z)
    2. Loss calculation      L = 1/2 (y_hat - y)^2
    3. Backpropagation       delta = dL/dz (chain rule),  dL/dw = delta * a_prev
    4. Optimization (SGD)    w <- w - eta * dL/dw
  Hover (or tap) a neuron to see its actual computation with live numbers.

  Required page elements: <canvas id="nn-canvas">, <div id="nn-tip">, and the
  panel elements with ids nn-step-*, nn-v-*, nn-loss-chart (see neural-network.html).
  Optional: data-zone="CSS selector" on the canvas limits interaction to those page areas
  (used on the website so the network only reacts over the banner and contact sections).
*/
(function () {
	'use strict';

	// ---- Configuration -------------------------------------------------------
	var CONFIG = {
		inputs: 2,
		hiddenLayers: 10,
		neuronsPerHidden: 5,
		outputs: 1,
		learningRate: 0.05,
		batchSize: 32,           // samples per mini-batch (the animated sample is part of the first one)
		updatesPerCycle: 10,     // SGD updates applied during each animated cycle
		stepMs: 260,             // signal travel time per layer
		lossMs: 1100,
		updateMs: 900,
		pauseMs: 350,
		target: function (x1, x2) { return 0.8 * x1 - 0.5 * x2 + 0.3 * x1 * x2 + 0.2; },
		colors: {
			positive: '56,220,205',  // teal: positive weight / activation
			negative: '90,170,255',  // blue: negative weight
			neutral: '200,220,240',
			gradient: '255,170,70',  // amber: backpropagated error
			update: '255,255,255'
		}
	};

	var canvas = document.getElementById('nn-canvas');
	if (!canvas || !canvas.getContext) return;
	var ctx = canvas.getContext('2d');
	var tip = document.getElementById('nn-tip');
	var ZONE = canvas.getAttribute('data-zone'); // e.g. "#banner, #cta" on the website
	function inZone(x, y) {
		if (!ZONE) return true;
		var el = document.elementFromPoint(x, y);
		if (!el || (el.closest && el.closest('a, button, input, #header, #menu, #nn-panel'))) return false;
		return !!(el.closest && el.closest(ZONE));
	}
	var zoneVisible = true, pausedAt = 0;
	if (ZONE && 'IntersectionObserver' in window) {
		var seen = {};
		var io = new IntersectionObserver(function (entries) {
			entries.forEach(function (e) { seen[e.target.id || 'z'] = e.isIntersecting; });
			zoneVisible = Object.keys(seen).some(function (k) { return seen[k]; });
		});
		Array.prototype.forEach.call(document.querySelectorAll(ZONE), function (el) { io.observe(el); });
	}
	var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
	var dpr = Math.min(window.devicePixelRatio || 1, 2);

	// ---- Model ---------------------------------------------------------------
	var sizes = [CONFIG.inputs];
	for (var h = 0; h < CONFIG.hiddenLayers; h++) sizes.push(CONFIG.neuronsPerHidden);
	sizes.push(CONFIG.outputs);
	var L = sizes.length;
	var W, B; // W[k][i][j]: layer k neuron i -> layer k+1 neuron j ; B[k][j]

	function gauss() { var u = 1 - Math.random(), v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

	function initModel() {
		W = []; B = [];
		for (var k = 0; k < L - 1; k++) {
			var std = Math.sqrt(2 / sizes[k]); // He initialization for ReLU
			var m = [];
			for (var i = 0; i < sizes[k]; i++) {
				var row = [];
				for (var j = 0; j < sizes[k + 1]; j++) row.push(gauss() * std);
				m.push(row);
			}
			W.push(m);
			var b = [];
			for (j = 0; j < sizes[k + 1]; j++) b.push(0.05);
			B.push(b);
		}
	}

	function relu(z) { return z > 0 ? z : 0; }

	function forward(x) {
		var a = [x.slice()], z = [x.slice()];
		for (var k = 0; k < L - 1; k++) {
			var zk = [], ak = [], last = k === L - 2;
			for (var j = 0; j < sizes[k + 1]; j++) {
				var s = B[k][j];
				for (var i = 0; i < sizes[k]; i++) s += a[k][i] * W[k][i][j];
				zk.push(s); ak.push(last ? s : relu(s));
			}
			z.push(zk); a.push(ak);
		}
		return { z: z, a: a };
	}

	function backward(f, y) {
		var yhat = f.a[L - 1][0];
		var delta = []; for (var k = 0; k < L; k++) delta.push(new Array(sizes[k]).fill(0));
		delta[L - 1][0] = yhat - y; // dL/dyhat for L = 1/2 (yhat - y)^2, linear output
		var gW = [], gB = [];
		for (k = L - 2; k >= 0; k--) {
			var gw = [], gb = delta[k + 1].slice();
			for (var i = 0; i < sizes[k]; i++) {
				var row = [], back = 0;
				for (var j = 0; j < sizes[k + 1]; j++) {
					row.push(f.a[k][i] * delta[k + 1][j]);
					back += W[k][i][j] * delta[k + 1][j];
				}
				gw.push(row);
				if (k > 0) delta[k][i] = f.z[k][i] > 0 ? back : 0; // ReLU derivative
			}
			gW[k] = gw; gB[k] = gb;
		}
		return { delta: delta, gW: gW, gB: gB };
	}

	function sample() {
		var x1 = Math.random() * 2 - 1, x2 = Math.random() * 2 - 1;
		return { x: [x1, x2], y: CONFIG.target(x1, x2) };
	}

	// Trains on the shown sample plus a small batch; returns per-weight update sizes for the flash
	function optimize(shown) {
		var acc = null, n = 0, batch = [shown];
		for (var s = 0; s < CONFIG.batchSize; s++) batch.push(sample());
		for (s = 0; s < batch.length; s++) {
			var f = forward(batch[s].x), g = backward(f, batch[s].y);
			if (!acc) acc = g; else {
				for (var k = 0; k < L - 1; k++) for (var i = 0; i < sizes[k]; i++) for (var j = 0; j < sizes[k + 1]; j++) acc.gW[k][i][j] += g.gW[k][i][j];
				for (k = 0; k < L - 1; k++) for (j = 0; j < sizes[k + 1]; j++) acc.gB[k][j] += g.gB[k][j];
			}
			n++;
		}
		var change = [], eta = CONFIG.learningRate;
		for (var k2 = 0; k2 < L - 1; k2++) {
			var ch = [];
			for (var i2 = 0; i2 < sizes[k2]; i2++) {
				var r = [];
				for (var j2 = 0; j2 < sizes[k2 + 1]; j2++) {
					var gr = Math.max(-1, Math.min(1, acc.gW[k2][i2][j2] / n)); // gradient clipping
					W[k2][i2][j2] -= eta * gr; r.push(eta * gr);
				}
				ch.push(r);
			}
			change.push(ch);
			for (j2 = 0; j2 < sizes[k2 + 1]; j2++) B[k2][j2] -= eta * Math.max(-1, Math.min(1, acc.gB[k2][j2] / n));
		}
		return change;
	}

	function evalLoss() {
		var t = 0, m = 64;
		for (var s = 0; s < m; s++) { var d = sample(), e = forward(d.x).a[L - 1][0] - d.y; t += 0.5 * e * e; }
		return t / m;
	}

	initModel();

	// ---- Training state & phases --------------------------------------------
	var state = { step: 0, phase: 'forward', phaseStart: 0, data: null, f: null, g: null, loss: 0, change: null, history: [] };
	var PH = {
		forward: function () { return (L - 1) * CONFIG.stepMs; },
		loss: function () { return CONFIG.lossMs; },
		backward: function () { return (L - 1) * CONFIG.stepMs; },
		update: function () { return CONFIG.updateMs + CONFIG.pauseMs; }
	};
	var ORDER = ['forward', 'loss', 'backward', 'update'];

	function newSample(t) {
		state.data = sample();
		state.f = forward(state.data.x);
		var e = state.f.a[L - 1][0] - state.data.y;
		state.loss = 0.5 * e * e;
		state.g = null; state.change = null;
		state.phase = 'forward'; state.phaseStart = t;
		state.step++;
	}

	function enterPhase(p, t) {
		state.phase = p; state.phaseStart = t;
		if (p === 'backward') state.g = backward(state.f, state.data.y);
		if (p === 'update') {
			state.change = optimize(state.data);
			for (var u = 1; u < CONFIG.updatesPerCycle; u++) {
				var more = optimize(sample());
				for (var a = 0; a < L - 1; a++) for (var i = 0; i < sizes[a]; i++) for (var j = 0; j < sizes[a + 1]; j++) state.change[a][i][j] += more[a][i][j];
			}
			state.updates = (state.updates || 0) + CONFIG.updatesPerCycle;
			var avg = evalLoss();
			// restart if every neuron in some layer has died (a known ReLU failure mode in deep, narrow nets)
			if (!isFinite(avg) || (state.step > 40 && state.history.length > 10 && Math.abs(avg - state.history[state.history.length - 10]) < 1e-9 && avg > 0.05)) {
				initModel(); state.history = []; state.step = 0; state.updates = 0;
			} else state.history.push(avg);
			if (state.history.length > 160) state.history.shift();
			updatePanel(true);
		}
	}

	function tick(t) {
		if (!state.data) newSample(t);
		var el = t - state.phaseStart;
		if (el > PH[state.phase]()) {
			var idx = ORDER.indexOf(state.phase);
			if (idx === ORDER.length - 1) newSample(t);
			else enterPhase(ORDER[idx + 1], t);
		}
	}

	// ---- Layout --------------------------------------------------------------
	var Wd, Ht, vertical, nodes = [], mouse = { x: -9999, y: -9999 }, hover = null, lastHover = null, drag = null, lastT = 0;

	// ---- Cosmic synth sounds (synthesized with Web Audio, no audio files) ----
	var sfx = (function () {
		var ac = null, on = true, lastBlip = 0, echoIn = null;
		var SCALE = [110, 130.81, 146.83, 164.81, 196, 220, 261.63, 293.66]; // A minor pentatonic-ish, low register
		// filtered feedback delay shared by all bleeps (dotted-eighth style echo)
		function echo(a) {
			if (echoIn) return echoIn;
			echoIn = a.createGain(); echoIn.gain.value = 0.8;
			var d = a.createDelay(1), fb = a.createGain(), lp = a.createBiquadFilter(), out = a.createGain();
			d.delayTime.value = 0.46; fb.gain.value = 0.58; lp.type = 'lowpass'; lp.frequency.value = 1500; out.gain.value = 0.75; // long, dark, repeating echo
			echoIn.connect(d); d.connect(lp); lp.connect(fb); fb.connect(d); lp.connect(out); out.connect(a.destination);
			return echoIn;
		}
		function ctxA() {
			if (!on) return null;
			if (!ac) { var AC = window.AudioContext || window.webkitAudioContext; if (!AC) return null; ac = new AC(); }
			if (ac.state === 'suspended') ac.resume();
			return ac;
		}
		function env(a, g, peak, attack, decay) {
			var t0 = a.currentTime;
			g.gain.setValueAtTime(0.0001, t0);
			g.gain.exponentialRampToValueAtTime(peak, t0 + attack);
			g.gain.exponentialRampToValueAtTime(0.0001, t0 + attack + decay);
		}
		return {
			setOn: function (v) { on = v; if (!v && ac) ac.suspend(); },
			isOn: function () { return on; },
			unlock: function () { ctxA(); },
			// Cosmic synth bleeps in the style of classic analog sequencer music:
			// a short pulse wave through a resonant low-pass filter whose cutoff sweeps,
			// sent into a filtered feedback echo for a spacey tail.
			bleep: function (freq, when, opts) {
				var a = ctxA(); if (!a) return;
				opts = opts || {};
				var t = a.currentTime + (when || 0), len = opts.len || 0.16;
				var o = a.createOscillator(), o2 = a.createOscillator(), f = a.createBiquadFilter(), g = a.createGain();
				o.type = 'square'; o2.type = 'sawtooth';
				o.frequency.setValueAtTime(freq, t); o2.frequency.setValueAtTime(freq * 1.005, t); // slight detune
				if (opts.glide) { o.frequency.exponentialRampToValueAtTime(freq * opts.glide, t + len); o2.frequency.exponentialRampToValueAtTime(freq * opts.glide * 1.005, t + len); }
				f.type = 'lowpass'; f.Q.setValueAtTime(opts.q || 14, t);
				var c0 = opts.from || 300, c1 = opts.to || 4200;
				f.frequency.setValueAtTime(c0, t);
				f.frequency.exponentialRampToValueAtTime(c1, t + len * 0.35);
				f.frequency.exponentialRampToValueAtTime(Math.max(200, c0 * 0.8), t + len);
				g.gain.setValueAtTime(0.0001, t);
				g.gain.exponentialRampToValueAtTime(opts.vol || 0.09, t + 0.008);
				g.gain.exponentialRampToValueAtTime(0.0001, t + len);
				o.connect(f); o2.connect(f); f.connect(g);
				g.connect(a.destination); g.connect(echo(a));
				o.start(t); o2.start(t); o.stop(t + len + 0.02); o2.stop(t + len + 0.02);
			},
			// grab: one rising "zwoop" bleep, pitched by layer
			grab: function (layer) {
				var base = SCALE[(layer || 0) % SCALE.length];
				this.bleep(base, 0, { len: 0.55, glide: 1.6, from: 140, to: 2600, q: 14, vol: 0.11 });
			},
			// release: a short sequencer run; longer pulls play more steps
			release: function (layer, dist) {
				var steps = 4 + Math.min(4, Math.round((dist || 0) / 70));
				var root = (layer || 0) % 3, pattern = [0, 4, 2, 5, 3, 6, 4, 7];
				for (var i = 0; i < steps; i++) {
					var f = SCALE[(root + pattern[i]) % SCALE.length] * (pattern[i] + root >= SCALE.length ? 2 : 1);
					this.bleep(f, i * 0.2, { len: 0.34, from: 160 + i * 40, to: 1400 + i * 300, q: 12, vol: 0.1 });
				}
			},
			// tiny soft blip on hover, pitched by layer (low on the left, high on the right)
			blip: function (layer) {
				var a = ac; if (!a || !on || a.state !== 'running') return; // only after the first click/tap
				var now = performance.now(); if (now - lastBlip < 90) return; lastBlip = now;
				var o = a.createOscillator(), g = a.createGain(), t0 = a.currentTime;
				o.type = 'sine'; o.frequency.setValueAtTime(420 + layer * 55, t0);
				env(a, g, 0.06, 0.004, 0.06); o.connect(g).connect(a.destination); o.start(t0); o.stop(t0 + 0.08);
			}
		};
	})();

	function layout() {
		Wd = window.innerWidth; Ht = window.innerHeight;
		canvas.width = Wd * dpr; canvas.height = Ht * dpr;
		canvas.style.width = Wd + 'px'; canvas.style.height = Ht + 'px';
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		vertical = Ht > Wd * 1.1;
		var panel = document.getElementById('nn-ui') || document.getElementById('nn-panel');
		var panelH = panel ? panel.offsetHeight + 16 : 0;
		var hdr = document.querySelector('header.title') || document.getElementById('header'), hdrH = hdr ? hdr.offsetHeight : 60;
		var top = hdrH + (vertical ? 34 : 40), bottom = Ht - panelH - (vertical ? 10 : 28);
		var left = Wd * (vertical ? 0.1 : 0.045), right = Wd * (vertical ? 0.9 : 0.955);
		nodes = [];
		for (var k = 0; k < L; k++) {
			var layer = [], frac = k / (L - 1);
			for (var n = 0; n < sizes[k]; n++) {
				// spread each layer evenly; smaller layers are centered with the same spacing
				var c = sizes[k] === 1 ? 0.5 : (CONFIG.neuronsPerHidden - 1) > 0 ?
					0.5 + (n - (sizes[k] - 1) / 2) / (CONFIG.neuronsPerHidden - 1) : 0.5;
				if (sizes[k] === CONFIG.inputs && k === 0 && sizes[k] > 1) c = 0.5 + (n - (sizes[k] - 1) / 2) * 0.5;
				var x = vertical ? left + (right - left) * c : left + (right - left) * frac;
				var y = vertical ? top + (bottom - top) * frac : top + (bottom - top) * c;
				layer.push({ bx: x, by: y, x: x, y: y, phase: Math.random() * 6.28, ox: 0, oy: 0, vx: 0, vy: 0 });
			}
			nodes.push(layer);
		}
	}

	// ---- Helpers -------------------------------------------------------------
	function fmt(v) { return (v >= 0 ? '' : '−') + Math.abs(v).toFixed(2); }
	function sub(n) { return String(n).split('').map(function (d) { return '₀₁₂₃₄₅₆₇₈₉'[+d]; }).join(''); }
	function layerName(k) { return k === 0 ? 'Input' : k === L - 1 ? 'Output' : 'Hidden ' + k; }
	function neuronName(k, n) { return k === 0 ? 'x' + sub(n + 1) : k === L - 1 ? 'ŷ' : 'h' + sub(k) + ',' + sub(n + 1); }
	function ease(p) { return p * p * (3 - 2 * p); }

	// ---- Drawing -------------------------------------------------------------
	function draw(t) {
		if (!zoneVisible) {
			if (!pausedAt) { pausedAt = t; ctx.clearRect(0, 0, Wd, Ht); if (tip) tip.style.display = 'none'; document.body.style.cursor = ''; }
			raf = requestAnimationFrame(draw); return;
		}
		if (pausedAt) { state.phaseStart += t - pausedAt; pausedAt = 0; lastT = t; }
		if (!reduce) tick(t);
		ctx.clearRect(0, 0, Wd, Ht);
		var C = CONFIG.colors, el = t - state.phaseStart, ph = state.phase;

		// background glows
		var glows = [[0.25, 0.35, '46,134,232', 0], [0.75, 0.6, '33,178,166', 2]];
		for (var gI = 0; gI < glows.length; gI++) {
			var G = glows[gI], gx = Wd * G[0] + Math.sin(t / 6500 + G[3]) * Wd * 0.1, gy = Ht * G[1] + Math.cos(t / 8000 + G[3]) * Ht * 0.08;
			var grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, Math.max(Wd, Ht) * 0.45);
			grad.addColorStop(0, 'rgba(' + G[2] + ',0.18)'); grad.addColorStop(1, 'rgba(' + G[2] + ',0)');
			ctx.fillStyle = grad; ctx.fillRect(0, 0, Wd, Ht);
		}

		// node float, drag offset with spring-back, hover detection.
		// Dragging only moves the drawing: weights, activations and the home position never change.
		var dt = Math.min(2.5, Math.max(0.5, (t - (lastT || t)) / 16.7)); lastT = t;
		var best = 34 * 34; hover = null;
		for (var k = 0; k < L; k++) for (var n = 0; n < sizes[k]; n++) {
			var N = nodes[k][n];
			var fx = N.bx + Math.sin(t / 1900 + N.phase) * 2.5, fy = N.by + Math.cos(t / 2200 + N.phase) * 2.5;
			if (drag && drag.k === k && drag.n === n) {
				N.ox = mouse.x - fx; N.oy = mouse.y - fy; N.vx = N.vy = 0;
			} else if (N.ox || N.oy || N.vx || N.vy) {
				// damped spring: overshoots and wobbles back to its home position
				N.vx += (-0.12 * N.ox - 0.13 * N.vx) * dt; N.vy += (-0.12 * N.oy - 0.13 * N.vy) * dt;
				N.ox += N.vx * dt; N.oy += N.vy * dt;
				if (Math.abs(N.ox) + Math.abs(N.oy) + Math.abs(N.vx) + Math.abs(N.vy) < 0.05) N.ox = N.oy = N.vx = N.vy = 0;
			}
			N.x = fx + N.ox; N.y = fy + N.oy;
			var dx = mouse.x - N.x, dy = mouse.y - N.y, d2 = dx * dx + dy * dy;
			if (d2 < best) { best = d2; hover = [k, n]; }
		}
		if (drag) hover = [drag.k, drag.n];
		if (hover && (!lastHover || hover[0] !== lastHover[0] || hover[1] !== lastHover[1]) && !drag) sfx.blip(hover[0]);
		lastHover = hover;
		document.body.style.cursor = drag ? 'grabbing' : hover ? 'grab' : '';

		// connections
		var fwdP = ph === 'forward' ? el / CONFIG.stepMs : (ph === 'loss' || ph === 'backward' || ph === 'update') ? 99 : -1;
		var bwdP = ph === 'backward' ? el / CONFIG.stepMs : -1;
		var updP = ph === 'update' ? Math.min(1, el / CONFIG.updateMs) : -1;
		var maxChange = 1e-9;
		if (state.change) for (k = 0; k < L - 1; k++) for (var i0 = 0; i0 < sizes[k]; i0++) for (var j0 = 0; j0 < sizes[k + 1]; j0++) maxChange = Math.max(maxChange, Math.abs(state.change[k][i0][j0]));

		for (k = 0; k < L - 1; k++) {
			for (var i = 0; i < sizes[k]; i++) {
				for (var j = 0; j < sizes[k + 1]; j++) {
					var w = W[k][i][j], A = nodes[k][i], Bn = nodes[k + 1][j];
					var aw = Math.min(1, Math.abs(w));
					var col = w >= 0 ? C.positive : C.negative;
					var hlIn = hover && hover[0] === k + 1 && hover[1] === j;
					var hlOut = hover && hover[0] === k && hover[1] === i;
					var alpha = hlIn ? 0.85 : hlOut ? 0.45 : 0.08 + aw * 0.16;
					ctx.strokeStyle = 'rgba(' + col + ',' + alpha.toFixed(3) + ')';
					ctx.lineWidth = hlIn ? 2 : 0.5 + aw * 1.1;
					ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(Bn.x, Bn.y); ctx.stroke();

					// update flash
					if (updP >= 0 && state.change) {
						var u = Math.abs(state.change[k][i][j]) / maxChange;
						if (u > 0.05) {
							ctx.strokeStyle = 'rgba(' + C.update + ',' + (u * (1 - updP) * 0.8).toFixed(3) + ')';
							ctx.lineWidth = 1 + u * 1.5;
							ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(Bn.x, Bn.y); ctx.stroke();
						}
					}

					// forward pulse: carries a_i * w
					var pf = fwdP - k;
					if (pf >= 0 && pf <= 1 && state.f) {
						var sig = Math.abs(state.f.a[k][i] * w);
						if (sig > 0.02) {
							var e1 = ease(pf), px = A.x + (Bn.x - A.x) * e1, py = A.y + (Bn.y - A.y) * e1;
							ctx.fillStyle = 'rgba(' + col + ',' + Math.min(1, 0.45 + sig).toFixed(3) + ')';
							ctx.beginPath(); ctx.arc(px, py, 1.4 + Math.min(3, sig * 2.5), 0, 6.283); ctx.fill();
						}
					}

					// backward pulse: carries delta_j * w back toward neuron i
					var pb = bwdP - (L - 2 - k);
					if (pb >= 0 && pb <= 1 && state.g) {
						var gs = Math.abs(state.g.delta[k + 1][j] * w);
						if (gs > 0.005) {
							var e2 = ease(pb), qx = Bn.x + (A.x - Bn.x) * e2, qy = Bn.y + (A.y - Bn.y) * e2;
							ctx.fillStyle = 'rgba(' + C.gradient + ',' + Math.min(1, 0.5 + gs * 3).toFixed(3) + ')';
							ctx.beginPath(); ctx.arc(qx, qy, 1.4 + Math.min(3, gs * 6), 0, 6.283); ctx.fill();
						}
					}

					// hover: continuous flow into the hovered neuron, with weight labels
					if (hlIn && state.f) {
						var hp = ((t / 900) + i * 0.17) % 1, e3 = ease(hp);
						ctx.fillStyle = 'rgba(' + col + ',0.95)';
						ctx.beginPath(); ctx.arc(A.x + (Bn.x - A.x) * e3, A.y + (Bn.y - A.y) * e3, 3, 0, 6.283); ctx.fill();
						var mx = A.x + (Bn.x - A.x) * 0.5, my = A.y + (Bn.y - A.y) * 0.5;
						ctx.font = '600 11px "Segoe UI", Arial, sans-serif'; ctx.textAlign = 'center';
						var lbl = 'w' + sub(i + 1) + '=' + fmt(w);
						var tw = ctx.measureText(lbl).width + 8;
						ctx.fillStyle = 'rgba(16,24,38,0.85)'; ctx.fillRect(mx - tw / 2, my - 9, tw, 16);
						ctx.fillStyle = 'rgba(' + col + ',1)'; ctx.fillText(lbl, mx, my + 3);
					}
				}
			}
		}

		// neurons
		for (k = 0; k < L; k++) {
			var reached = ph === 'forward' ? Math.floor(fwdP) >= k : true;
			for (n = 0; n < sizes[k]; n++) {
				var Nd = nodes[k][n], isIO = k === 0 || k === L - 1;
				var act = state.f ? state.f.a[k][n] : 0, R = isIO ? 9 : 6;
				var on = reached && Math.abs(act) > 1e-6;
				var glow = on ? Math.min(1, Math.abs(act)) : 0;
				var c = on ? (act >= 0 ? C.positive : C.negative) : C.neutral;
				if (ph === 'backward' && state.g) {
					var bReach = Math.floor(bwdP) >= (L - 1 - k);
					var dv = state.g.delta[k][n];
					if (bReach && Math.abs(dv) > 1e-6) { c = C.gradient; glow = Math.min(1, Math.abs(dv) * 4); }
				}
				if (ph === 'loss' && k === L - 1) { c = C.gradient; glow = 0.6 + 0.4 * Math.sin(el / 120); }
				if (hover && hover[0] === k && hover[1] === n) glow = Math.max(glow, 0.8);
				if (glow > 0.03) {
					ctx.fillStyle = 'rgba(' + c + ',' + (glow * 0.28).toFixed(3) + ')';
					ctx.beginPath(); ctx.arc(Nd.x, Nd.y, R * (2 + glow * 1.8), 0, 6.283); ctx.fill();
				}
				ctx.fillStyle = 'rgba(16,24,38,0.9)';
				ctx.beginPath(); ctx.arc(Nd.x, Nd.y, R, 0, 6.283); ctx.fill();
				ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(' + c + ',' + (0.55 + glow * 0.45).toFixed(3) + ')'; ctx.stroke();
				ctx.fillStyle = 'rgba(' + c + ',' + (0.12 + glow * 0.88).toFixed(3) + ')';
				ctx.beginPath(); ctx.arc(Nd.x, Nd.y, R * 0.5, 0, 6.283); ctx.fill();
				if (isIO) {
					ctx.font = '700 12px "Segoe UI", Arial, sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.9)';
					ctx.textAlign = 'center';
					if (vertical && k === L - 1) { ctx.textAlign = 'left'; ctx.fillText(neuronName(k, n) + (state.f ? ' = ' + fmt(act) : ''), Nd.x + R + 8, Nd.y + 4); }
					else ctx.fillText(neuronName(k, n) + (state.f ? ' = ' + fmt(act) : ''), Nd.x, Nd.y + (vertical && k === 0 ? -R - 10 : R + 18));
				}
			}
		}

		// layer labels
		ctx.font = '600 10px "Segoe UI", Arial, sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.55)';
		for (k = 0; k < L; k++) {
			var lb = k === 0 ? 'INPUT' : k === L - 1 ? 'OUTPUT' : 'H' + k + (vertical ? '' : ' · ReLU');
			if (vertical) { ctx.textAlign = 'left'; ctx.fillText(lb, 6, nodes[k][0].by - 12); }
			else { ctx.textAlign = 'center'; ctx.fillText(lb, nodes[k][0].bx, nodes[1][0].by - 24); }
		}

		updateTip();
		updatePanel(false);
		if (!reduce) raf = requestAnimationFrame(draw);
	}

	// ---- Tooltip: the actual algorithm for the hovered neuron ----------------
	function updateTip() {
		if (!tip) return;
		if (!hover || !state.f) { tip.style.display = 'none'; return; }
		var k = hover[0], n = hover[1], f = state.f, html = '<b>' + layerName(k) + ' · ' + neuronName(k, n) + '</b>';
		if (k === 0) {
			html += '<div>Input value: ' + neuronName(k, n) + ' = ' + fmt(f.a[0][n]) + '</div>';
		} else {
			var terms = [], prev = k - 1;
			for (var i = 0; i < sizes[prev]; i++) {
				var src = prev === 0 ? 'x' + sub(i + 1) : 'a' + sub(i + 1);
				terms.push('(' + fmt(f.a[prev][i]) + ' × ' + fmt(W[prev][i][n]) + ')');
			}
			var sym = [];
			for (i = 0; i < sizes[prev]; i++) sym.push('(' + (prev === 0 ? 'x' : 'a') + sub(i + 1) + '·w' + sub(i + 1) + ')');
			html += '<div class="f">z = ' + sym.join(' + ') + ' + b</div>';
			html += '<div class="n">z = ' + terms.join(' + ') + ' + ' + fmt(B[prev][n]) + '</div>';
			html += '<div class="n">z = <b>' + fmt(f.z[k][n]) + '</b></div>';
			if (k < L - 1) html += '<div class="f">a = ReLU(z) = max(0, z) = <b>' + fmt(f.a[k][n]) + '</b>' + (f.z[k][n] <= 0 ? ' <span class="muted">(inactive)</span>' : '') + '</div>';
			else {
				html += '<div class="f">ŷ = z = <b>' + fmt(f.a[k][n]) + '</b> (linear output)</div>';
				html += '<div class="f">Target y = ' + fmt(state.data.y) + '</div>';
				html += '<div class="f">Loss L = ½(ŷ − y)² = <b>' + state.loss.toFixed(4) + '</b></div>';
			}
		}
		if (state.g && (state.phase === 'backward' || state.phase === 'update')) {
			html += '<div class="g">δ = ∂L/∂z = ' + fmt(state.g.delta[k][n]) + (k > 0 && k < L - 1 && f.z[k][n] <= 0 ? ' (ReLU blocks gradient)' : '') + '</div>';
			if (k > 0) html += '<div class="g">∂L/∂w' + sub(1) + ' = δ · ' + (k - 1 === 0 ? 'x' : 'a') + sub(1) + ' = ' + fmt(state.g.delta[k][n] * f.a[k - 1][0]) + '</div>';
		}
		tip.innerHTML = html;
		tip.style.display = 'block';
		var N = nodes[k][n], tw = tip.offsetWidth, th = tip.offsetHeight;
		var x = N.x + 18, y = N.y - th / 2;
		if (x + tw > Wd - 8) x = N.x - tw - 18;
		y = Math.max(8, Math.min(Ht - th - 8, y));
		tip.style.left = Math.max(8, x) + 'px'; tip.style.top = y + 'px';
	}

	// ---- Side panel: steps, live numbers, loss chart -------------------------
	var lastPanelPhase = null;
	function setText(id, v) { var e = document.getElementById(id); if (e) e.textContent = v; }
	function updatePanel(force) {
		if (!state.f) return;
		if (state.phase !== lastPanelPhase || force) {
			ORDER.forEach(function (p) { var e = document.getElementById('nn-step-' + p); if (e) e.classList.toggle('active', p === state.phase); });
			lastPanelPhase = state.phase;
			var f = state.f, x = state.data.x;
			setText('nn-v-forward', 'z = (' + fmt(x[0]) + ' × ' + fmt(W[0][0][0]) + ') + (' + fmt(x[1]) + ' × ' + fmt(W[0][1][0]) + ') + ' + fmt(B[0][0]) + ' = ' + fmt(f.z[1][0]) + ',  ReLU → ' + fmt(f.a[1][0]));
			setText('nn-v-loss', 'ŷ = ' + fmt(f.a[L - 1][0]) + ',  y = ' + fmt(state.data.y) + ',  L = ' + state.loss.toFixed(4));
			setText('nn-v-backward', state.g ? 'δₒᵤₜ = ŷ − y = ' + fmt(state.g.delta[L - 1][0]) : 'Errors flow from output back to inputs');
			setText('nn-v-update', 'η = ' + CONFIG.learningRate + ',  ' + (state.updates || 0) + ' updates,  avg loss ' + (state.history.length ? state.history[state.history.length - 1].toFixed(4) : '—'));
			drawChart();
		}
	}

	function drawChart() {
		var c = document.getElementById('nn-loss-chart'); if (!c) return;
		var w = c.clientWidth, h = c.clientHeight, x2 = c.getContext('2d');
		c.width = w * dpr; c.height = h * dpr; x2.setTransform(dpr, 0, 0, dpr, 0, 0);
		x2.clearRect(0, 0, w, h);
		if (state.history.length < 2) return;
		var hs = [], ema = state.history[0];
		for (var q = 0; q < state.history.length; q++) { ema = q ? 0.6 * ema + 0.4 * state.history[q] : ema; hs.push(ema); } // smoothed trend
		var mx = Math.max.apply(null, hs), mn = Math.min.apply(null, hs); if (mx - mn < 1e-6) mx = mn + 1e-6;
		x2.strokeStyle = 'rgba(255,170,70,0.95)'; x2.lineWidth = 1.5; x2.beginPath();
		for (var i = 0; i < hs.length; i++) {
			var px = (i / (hs.length - 1)) * (w - 4) + 2, py = h - 3 - ((hs[i] - mn) / (mx - mn)) * (h - 6);
			if (i) x2.lineTo(px, py); else x2.moveTo(px, py);
		}
		x2.stroke();
	}

	// ---- Events --------------------------------------------------------------
	var raf;
	function setMouse(x, y) { mouse.x = x; mouse.y = y; }
	window.addEventListener('resize', function () { layout(); if (reduce) draw(0); });
	function nearest(x, y) {
		var best = 30 * 30, hit = null;
		for (var k = 0; k < L; k++) for (var n = 0; n < sizes[k]; n++) {
			var N = nodes[k][n], dx = x - N.x, dy = y - N.y, d2 = dx * dx + dy * dy;
			if (d2 < best) { best = d2; hit = { k: k, n: n }; }
		}
		return hit;
	}
	function startDrag(x, y) {
		if (!inZone(x, y)) return false;
		if (!ZONE) sfx.unlock(); // on the website, sound starts only after a neuron is grabbed
		setMouse(x, y);
		var hit = nearest(x, y);
		if (hit) { sfx.unlock(); drag = hit; sfx.grab(hit.k); return true; }
		return false;
	}
	function endDrag() {
		if (!drag) return;
		var N = nodes[drag.k][drag.n];
		var dist = Math.sqrt(N.ox * N.ox + N.oy * N.oy);
		sfx.release(drag.k, dist);
		drag = null;
	}
	function track(x, y) { if (drag || inZone(x, y)) setMouse(x, y); else setMouse(-9999, -9999); }
	window.addEventListener('mousemove', function (e) { track(e.clientX, e.clientY); });
	window.addEventListener('scroll', function () { if (!drag) setMouse(-9999, -9999); }, { passive: true });
	window.addEventListener('mouseout', function (e) { if (!e.relatedTarget) { endDrag(); setMouse(-9999, -9999); } });
	window.addEventListener('mousedown', function (e) {
		if (e.button !== 0 || (e.target.closest && e.target.closest('button, #nn-panel'))) return;
		if (startDrag(e.clientX, e.clientY)) e.preventDefault();
	});
	window.addEventListener('mouseup', endDrag);
	window.addEventListener('touchstart', function (e) {
		var p = e.touches[0];
		if (e.target.closest && e.target.closest('button')) return;
		if (startDrag(p.clientX, p.clientY)) e.preventDefault();
	}, { passive: false });
	window.addEventListener('touchmove', function (e) {
		var p = e.touches[0];
		if (drag) { setMouse(p.clientX, p.clientY); e.preventDefault(); }
	}, { passive: false });
	window.addEventListener('touchend', endDrag);
	window.addEventListener('touchcancel', endDrag);

	var soundBtn = document.getElementById('nn-sound');
	if (soundBtn) {
		var paint = function () {
			soundBtn.textContent = sfx.isOn() ? '\uD83D\uDD0A Sound on' : '\uD83D\uDD07 Sound off';
			soundBtn.setAttribute('aria-pressed', sfx.isOn() ? 'true' : 'false');
		};
		soundBtn.addEventListener('click', function () { sfx.setOn(!sfx.isOn()); if (sfx.isOn()) { sfx.unlock(); sfx.grab(5); } paint(); });
		paint();
	}
	document.addEventListener('visibilitychange', function () {
		if (reduce) return;
		if (document.hidden) cancelAnimationFrame(raf); else raf = requestAnimationFrame(draw);
	});

	layout();
	if (reduce) { newSample(0); enterPhase('loss', 0); draw(0); }
	else raf = requestAnimationFrame(draw);

	// expose for testing in the console
	window.nnDemo = { sfx: sfx, config: CONFIG, init: initModel, forward: forward, backward: backward, optimize: optimize, evalLoss: evalLoss, sample: sample, state: state };
})();

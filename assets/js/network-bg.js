/* Animated network background drawn over the landing photo.
   Designed and created by Gregory Gasowski (https://gregorygasowski.github.io).
   Original implementation: drifting nodes, proximity links, soft glows, cursor attraction. */
(function () {
	var canvas = document.getElementById('bg-network');
	if (!canvas || !canvas.getContext) return;
	var ctx = canvas.getContext('2d');
	var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
	var dpr = Math.min(window.devicePixelRatio || 1, 2);
	var w, h, nodes = [], glows = [], mouse = { x: -9999, y: -9999 };
	var COLORS = ['56,220,205', '90,170,255', '200,220,240'];
	var LINK = 170;

	function resize() {
		w = window.innerWidth; h = window.innerHeight;
		canvas.width = w * dpr; canvas.height = h * dpr;
		canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		var count = Math.round(Math.min(110, (w * h) / 14000));
		nodes = [];
		for (var i = 0; i < count; i++) {
			nodes.push({
				x: Math.random() * w, y: Math.random() * h,
				vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35,
				r: Math.random() * 2 + 1.4,
				c: COLORS[i % COLORS.length]
			});
		}
		glows = [
			{ x: w * 0.2, y: h * 0.3, r: Math.max(w, h) * 0.45, c: '46,134,232', t: 0 },
			{ x: w * 0.8, y: h * 0.7, r: Math.max(w, h) * 0.4, c: '33,178,166', t: 2 }
		];
	}

	function frame(time) {
		ctx.clearRect(0, 0, w, h);
		var t = (time || 0) / 1000;

		// Soft drifting color glows
		for (var g = 0; g < glows.length; g++) {
			var G = glows[g];
			var gx = G.x + Math.sin(t * 0.15 + G.t) * w * 0.12;
			var gy = G.y + Math.cos(t * 0.12 + G.t) * h * 0.1;
			var grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, G.r);
			grad.addColorStop(0, 'rgba(' + G.c + ',0.28)');
			grad.addColorStop(1, 'rgba(' + G.c + ',0)');
			ctx.fillStyle = grad;
			ctx.fillRect(0, 0, w, h);
		}

		// Move nodes
		for (var i = 0; i < nodes.length; i++) {
			var n = nodes[i];
			var dx = mouse.x - n.x, dy = mouse.y - n.y, d = Math.sqrt(dx * dx + dy * dy);
			if (d < 200) { n.x += dx * 0.004; n.y += dy * 0.004; }
			n.x += n.vx; n.y += n.vy;
			if (n.x < -20) n.x = w + 20; else if (n.x > w + 20) n.x = -20;
			if (n.y < -20) n.y = h + 20; else if (n.y > h + 20) n.y = -20;
		}

		// Links
		ctx.lineWidth = 1.1;
		for (var a = 0; a < nodes.length; a++) {
			for (var b = a + 1; b < nodes.length; b++) {
				var p = nodes[a], q = nodes[b];
				var lx = p.x - q.x, ly = p.y - q.y, ld = lx * lx + ly * ly;
				if (ld < LINK * LINK) {
					var o = 1 - Math.sqrt(ld) / LINK;
					ctx.strokeStyle = 'rgba(' + p.c + ',' + (o * 0.85).toFixed(3) + ')';
					ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
				}
			}
			var m = nodes[a], mx = m.x - mouse.x, my = m.y - mouse.y, md = Math.sqrt(mx * mx + my * my);
			if (md < 200) {
				ctx.strokeStyle = 'rgba(255,255,255,' + ((1 - md / 200) * 0.5).toFixed(3) + ')';
				ctx.beginPath(); ctx.moveTo(m.x, m.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
			}
		}

		// Nodes
		for (var k = 0; k < nodes.length; k++) {
			var N = nodes[k];
			ctx.fillStyle = 'rgba(' + N.c + ',0.18)';
			ctx.beginPath(); ctx.arc(N.x, N.y, N.r * 3, 0, Math.PI * 2); ctx.fill();
			ctx.fillStyle = 'rgba(' + N.c + ',1)';
			ctx.beginPath(); ctx.arc(N.x, N.y, N.r, 0, Math.PI * 2); ctx.fill();
		}

		if (!reduce) raf = requestAnimationFrame(frame);
	}

	var raf;
	window.addEventListener('resize', function () { resize(); if (reduce) frame(0); });
	window.addEventListener('mousemove', function (e) { mouse.x = e.clientX; mouse.y = e.clientY; });
	window.addEventListener('mouseout', function () { mouse.x = mouse.y = -9999; });
	document.addEventListener('visibilitychange', function () {
		if (reduce) return;
		if (document.hidden) cancelAnimationFrame(raf); else raf = requestAnimationFrame(frame);
	});
	resize();
	frame(0);
})();

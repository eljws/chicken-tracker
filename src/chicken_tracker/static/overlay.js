// One frame's overlay - boxes, names, trails, pecks, skeletons, dots, stations - drawn the same way by the
// clip player and the Timeline. The caller sets ctx's transform (zoom/pan) and passes:
//   D overlay.json, P poses.json (or null), DT dots.json (or null, with DT.ci = {hen: chicken index}),
//   frame, layers {boxes, trails, stations, hidden, groups, pecks, dots, skeletons}, focus (chicken index|null),
//   s (canvas px per video px, before zoom), u (size unit, already divided by the zoom), U (size unit on screen),
//   warnY / warnX (canvas px: baseline and left of the illegal-move notes, drawn unzoomed), before(s, u, lw, fs, label) optional.
// Returns {status: per-chicken text, visible: Set of chicken indexes}.
const CO = (() => {
  const ST_COL = { feeder: "#ff9800", water: "#29b6f6" }, ACT = { 1: "eating", 2: "drinking" };
  function boxesAt(D, f) {
    const a = D.frames[f]; const out = [];
    if (a) for (let k = 0; k < a.length; k += 6) out.push({ c: a[k], b: [a[k + 1], a[k + 2], a[k + 3], a[k + 4]], st: a[k + 5] });
    return out;
  }
  function draw(ctx, o) {
    const { D, P, DT, frame, layers, focus, s, u } = o, t = frame / D.fps;
    const lw = 2 * u, fs = 13 * u;
    ctx.font = `600 ${fs}px system-ui`; ctx.lineJoin = "round";
    const alpha = i => focus === null || focus === undefined || focus === i ? 1 : 0.25;
    const label = (txt, x, y, col, a) => {
      ctx.globalAlpha = a; ctx.lineWidth = 2 * u; ctx.strokeStyle = "rgba(0,0,0,.85)"; ctx.strokeText(txt, x, y);
      ctx.fillStyle = col; ctx.fillText(txt, x, y); ctx.globalAlpha = 1;
    };
    if (layers.stations) for (const st of D.stations) {
      const [x1, y1, x2, y2] = st.box.map(v => v * s);
      ctx.strokeStyle = ST_COL[st.type] || "#fff"; ctx.lineWidth = lw; ctx.setLineDash([]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); label(st.type.toUpperCase(), x1, y2 + fs + u, ctx.strokeStyle, 1);
    }
    const now = boxesAt(D, frame), status = D.chickens.map(() => "away"), visible = new Set();
    const grp = (D.groups || {})[frame];
    if (layers.groups && grp) for (let k = 0; k < grp.length; k += 4) {
      const [x1, y1, x2, y2] = grp.slice(k, k + 4).map(v => v * s);
      ctx.strokeStyle = "rgba(220,220,220,.8)"; ctx.lineWidth = lw; ctx.setLineDash([4 * u, 4 * u]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
      label("2+ chickens", x1, y2 + fs + u, "#ddd", 0.9);
    }
    if (layers.trails) {
      const back = Math.round(4 * D.fps), pts = D.chickens.map(() => []);
      for (let f = frame - back; f <= frame; f++) for (const o of boxesAt(D, f)) pts[o.c].push([(o.b[0] + o.b[2]) / 2 * s, o.b[3] * s, f]);
      pts.forEach((p, i) => {
        if (p.length < 2) return;
        ctx.globalAlpha = alpha(i) * 0.9; ctx.strokeStyle = D.chickens[i].color; ctx.lineWidth = lw; ctx.beginPath();
        for (let k = 0; k < p.length; k++) {
          if (k && p[k][2] - p[k - 1][2] > 8) ctx.moveTo(p[k][0], p[k][1]); else k ? ctx.lineTo(p[k][0], p[k][1]) : ctx.moveTo(p[k][0], p[k][1]);
        }
        ctx.stroke(); ctx.globalAlpha = 1;
      });
    }
    if (o.before) o.before(s, u, lw, fs, label);  // the player's "picture's piece" box: under the names
    for (const o of now) {
      const c = D.chickens[o.c], [x1, y1, x2, y2] = o.b.map(v => v * s);
      visible.add(o.c); status[o.c] = o.st ? ACT[o.st] : "visible";
      if (!layers.boxes) continue;
      ctx.globalAlpha = alpha(o.c); ctx.strokeStyle = c.color; ctx.lineWidth = focus === o.c ? lw * 1.8 : lw; ctx.setLineDash([]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.globalAlpha = 1;
      label(c.name + (o.st ? ` (${ACT[o.st]})` : ""), x1, y1 - 3 * u, c.color, alpha(o.c));
    }
    const sk = layers.skeletons && P && P.frames[frame];
    if (sk) {
      const K = P.keypoints.length, n = 1 + 3 * K, r = 3 * u;
      for (let k = 0; k < sk.length; k += n) {
        const ci = sk[k], c = D.chickens[ci], pts = [];
        for (let j = 0; j < K; j++) pts.push([sk[k + 1 + 3 * j] * s, sk[k + 2 + 3 * j] * s, sk[k + 3 + 3 * j]]);
        ctx.globalAlpha = alpha(ci);
        ctx.lineWidth = lw * 1.4; ctx.strokeStyle = "rgba(0,0,0,.7)";   // dark under-stroke so it shows on white hens
        for (const [a, b] of P.skeleton) if (pts[a][2] && pts[b][2]) { ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke(); }
        ctx.lineWidth = lw * 0.8; ctx.strokeStyle = c.color;
        for (const [a, b] of P.skeleton) if (pts[a][2] && pts[b][2]) { ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke(); }
        pts.forEach(([x, y, seen], j) => {
          if (!seen) return;
          ctx.beginPath(); ctx.arc(x, y, j === 0 ? r * 1.3 : r, 0, 7);
          ctx.fillStyle = j === 0 ? "#ff5252" : c.color; ctx.fill();   // beak in red: shows which way she faces
          ctx.lineWidth = u / 2; ctx.strokeStyle = "#000"; ctx.stroke();
        });
        ctx.globalAlpha = 1;
      }
    }
    if (layers.pecks && D.pecks) for (let back = 0; back < 5; back++) {   // a ring that grows and fades over ~0.3 s
      const pk = D.pecks[frame - back];
      if (!pk) continue;
      for (let k = 0; k < pk.length; k += 3) {
        const c = D.chickens[pk[k]];
        ctx.globalAlpha = alpha(pk[k]) * (1 - back / 5); ctx.strokeStyle = "#fff"; ctx.lineWidth = lw;
        ctx.beginPath(); ctx.arc(pk[k + 1] * s, pk[k + 2] * s, (4 + back * 3) * u, 0, 7); ctx.stroke();
        ctx.strokeStyle = c.color; ctx.lineWidth = lw * 0.6; ctx.stroke(); ctx.globalAlpha = 1;
      }
    }
    for (const h of D.hidden) {
      if (t < h.from || t > h.to || visible.has(h.c)) continue;
      const conf = Math.exp(-(t - h.from) / 20);
      status[h.c] = `hidden ${Math.round(conf * 100)}%`;
      if (!layers.hidden) continue;
      const [x1, y1, x2, y2] = h.box.map(v => v * s), c = D.chickens[h.c];
      ctx.globalAlpha = alpha(h.c) * (0.35 + 0.65 * conf); ctx.strokeStyle = c.color; ctx.lineWidth = lw;
      ctx.setLineDash([6 * u, 5 * u]); ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      label(`${c.name}? ${Math.round(conf * 100)}%`, x1, y1 - 3 * u, c.color, alpha(h.c) * (0.35 + 0.65 * conf));
    }
    if (layers.dots && DT) drawDots(s, lw, fs, alpha, label, status, visible, u);
    return { status, visible };

    function drawDots(s, lw, fs, alpha, label, status, visible, u) {
      const r = 7 * u;
      for (const hen of DT.hens) {
        const i = DT.ci[hen], d = DT.dots[hen];
        if (i === undefined || frame >= d.src.length) continue;
        const src = d.src[frame], c = D.chickens[i];
        if (src === "-") continue;
        const x = d.x[frame] * s, y = d.y[frame] * s;
        if (src === "F") {   // followed by pixels: the last 3 s of its path, then a hollow dot
          ctx.globalAlpha = alpha(i); ctx.strokeStyle = c.color; ctx.lineWidth = lw; ctx.setLineDash([3 * u, 3 * u]);
          ctx.beginPath(); let on = false;
          for (let f = Math.max(0, frame - Math.round(3 * DT.fps)); f <= frame; f++) {
            if (d.src[f] !== "F" && d.src[f] !== "S") { on = false; continue; }
            on ? ctx.lineTo(d.x[f] * s, d.y[f] * s) : ctx.moveTo(d.x[f] * s, d.y[f] * s); on = true;
          }
          ctx.stroke(); ctx.setLineDash([]);
          ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.lineWidth = lw * 2; ctx.strokeStyle = "#000"; ctx.stroke();
          ctx.lineWidth = lw * 1.2; ctx.strokeStyle = c.color; ctx.stroke(); ctx.globalAlpha = 1;
          if (!visible.has(i)) { label(hen + " (followed)", x + r + 1.5 * u, y + 2 * u, c.color, alpha(i)); status[i] = "followed by pixels"; }
        } else if (src === "L" || src === "W") {   // not seen: waiting where last seen
          ctx.globalAlpha = alpha(i) * 0.45; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = "#000"; ctx.fill();
          ctx.lineWidth = lw; ctx.strokeStyle = c.color; ctx.stroke();
          ctx.fillStyle = c.color; ctx.textAlign = "center"; ctx.fillText("?", x, y + fs * 0.35); ctx.textAlign = "start"; ctx.globalAlpha = 1;
          if (!visible.has(i) && status[i] === "away") status[i] = "lost — last seen here";
        } else {
          ctx.globalAlpha = alpha(i); ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = c.color; ctx.fill();
          ctx.lineWidth = lw; ctx.strokeStyle = "#000"; ctx.stroke(); ctx.globalAlpha = 1;
        }
      }
      // illegal moves happening now: red ring on the hen's dot and the reason at the top
      const now = DT.illegal.filter(e => frame >= e.from_frame - DT.fps && frame <= e.to_frame + DT.fps);
      now.forEach((e, k) => {
        const d = DT.dots[e.hen];
        if (d && d.src[frame] !== "-") {
          const x = d.x[frame] * s, y = d.y[frame] * s;
          ctx.beginPath(); ctx.arc(x, y, r * 2.4, 0, 7); ctx.lineWidth = lw * 1.5; ctx.strokeStyle = "#ff3b3b"; ctx.stroke();
        }
        // the reason stays fixed at the bottom of the screen, not zoomed with the video
        const U = o.U, f2 = 13 * U, txt = "⚠ " + e.detail, yy = o.warnY - k * 24 * U;
        ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.font = `600 ${f2}px system-ui`;
        const x0 = o.warnX || 0;  // canvas px where the note starts (the video's left edge)
        ctx.fillStyle = "rgba(0,0,0,.72)"; ctx.fillRect(x0, yy - f2 - 4 * U, ctx.measureText(txt).width + 20 * U, f2 + 10 * U);
        ctx.fillStyle = "#ff8a80"; ctx.fillText(txt, x0 + 10 * U, yy); ctx.restore();
      });
    }
  }
  return { draw, boxesAt, ST_COL, ACT };
})();

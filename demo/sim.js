// Bản JS của simulate_rpdi_exit (method_c_rpdi_ee/ablation/step2_ablate_grid.py), min_steps = W.
// H: entropy từng token pha suy nghĩ; b: 1 nếu token là token biên câu; end: natural_think_end_idx hoặc null.
// Trả về số token suy nghĩ trước khi chèn </think>, hoặc null nếu không phanh.
function simulateRpdi(H, b, end, W, lam) {
  const maxStep = end != null ? end : H.length;
  let sG = 0, sL = 0;
  for (let i = 1; i <= maxStep; i++) {
    const h = H[i - 1];
    sG += h; sL += h;
    if (i > W) sL -= H[i - W - 1];
    if (i > 1 && b[i - 2] && i >= W) {
      const gtf = sG / i, ltf = sL / W;
      const rpdi = gtf > 0 ? ltf / gtf : 0;
      if (rpdi > lam) return i - 1;
    }
  }
  return null;
}

// Chuỗi LTF, GTF, RPDI theo bước i (chỉ số mảng i-1), null khi chưa đủ cửa sổ W.
function rpdiSeries(H, W) {
  const n = H.length, ltf = new Array(n).fill(null), gtf = new Array(n), rp = new Array(n).fill(null);
  let sG = 0, sL = 0;
  for (let i = 1; i <= n; i++) {
    sG += H[i - 1]; sL += H[i - 1];
    if (i > W) sL -= H[i - W - 1];
    gtf[i - 1] = sG / i;
    if (i >= W) {
      ltf[i - 1] = sL / W;
      rp[i - 1] = gtf[i - 1] > 0 ? ltf[i - 1] / gtf[i - 1] : 0;
    }
  }
  return { ltf, gtf, rp };
}
if (typeof module !== "undefined") module.exports = { simulateRpdi, rpdiSeries };

/* Hand-written neural-network mathematics. All weights use row-vector xW notation.
 * This file is standalone: browser global NNMath and CommonJS module.exports.
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NNMath = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var RESIDUAL_SCALE = 0.3;

  function sigmoid(x) {
    if (x >= 0) return 1 / (1 + Math.exp(-x));
    var e = Math.exp(x);
    return e / (1 + e);
  }

  function activate(x, kind) {
    switch (kind || 'tanh') {
      case 'sigmoid': return sigmoid(x);
      case 'tanh': return Math.tanh(x);
      case 'relu': return Math.max(0, x);
      case 'leaky-relu': case 'leaky_relu': return x >= 0 ? x : 0.01 * x;
      case 'linear': return x;
      default: throw new Error('Unknown activation: ' + kind);
    }
  }

  function derivative(z, a, kind) {
    switch (kind || 'tanh') {
      case 'sigmoid': if (a === undefined) a = sigmoid(z); return a * (1 - a);
      case 'tanh': if (a === undefined) a = Math.tanh(z); return 1 - a * a;
      case 'relu': return z > 0 ? 1 : 0;
      case 'leaky-relu': case 'leaky_relu': return z >= 0 ? 1 : 0.01;
      case 'linear': return 1;
      default: throw new Error('Unknown activation: ' + kind);
    }
  }

  function makeRng(seed) {
    var state = Number(seed) >>> 0;
    return function () {
      state += 0x6D2B79F5;
      var t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normalRng(rng) {
    var spare = null;
    return function () {
      if (spare !== null) { var result = spare; spare = null; return result; }
      var u = Math.max(Number.MIN_VALUE, rng());
      var v = rng();
      var r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
  }

  function zeros(n) { return new Array(n).fill(0); }
  function rms(a) {
    if (!a.length) return 0;
    var sum = 0;
    for (var i = 0; i < a.length; i++) sum += a[i] * a[i];
    return Math.sqrt(sum / a.length);
  }

  function toyForward(state) {
    state = state || {};
    var x = state.x || [0.6, 0.2];
    var W0 = state.W0 || [[0.1, -0.2], [0.4, 0.3]];
    var W1 = state.W1 || [[0.2, -0.3], [-0.4, 0.1]];
    var d = state.d || [1, 0];
    var hidden = W0[0].length, outputs = W1[0].length;
    var a1 = zeros(hidden), h = zeros(hidden), a2 = zeros(outputs);
    var y = zeros(outputs), d2 = zeros(outputs), d1 = zeros(hidden);
    var g0 = W0.map(function (row) { return zeros(row.length); });
    var g1 = W1.map(function (row) { return zeros(row.length); });
    var i, j, k, loss = 0;
    for (j = 0; j < hidden; j++) {
      for (i = 0; i < x.length; i++) a1[j] += x[i] * W0[i][j];
      h[j] = sigmoid(a1[j]);
    }
    for (k = 0; k < outputs; k++) {
      for (j = 0; j < hidden; j++) a2[k] += h[j] * W1[j][k];
      y[k] = sigmoid(a2[k]);
      loss += 0.5 * (y[k] - d[k]) * (y[k] - d[k]);
      d2[k] = (y[k] - d[k]) * y[k] * (1 - y[k]);
    }
    for (j = 0; j < hidden; j++) {
      for (k = 0; k < outputs; k++) {
        d1[j] += W1[j][k] * d2[k];
        g1[j][k] = h[j] * d2[k];
      }
      d1[j] *= h[j] * (1 - h[j]);
      for (i = 0; i < x.length; i++) g0[i][j] = x[i] * d1[j];
    }
    return { a1: a1, h: h, a2: a2, y: y, loss: loss, d2: d2, d1: d1, g0: g0, g1: g1 };
  }

  function toyStep(state, lr) {
    var f = toyForward(state);
    return {
      x: state.x.slice(), d: state.d.slice(),
      W0: state.W0.map(function (row, i) {
        return row.map(function (v, j) { return v - lr * f.g0[i][j]; });
      }),
      W1: state.W1.map(function (row, i) {
        return row.map(function (v, j) { return v - lr * f.g1[i][j]; });
      })
    };
  }

  function softmax(logits, temp) {
    temp = temp === undefined ? 1 : temp;
    if (!(temp > 0) || !Number.isFinite(temp)) throw new Error('Temperature must be positive.');
    if (!logits.length) throw new Error('Softmax needs at least one logit.');
    var max = Math.max.apply(null, logits);
    var p = logits.map(function (v) { return Math.exp((v - max) / temp); });
    var sum = p.reduce(function (a, b) { return a + b; }, 0);
    return p.map(function (v) { return v / sum; });
  }

  function softmaxLossGrad(logits, target, temp) {
    temp = temp === undefined ? 1 : temp;
    var p = softmax(logits, temp);
    var max = Math.max.apply(null, logits);
    var logZ = Math.log(logits.reduce(function (sum, v) {
      return sum + Math.exp((v - max) / temp);
    }, 0));
    var targetVector;
    if (Array.isArray(target) || ArrayBuffer.isView(target)) {
      if (target.length !== p.length) throw new Error('Target size does not match logits.');
      targetVector = Array.from(target);
    } else {
      if (!Number.isInteger(target) || target < 0 || target >= p.length) throw new Error('Invalid class index.');
      targetVector = zeros(p.length); targetVector[target] = 1;
    }
    var loss = 0, targetSum = 0;
    for (var k = 0; k < p.length; k++) {
      loss -= targetVector[k] * ((logits[k] - max) / temp - logZ);
      targetSum += targetVector[k];
    }
    var grad = p.map(function (v, k) { return (v * targetSum - targetVector[k]) / temp; });
    return { p: p, loss: loss, grad: grad };
  }

  function affine(x, w, b) {
    var y = b.data.slice(), cols = w.cols;
    for (var i = 0; i < w.rows; i++) {
      for (var j = 0; j < cols; j++) y[j] += x[i] * w.data[i * cols + j];
    }
    return y;
  }

  class DenseNet {
    constructor(options) {
      var o = options || {};
      this.input = o.input === undefined ? 2 : o.input;
      this.width = o.width === undefined ? 6 : o.width;
      this.depth = o.depth === undefined ? 2 : o.depth;
      this.task = o.task || 'classification';
      this.output = this.task === 'regression' ? 1 : (o.output === undefined ? 2 : o.output);
      this.activation = o.activation || 'tanh';
      this.residual = !!o.residual;
      this.seed = o.seed === undefined ? 42 : o.seed;
      this.gain = o.gain === undefined ? 1 : o.gain;
      this.residualScale = RESIDUAL_SCALE;
      if (!Number.isInteger(this.input) || this.input < 1 || !Number.isInteger(this.width) || this.width < 1 ||
          !Number.isInteger(this.depth) || this.depth < 0 || !Number.isInteger(this.output) || this.output < 1)
        throw new Error('Layer sizes must be positive integers; depth may be zero.');
      if (this.task !== 'regression' && this.task !== 'classification') throw new Error('Unknown task.');
      activate(0, this.activation);
      this.params = [];
      this.layers = [];
      this._rng = normalRng(makeRng(this.seed));
      this._optimizer = null;
      var current = this.input;
      if (this.residual) {
        this._addLayer(current, this.width, 'in', false, true); current = this.width;
        for (var r = 0; r < this.depth; r++) this._addLayer(current, this.width, 'res' + r, true, true);
      } else {
        for (var h = 0; h < this.depth; h++) {
          this._addLayer(current, this.width, String(h), false, true); current = this.width;
        }
      }
      this._addLayer(current, this.output, 'out', false, false);
    }

    _addLayer(rows, cols, name, residual, activated) {
      var he = this.activation === 'relu' || this.activation === 'leaky-relu' || this.activation === 'leaky_relu';
      var scale = this.gain * Math.sqrt((activated && he ? 2 : 1) / rows);
      var w = { name: 'W_' + name, rows: rows, cols: cols, data: zeros(rows * cols) };
      for (var i = 0; i < w.data.length; i++) w.data[i] = this._rng() * scale;
      var b = { name: 'b_' + name, rows: 1, cols: cols, data: zeros(cols) };
      var wi = this.params.length;
      this.params.push(w, b);
      this.layers.push({ name: name, wi: wi, bi: wi + 1, residual: residual, activated: activated });
    }

    forward(x) {
      if (x.length !== this.input) throw new Error('Input dimension mismatch.');
      var h = Array.from(x), caches = [];
      for (var l = 0; l < this.layers.length; l++) {
        var spec = this.layers[l], input = h;
        var z = affine(input, this.params[spec.wi], this.params[spec.bi]);
        var a = spec.activated ? z.map((v) => activate(v, this.activation)) : z.slice();
        h = spec.residual ? a.map((v, j) => input[j] + this.residualScale * v) : a.slice();
        caches.push({ name: spec.name, input: input, z: z, a: a, output: h,
          residual: spec.residual, activated: spec.activated, wi: spec.wi, bi: spec.bi });
      }
      return { output: h, cache: { input: Array.from(x), layers: caches, output: h,
        residualScale: this.residualScale } };
    }

    lossGrad(xs, ys) {
      if (!xs.length || xs.length !== ys.length) throw new Error('Provide equally sized nonempty xs and ys.');
      var grads = this.params.map(function (p) { return zeros(p.data.length); });
      var layerSquares = zeros(this.layers.length), layerCounts = zeros(this.layers.length);
      var loss = 0, n = xs.length;
      for (var sample = 0; sample < n; sample++) {
        var f = this.forward(xs[sample]), delta;
        if (this.task === 'classification') {
          var sm = softmaxLossGrad(f.output, ys[sample]);
          loss += sm.loss; delta = sm.grad;
        } else {
          var target = Array.isArray(ys[sample]) || ArrayBuffer.isView(ys[sample]) ? ys[sample][0] : ys[sample];
          var error = f.output[0] - target;
          loss += 0.5 * error * error; delta = [error];
        }
        for (var l = this.layers.length - 1; l >= 0; l--) {
          var c = f.cache.layers[l], w = this.params[c.wi], cols = w.cols;
          var dz = zeros(cols), dx = zeros(w.rows);
          for (var j = 0; j < cols; j++) {
            layerSquares[l] += delta[j] * delta[j]; layerCounts[l]++;
            dz[j] = delta[j] * (c.residual ? this.residualScale : 1) *
              (c.activated ? derivative(c.z[j], c.a[j], this.activation) : 1);
            grads[c.bi][j] += dz[j] / n;
          }
          for (var i = 0; i < w.rows; i++) {
            for (var k = 0; k < cols; k++) {
              grads[c.wi][i * cols + k] += c.input[i] * dz[k] / n;
              dx[i] += w.data[i * cols + k] * dz[k];
            }
            if (c.residual) dx[i] += delta[i];
          }
          delta = dx;
        }
      }
      return { loss: loss / n, grads: grads, gradRms: grads.map(rms),
        layerGradRms: layerSquares.map(function (s, i) { return Math.sqrt(s / layerCounts[i]); }) };
    }

    step(grads, lr, optimizer) {
      optimizer = optimizer || 'adam';
      optimizer = optimizer.toLowerCase();
      if (optimizer !== 'adam' && optimizer !== 'sgd' && optimizer !== 'momentum') throw new Error('Unknown optimizer.');
      if (!(lr >= 0) || !Number.isFinite(lr)) throw new Error('Learning rate must be finite and nonnegative.');
      if (grads.length !== this.params.length || grads.some((g, i) => g.length !== this.params[i].data.length))
        throw new Error('Gradient dimensions do not match parameters.');
      if (grads.some(function (g) { return g.some(function (v) { return !Number.isFinite(v); }); }))
        throw new Error('Gradient is not finite; lower the learning rate.');
      if (!this._optimizer || this._optimizer.name !== optimizer) {
        this._optimizer = { name: optimizer, t: 0,
          m: this.params.map(function (p) { return zeros(p.data.length); }),
          v: this.params.map(function (p) { return zeros(p.data.length); }) };
      }
      var state = this._optimizer; state.t++;
      for (var p = 0; p < this.params.length; p++) {
        var data = this.params[p].data;
        for (var j = 0; j < data.length; j++) {
          var g = grads[p][j];
          if (optimizer === 'sgd') data[j] -= lr * g;
          else if (optimizer === 'momentum') {
            state.m[p][j] = 0.9 * state.m[p][j] + g;
            data[j] -= lr * state.m[p][j];
          } else {
            state.m[p][j] = 0.9 * state.m[p][j] + 0.1 * g;
            state.v[p][j] = 0.999 * state.v[p][j] + 0.001 * g * g;
            var mhat = state.m[p][j] / (1 - Math.pow(0.9, state.t));
            var vhat = state.v[p][j] / (1 - Math.pow(0.999, state.t));
            data[j] -= lr * mhat / (Math.sqrt(vhat) + 1e-8);
          }
        }
      }
      return this;
    }

    predict(x) {
      var output = this.forward(x).output;
      return this.task === 'classification' ? softmax(output) : output[0];
    }

    parameterCount() {
      return this.params.reduce(function (sum, p) { return sum + p.data.length; }, 0);
    }
  }

  function convolution(image2d, kernel2d, stride, padding) {
    stride = stride === undefined ? 1 : stride; padding = padding === undefined ? 0 : padding;
    if (!Number.isInteger(stride) || stride < 1 || !Number.isInteger(padding) || padding < 0)
      throw new Error('Stride must be positive and padding nonnegative integers.');
    var ih = image2d.length, iw = ih ? image2d[0].length : 0;
    var kh = kernel2d.length, kw = kh ? kernel2d[0].length : 0;
    if (!ih || !iw || !kh || !kw || image2d.some(function (r) { return r.length !== iw; }) ||
        kernel2d.some(function (r) { return r.length !== kw; })) throw new Error('Use nonempty rectangular arrays.');
    var oh = Math.floor((ih + 2 * padding - kh) / stride) + 1;
    var ow = Math.floor((iw + 2 * padding - kw) / stride) + 1;
    if (oh < 1 || ow < 1) return [];
    var result = [];
    for (var y = 0; y < oh; y++) {
      var row = zeros(ow);
      for (var x = 0; x < ow; x++) {
        for (var ky = 0; ky < kh; ky++) {
          for (var kx = 0; kx < kw; kx++) {
            var iy = y * stride + ky - padding, ix = x * stride + kx - padding;
            if (iy >= 0 && iy < ih && ix >= 0 && ix < iw) row[x] += image2d[iy][ix] * kernel2d[ky][kx];
          }
        }
      }
      result.push(row);
    }
    return result; // Deep-learning convention: cross-correlation, no kernel reversal.
  }

  function dropout(values, prob, seed) {
    if (!(prob >= 0 && prob < 1)) throw new Error('Dropout probability must be in [0, 1).');
    var rng = makeRng(seed === undefined ? 0 : seed), keep = 1 - prob;
    var mask = values.map(function () { return rng() < keep ? 1 / keep : 0; });
    return { values: values.map(function (v, i) { return v * mask[i]; }), mask: mask };
  }

  function batchNorm(values2d, eps) {
    eps = eps === undefined ? 1e-5 : eps;
    var n = values2d.length, d = n ? values2d[0].length : 0;
    if (!n || !d || values2d.some(function (r) { return r.length !== d; })) throw new Error('Use a nonempty batch matrix.');
    if (!(eps > 0)) throw new Error('BatchNorm epsilon must be positive.');
    var mean = zeros(d), variance = zeros(d);
    for (var i = 0; i < n; i++) for (var j = 0; j < d; j++) mean[j] += values2d[i][j] / n;
    for (var k = 0; k < n; k++) for (var c = 0; c < d; c++) variance[c] += Math.pow(values2d[k][c] - mean[c], 2) / n;
    return { mean: mean, variance: variance, normalized: values2d.map(function (row) {
      return row.map(function (v, col) { return (v - mean[col]) / Math.sqrt(variance[col] + eps); });
    }) };
  }

  function gradientCheck(net, xs, ys, eps) {
    eps = eps === undefined ? 1e-5 : eps;
    if (!(eps > 0)) throw new Error('Difference step must be positive.');
    var analytic = net.lossGrad(xs, ys).grads, maxAbs = 0, maxRel = 0, worst = null, checked = 0;
    for (var p = 0; p < net.params.length; p++) {
      for (var j = 0; j < net.params[p].data.length; j++) {
        var old = net.params[p].data[j], plus, minus;
        try {
          net.params[p].data[j] = old + eps; plus = net.lossGrad(xs, ys).loss;
          net.params[p].data[j] = old - eps; minus = net.lossGrad(xs, ys).loss;
        } finally { net.params[p].data[j] = old; }
        var numeric = (plus - minus) / (2 * eps), actual = analytic[p][j];
        var error = Math.abs(numeric - actual);
        var relative = error / Math.max(1e-8, Math.abs(numeric) + Math.abs(actual));
        if (worst === null || error > maxAbs) { maxAbs = error; worst = {
          name: net.params[p].name, index: j, analytic: actual, numeric: numeric }; }
        maxRel = Math.max(maxRel, relative); checked++;
      }
    }
    return { maxAbs: maxAbs, maxRel: maxRel, checked: checked, worst: worst };
  }

  return { sigmoid: sigmoid, activate: activate, derivative: derivative,
    toyForward: toyForward, toyStep: toyStep, softmax: softmax,
    softmaxLossGrad: softmaxLossGrad, DenseNet: DenseNet, convolution: convolution,
    dropout: dropout, batchNorm: batchNorm, gradientCheck: gradientCheck,
    makeRng: makeRng, RESIDUAL_SCALE: RESIDUAL_SCALE };
}));

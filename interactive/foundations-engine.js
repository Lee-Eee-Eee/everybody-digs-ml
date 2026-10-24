/* Explicit scalar computation graphs for teaching the chain rule.
 * All operation derivatives are written below. No external math/autodiff package.
 * W0[i][j] connects input i to hidden unit j; all vectors use xW notation.
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.FoundationMath = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    var e = Math.exp(z); return e / (1 + e);
  }
  function finite(x, name) {
    if (!Number.isFinite(x)) throw new Error(name + ' must be finite.');
    return x;
  }
  function pair(x, name) {
    if (!x || x.length !== 2) throw new Error(name + ' must contain two numbers.');
    return [finite(x[0], name), finite(x[1], name)];
  }
  function matrix(x, name) {
    if (!x || x.length !== 2) throw new Error(name + ' must be 2 by 2.');
    return [pair(x[0], name), pair(x[1], name)];
  }
  function normalizeNetwork(state) {
    var s = state || {};
    return { x: pair(s.x === undefined ? [0.6, 0.2] : s.x, 'x'),
      W0: matrix(s.W0 === undefined ? [[0.1, -0.2], [0.4, 0.3]] : s.W0, 'W0'),
      W1: matrix(s.W1 === undefined ? [[0.2, -0.3], [-0.4, 0.1]] : s.W1, 'W1'),
      d: pair(s.d === undefined ? (s.target === undefined ? [1, 0] : s.target) : s.d, 'd') };
  }
  function normalizeNeuron(state) {
    var s = state || {};
    return { x: finite(s.x === undefined ? 0.6 : s.x, 'x'),
      w: finite(s.w === undefined ? 0.1 : s.w, 'w'),
      b: finite(s.b === undefined ? 0.2 : s.b, 'b'),
      d: finite(s.d === undefined ? (s.target === undefined ? 1 : s.target) : s.d, 'd') };
  }

  function Builder(kind, state) {
    this.kind = kind; this.state = state; this.nodes = []; this.byId = {};
    this.stages = []; this.stageMap = {};
  }
  Builder.prototype.stage = function (id, title) {
    if (!this.stageMap[id]) {
      var s = { id: id, title: title, nodeIds: [] };
      this.stageMap[id] = s; this.stages.push(s);
    }
    return id;
  };
  Builder.prototype.add = function (id, label, op, value, parents, stage, expression, role) {
    if (this.byId[id]) throw new Error('Duplicate graph node ' + id);
    var node = { id: id, label: label, op: op, value: value,
      parents: parents || [], stage: stage, expression: expression || label,
      role: role || 'intermediate', adjoint: 0 };
    this.nodes.push(node); this.byId[id] = node;
    this.stageMap[stage].nodeIds.push(id);
    return id;
  };
  Builder.prototype.leaf = function (id, label, value, stage, role) {
    return this.add(id, label, 'leaf', value, [], stage, label, role);
  };
  Builder.prototype.mul = function (id, label, a, b, stage) {
    var av = this.byId[a].value, bv = this.byId[b].value;
    return this.add(id, label, 'multiply', av * bv,
      [{ id: a, localDerivative: bv, formula: '∂(a×b)/∂a = b' },
       { id: b, localDerivative: av, formula: '∂(a×b)/∂b = a' }],
      stage, this.byId[a].label + ' × ' + this.byId[b].label);
  };
  Builder.prototype.sum = function (id, label, a, b, stage) {
    return this.add(id, label, 'add', this.byId[a].value + this.byId[b].value,
      [{ id: a, localDerivative: 1, formula: '∂(a+b)/∂a = 1' },
       { id: b, localDerivative: 1, formula: '∂(a+b)/∂b = 1' }],
      stage, this.byId[a].label + ' + ' + this.byId[b].label);
  };
  Builder.prototype.subtract = function (id, label, a, b, stage) {
    return this.add(id, label, 'subtract', this.byId[a].value - this.byId[b].value,
      [{ id: a, localDerivative: 1, formula: '∂(a−b)/∂a = 1' },
       { id: b, localDerivative: -1, formula: '∂(a−b)/∂b = −1' }],
      stage, this.byId[a].label + ' − ' + this.byId[b].label);
  };
  Builder.prototype.sigmoid = function (id, label, a, stage) {
    var y = sigmoid(this.byId[a].value);
    return this.add(id, label, 'sigmoid', y,
      [{ id: a, localDerivative: y * (1 - y), formula: 'σ′(z) = σ(z)[1−σ(z)]' }],
      stage, 'σ(' + this.byId[a].label + ')');
  };
  Builder.prototype.square = function (id, label, a, stage) {
    var x = this.byId[a].value;
    return this.add(id, label, 'square', x * x,
      [{ id: a, localDerivative: 2 * x, formula: '∂(e²)/∂e = 2e' }],
      stage, this.byId[a].label + '²');
  };
  Builder.prototype.finish = function (lossId) {
    var adjoints = {}, reverseEvents = [], edges = [], byId = this.byId;
    this.nodes.forEach(function (node) { adjoints[node.id] = 0; });
    adjoints[lossId] = 1;
    for (var k = this.nodes.length - 1; k >= 0; k--) {
      var child = this.nodes[k];
      for (var j = 0; j < child.parents.length; j++) {
        var edge = child.parents[j], before = adjoints[edge.id];
        var contribution = adjoints[child.id] * edge.localDerivative;
        adjoints[edge.id] = before + contribution;
        reverseEvents.push({ index: reverseEvents.length, childId: child.id, parentId: edge.id,
          parentIndex: j, childAdjoint: adjoints[child.id], localDerivative: edge.localDerivative,
          contribution: contribution, before: before, after: adjoints[edge.id],
          childValue: child.value, parentValue: byId[edge.id].value,
          localFormula: edge.formula,
          formula: '∂L/∂' + byId[edge.id].label + ' += (∂L/∂' + child.label + ') × (∂' + child.label + '/∂' + byId[edge.id].label + ')' });
      }
    }
    this.nodes.forEach(function (node) {
      node.adjoint = adjoints[node.id];
      node.parents.forEach(function (p, i) {
        edges.push({ id: p.id + '→' + node.id + ':' + i, parentId: p.id, childId: node.id,
          parentIndex: i, localDerivative: p.localDerivative, formula: p.formula });
      });
    });
    return { kind: this.kind, state: this.state, nodes: this.nodes, byId: this.byId,
      edges: edges, forwardStages: this.stages, reverseEvents: reverseEvents,
      seedEvent: { nodeId: lossId, before: 0, after: 1, formula: '∂L/∂L = 1' },
      adjoints: adjoints, lossId: lossId,
      notation: 'W0[i][j]: input i → hidden j; W1[j][k]: hidden j → output k. Vectors multiply from the left (xW).' };
  };

  function networkGraph(state) {
    var s = normalizeNetwork(state), g = new Builder('network', s);
    var leaves = g.stage('leaves', '输入、权重与目标');
    var hiddenProducts = g.stage('hidden-products', '隐藏层：逐项相乘');
    var hiddenSums = g.stage('hidden-sums', '隐藏层：相加得到加权和');
    var hiddenActivations = g.stage('hidden-activations', '隐藏层：Sigmoid');
    var outputProducts = g.stage('output-products', '输出层：逐项相乘');
    var outputSums = g.stage('output-sums', '输出层：相加得到加权和');
    var outputActivations = g.stage('output-activations', '输出层：Sigmoid');
    var errors = g.stage('errors', '预测减去目标');
    var squares = g.stage('squares', '误差平方');
    var halves = g.stage('halves', '每项平方误差乘 1/2');
    var total = g.stage('total', '两项损失相加');
    var i, j, k;
    for (i = 0; i < 2; i++) g.leaf('x' + (i + 1), 'x' + (i + 1), s.x[i], leaves, 'input');
    for (i = 0; i < 2; i++) for (j = 0; j < 2; j++)
      g.leaf('w0_' + (i + 1) + (j + 1), 'W⁽⁰⁾' + (i + 1) + (j + 1), s.W0[i][j], leaves, 'weight');
    for (j = 0; j < 2; j++) for (k = 0; k < 2; k++)
      g.leaf('w1_' + (j + 1) + (k + 1), 'W⁽¹⁾' + (j + 1) + (k + 1), s.W1[j][k], leaves, 'weight');
    for (k = 0; k < 2; k++) g.leaf('d' + (k + 1), 'd' + (k + 1), s.d[k], leaves, 'target');
    g.leaf('half', '1/2', 0.5, leaves, 'constant');
    for (j = 0; j < 2; j++) {
      for (i = 0; i < 2; i++) g.mul('p0_' + (i + 1) + (j + 1), 'p⁽⁰⁾' + (i + 1) + (j + 1),
        'x' + (i + 1), 'w0_' + (i + 1) + (j + 1), hiddenProducts);
    }
    for (j = 0; j < 2; j++) g.sum('a1_' + (j + 1), 'a⁽¹⁾' + (j + 1),
      'p0_1' + (j + 1), 'p0_2' + (j + 1), hiddenSums);
    for (j = 0; j < 2; j++) g.sigmoid('h' + (j + 1), 'h' + (j + 1), 'a1_' + (j + 1), hiddenActivations);
    for (k = 0; k < 2; k++) for (j = 0; j < 2; j++)
      g.mul('p1_' + (j + 1) + (k + 1), 'p⁽¹⁾' + (j + 1) + (k + 1),
        'h' + (j + 1), 'w1_' + (j + 1) + (k + 1), outputProducts);
    for (k = 0; k < 2; k++) g.sum('a2_' + (k + 1), 'a⁽²⁾' + (k + 1),
      'p1_1' + (k + 1), 'p1_2' + (k + 1), outputSums);
    for (k = 0; k < 2; k++) g.sigmoid('y' + (k + 1), 'y' + (k + 1), 'a2_' + (k + 1), outputActivations);
    for (k = 0; k < 2; k++) g.subtract('e' + (k + 1), 'e' + (k + 1), 'y' + (k + 1), 'd' + (k + 1), errors);
    for (k = 0; k < 2; k++) g.square('q' + (k + 1), 'e' + (k + 1) + '²', 'e' + (k + 1), squares);
    for (k = 0; k < 2; k++) g.mul('loss' + (k + 1), 'L' + (k + 1), 'half', 'q' + (k + 1), halves);
    g.sum('L', 'L', 'loss1', 'loss2', total);
    var graph = g.finish('L'), v = function (id) { return g.byId[id].value; };
    var a = graph.adjoints;
    graph.result = { a1: [v('a1_1'), v('a1_2')], h: [v('h1'), v('h2')],
      a2: [v('a2_1'), v('a2_2')], y: [v('y1'), v('y2')], loss: v('L'),
      d2: [a.a2_1, a.a2_2], d1: [a.a1_1, a.a1_2],
      g0: [[a.w0_11, a.w0_12], [a.w0_21, a.w0_22]],
      g1: [[a.w1_11, a.w1_12], [a.w1_21, a.w1_22]],
      gx: [a.x1, a.x2], gd: [a.d1, a.d2] };
    return graph;
  }

  function neuronGraph(state) {
    var s = normalizeNeuron(state), g = new Builder('neuron', s);
    var leaves = g.stage('leaves', '输入、权重、偏置与目标');
    var product = g.stage('product', '输入与权重相乘');
    var sum = g.stage('sum', '加上偏置');
    var activation = g.stage('activation', '通过 Sigmoid');
    var error = g.stage('error', '预测减去目标');
    var square = g.stage('square', '误差平方');
    var loss = g.stage('loss', '乘以 1/2 得到损失');
    g.leaf('x', 'x', s.x, leaves, 'input'); g.leaf('w', 'w', s.w, leaves, 'weight');
    g.leaf('b', 'b', s.b, leaves, 'bias'); g.leaf('d', 'd', s.d, leaves, 'target');
    g.leaf('half', '1/2', 0.5, leaves, 'constant');
    g.mul('wx', 'wx', 'w', 'x', product); g.sum('z', 'z', 'wx', 'b', sum);
    g.sigmoid('y', 'y', 'z', activation); g.subtract('e', 'e', 'y', 'd', error);
    g.square('q', 'e²', 'e', square); g.mul('L', 'L', 'half', 'q', loss);
    var graph = g.finish('L'), a = graph.adjoints;
    graph.result = { z: g.byId.z.value, y: g.byId.y.value, loss: g.byId.L.value,
      gx: a.x, gw: a.w, gb: a.b, gd: a.d, delta: a.z };
    return graph;
  }

  function replay(graph, eventCount) {
    if (!graph || !graph.reverseEvents || !graph.nodes) throw new Error('Provide a scalar graph.');
    var count = eventCount === undefined ? graph.reverseEvents.length : eventCount;
    if (!Number.isInteger(count) || count < 0 || count > graph.reverseEvents.length)
      throw new Error('eventCount must be between zero and the event count.');
    var adjoints = {};
    graph.nodes.forEach(function (node) { adjoints[node.id] = 0; });
    adjoints[graph.lossId] = 1;
    for (var i = 0; i < count; i++) {
      var event = graph.reverseEvents[i];
      adjoints[event.parentId] += adjoints[event.childId] * event.localDerivative;
    }
    return adjoints;
  }

  function finiteDifference(kind, state, eps) {
    if (kind !== 'neuron' && kind !== 'network') throw new Error('kind must be neuron or network.');
    eps = eps === undefined ? 1e-5 : eps;
    if (!(eps > 0) || !Number.isFinite(eps)) throw new Error('eps must be finite and positive.');
    var graphFn = kind === 'neuron' ? neuronGraph : networkGraph;
    var initial = graphFn(state), normalized = initial.state, paths = [];
    if (kind === 'neuron') {
      ['x', 'w', 'b', 'd'].forEach(function (key) { paths.push({ id: key, path: [key] }); });
    } else {
      for (var i = 0; i < 2; i++) paths.push({ id: 'x' + (i + 1), path: ['x', i] });
      ['W0', 'W1'].forEach(function (key, layer) {
        for (var i = 0; i < 2; i++) for (var j = 0; j < 2; j++)
          paths.push({ id: 'w' + layer + '_' + (i + 1) + (j + 1), path: [key, i, j] });
      });
      for (var d = 0; d < 2; d++) paths.push({ id: 'd' + (d + 1), path: ['d', d] });
    }
    var rows = [], maxAbs = 0;
    paths.forEach(function (item) {
      var container = normalized;
      for (var p = 0; p < item.path.length - 1; p++) container = container[item.path[p]];
      var key = item.path[item.path.length - 1], original = container[key], plus, minus;
      try {
        container[key] = original + eps; plus = graphFn(normalized).result.loss;
        container[key] = original - eps; minus = graphFn(normalized).result.loss;
      } finally { container[key] = original; }
      var numeric = (plus - minus) / (2 * eps), analytic = initial.adjoints[item.id];
      var absoluteError = Math.abs(numeric - analytic); maxAbs = Math.max(maxAbs, absoluteError);
      rows.push({ id: item.id, path: item.path, analytic: analytic, numeric: numeric, absoluteError: absoluteError });
    });
    return { kind: kind, eps: eps, rows: rows, checked: rows.length, maxAbs: maxAbs };
  }

  return { networkGraph: networkGraph, neuronGraph: neuronGraph, replay: replay,
    finiteDifference: finiteDifference, sigmoid: sigmoid };
}));

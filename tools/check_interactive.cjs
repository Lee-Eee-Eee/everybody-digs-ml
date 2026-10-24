#!/usr/bin/env node
'use strict';

/* Run with: node tools/check_interactive.cjs
 * Only Node built-ins are used. No browser, package installation, or training.
 * Finite differences use separately defined scalar objectives and restore every
 * perturbed value. Smooth activations avoid ambiguous derivatives at ReLU zero.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const interactive = path.resolve(__dirname, '..', 'interactive');
const M = require(path.join(interactive, 'engine.js'));
const F = require(path.join(interactive, 'foundations-engine.js'));

const clone = value => JSON.parse(JSON.stringify(value));
const sum = values => values.reduce((total, value) => total + value, 0);
let comparisons = 0;

function close(actual, expected, label, tolerance = 1e-9) {
  assert.ok(Number.isFinite(actual), `${label}: nonfinite actual value`);
  assert.ok(Number.isFinite(expected), `${label}: nonfinite reference value`);
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${label}: ${actual} != ${expected}, error=${Math.abs(actual - expected)}`);
  comparisons++;
}

function arrayClose(actual, expected, label, tolerance = 1e-9) {
  assert.equal(actual.length, expected.length, `${label}: length`);
  for (let i = 0; i < expected.length; i++) {
    if (Array.isArray(expected[i])) arrayClose(actual[i], expected[i], `${label}[${i}]`, tolerance);
    else close(actual[i], expected[i], `${label}[${i}]`, tolerance);
  }
}

function centralDifference(evaluate, object, key, epsilon = 1e-5) {
  const before = object[key];
  let plus, minus;
  try {
    object[key] = before + epsilon;
    plus = evaluate();
    object[key] = before - epsilon;
    minus = evaluate();
  } finally {
    object[key] = before;
  }
  return (plus - minus) / (2 * epsilon);
}

// These objectives describe the tiny networks directly, without graph nodes or
// backward helpers from either engine.
const sigmoid = value => 1 / (1 + Math.exp(-value));
function networkObjective(s) {
  const h1 = sigmoid(s.x[0] * s.W0[0][0] + s.x[1] * s.W0[1][0]);
  const h2 = sigmoid(s.x[0] * s.W0[0][1] + s.x[1] * s.W0[1][1]);
  const y1 = sigmoid(h1 * s.W1[0][0] + h2 * s.W1[1][0]);
  const y2 = sigmoid(h1 * s.W1[0][1] + h2 * s.W1[1][1]);
  return ((y1 - s.d[0]) ** 2 + (y2 - s.d[1]) ** 2) / 2;
}
const neuronObjective = s => (sigmoid(s.w * s.x + s.b) - s.d) ** 2 / 2;

function checkScalarGraph(kind, state) {
  const graph = kind === 'network' ? F.networkGraph(state) : F.neuronGraph(state);
  const objective = kind === 'network' ? networkObjective : neuronObjective;
  const independentState = clone(graph.state);
  const stateBefore = clone(state);
  const report = F.finiteDifference(kind, state, 1e-5);
  const required = kind === 'network'
    ? ['x1', 'x2', 'w0_11', 'w0_12', 'w0_21', 'w0_22',
       'w1_11', 'w1_12', 'w1_21', 'w1_22', 'd1', 'd2']
    : ['x', 'w', 'b', 'd'];
  assert.deepEqual(report.rows.map(row => row.id).sort(), required.sort(), 'all differentiable leaves checked');
  assert.equal(report.checked, required.length);
  close(graph.result.loss, objective(independentState), `${kind}: independent forward`);
  let maximum = 0;
  for (const row of report.rows) {
    let container = independentState;
    for (const key of row.path.slice(0, -1)) container = container[key];
    const numeric = centralDifference(() => objective(independentState), container, row.path.at(-1));
    close(row.analytic, numeric, `${kind}: ${row.id}`, 2e-8);
    close(row.numeric, numeric, `${kind}: independent difference ${row.id}`, 2e-8);
    maximum = Math.max(maximum, Math.abs(row.analytic - numeric));
  }
  assert.deepEqual(state, stateBefore, 'finiteDifference preserves caller state');
  assert.equal(graph.reverseEvents.length, graph.edges.length, 'one reverse event per graph edge');
  const replayed = F.replay(graph);
  for (const node of graph.nodes) close(replayed[node.id], graph.adjoints[node.id], `replay ${node.id}`);
  console.log(`PASS scalar ${kind}: ${required.length} leaves, max |FD error|=${maximum.toExponential(2)}`);
}

checkScalarGraph('network', {
  x: [.6, .2], W0: [[.1, -.2], [.4, .3]], W1: [[.2, -.3], [-.4, .1]], d: [1, 0]
});
checkScalarGraph('network', {
  x: [-.37, .83], W0: [[.61, -.43], [-.22, .17]], W1: [[-.57, .34], [.76, -.29]], d: [.2, .8]
});
checkScalarGraph('neuron', {x: -.7, w: .8, b: -.15, d: .9});

// Fixed numerical statements used by the lesson's initial example.
const initial = F.networkGraph().result;
arrayClose(initial.a1, [.14, -.06], 'initial preactivations');
arrayClose(initial.y, [.4782604120989083, .4720336104959959], 'initial predictions');
close(initial.loss, .2475139635105433, 'initial squared loss');
close(initial.d2[0], -.1301883177944912, 'initial output derivative');

// Stable CE defined independently from probabilities; tiny probabilities may
// underflow to zero while the finite log-loss remains perfectly well defined.
function crossEntropy(logits, target, temperature = 1) {
  const scaled = logits.map(value => value / temperature);
  const maximum = Math.max(...scaled);
  const logNormalizer = Math.log(sum(scaled.map(value => Math.exp(value - maximum))));
  const targetVector = Array.isArray(target)
    ? target : logits.map((_, index) => Number(index === target));
  return -sum(targetVector.map((weight, index) => weight * (scaled[index] - maximum - logNormalizer)));
}

function checkSoftmax() {
  for (const temperature of [.2, .7, 1, 2.5]) {
    for (const logits of [[1000, 999, -1000], [-.7, 1.3, .2], [.4, .4, .4]]) {
      const probabilities = M.softmax(logits, temperature);
      close(sum(probabilities), 1, 'Softmax row sum', 2e-14);
      assert.ok(probabilities.every(value => value >= 0 && value <= 1));
      arrayClose(M.softmax(logits.map(value => value + 137), temperature), probabilities,
        'Softmax common-shift invariance', 1e-13);
    }
    for (const target of [1, [.2, .3, .5]]) {
      const logits = [-.7, 1.3, .2];
      const result = M.softmaxLossGrad(logits, target, temperature);
      close(result.loss, crossEntropy(logits, target, temperature), 'CE independent objective');
      for (let j = 0; j < logits.length; j++) {
        const targetWeight = Array.isArray(target) ? target[j] : Number(j === target);
        close(result.grad[j], (result.p[j] - targetWeight) / temperature, 'CE temperature factor');
        const numeric = centralDifference(() => crossEntropy(logits, target, temperature), logits, j);
        close(result.grad[j], numeric, 'CE finite difference', 2e-8);
      }
      close(sum(result.grad), 0, 'CE gradient row sum', 1e-13);
    }
  }
  const saturated = M.softmaxLossGrad([-4, 4], 0, .2);
  close(saturated.loss, 40, 'extreme temperature CE', 1e-12);
  for (let j = 0; j < 2; j++) {
    const logits = [-4, 4];
    const numeric = centralDifference(() => crossEntropy(logits, 0, .2), logits, j);
    close(saturated.grad[j], numeric, 'extreme CE finite difference', 2e-8);
  }
  close(M.softmaxLossGrad([-1000, 1000], 0).loss, 2000, 'underflow-safe CE');
  console.log('PASS Softmax: normalized rows, shift invariance, CE temperature/FD, extreme CE=40');
}
checkSoftmax();

function denseObjective(net, xs, ys) {
  return sum(xs.map((x, index) => {
    const output = net.forward(x).output;
    return net.task === 'classification'
      ? crossEntropy(output, ys[index]) : (output[0] - ys[index]) ** 2 / 2;
  })) / xs.length;
}

function checkDense(options, xs, ys) {
  const net = new M.DenseNet(options);
  const snapshot = net.params.map(parameter => parameter.data.slice());
  const analytic = net.lossGrad(xs, ys);
  close(analytic.loss, denseObjective(net, xs, ys), 'DenseNet batch objective');
  let maximum = 0, checked = 0;
  for (let p = 0; p < net.params.length; p++) {
    const parameter = net.params[p];
    for (let j = 0; j < parameter.data.length; j++) {
      const numeric = centralDifference(() => denseObjective(net, xs, ys), parameter.data, j);
      close(analytic.grads[p][j], numeric, `${parameter.name}[${j}]`, 2e-7);
      maximum = Math.max(maximum, Math.abs(analytic.grads[p][j] - numeric));
      checked++;
    }
  }
  assert.equal(checked, net.parameterCount(), 'all dense weights and biases checked');
  assert.deepEqual(net.params.map(parameter => parameter.data), snapshot, 'FD restores parameters');
  const lr = .01;
  net.step(analytic.grads, lr, 'sgd');
  for (let p = 0; p < net.params.length; p++) {
    arrayClose(net.params[p].data, snapshot[p].map((value, j) => value - lr * analytic.grads[p][j]),
      'SGD applies the previously computed complete gradient');
  }
  console.log(`PASS dense ${options.residual ? 'residual' : 'plain'} ${net.task} depth=${net.depth}: ` +
    `${checked} parameters, max |FD error|=${maximum.toExponential(2)}`);
}

checkDense({input: 2, width: 3, depth: 0, output: 3, seed: 11},
  [[.2, -.4], [-.7, .3], [.6, .8]], [2, 0, 1]);
checkDense({input: 2, width: 3, depth: 2, output: 2, activation: 'tanh', seed: 17},
  [[.2, -.4], [-.7, .3], [.6, .8]], [1, 0, 1]);
checkDense({input: 1, width: 3, depth: 2, task: 'regression', activation: 'tanh', seed: 23},
  [[-.6], [.15], [.8]], [-.3, .2, .7]);
checkDense({input: 1, width: 3, depth: 2, task: 'regression', residual: true,
  activation: 'tanh', seed: 29}, [[-.6], [.15], [.8]], [-.3, .2, .7]);
checkDense({input: 2, width: 3, depth: 1, output: 2, residual: true,
  activation: 'sigmoid', seed: 31}, [[.2, -.4], [-.7, .3], [.6, .8]], [1, 0, 1]);

// A hand-chosen linear residual block detects a missing identity path or a
// misplaced 0.3 scale, independently of a random finite-difference case.
const residual = new M.DenseNet({input: 2, width: 2, depth: 1, task: 'regression',
  activation: 'linear', residual: true, seed: 1});
const setParameter = (name, values) => {
  const parameter = residual.params.find(item => item.name === name);
  assert.ok(parameter, name);
  assert.equal(parameter.data.length, values.length);
  parameter.data = values.slice();
};
setParameter('W_in', [1, 0, 0, 1]); setParameter('b_in', [0, 0]);
setParameter('W_res0', [0, 0, 0, 0]); setParameter('b_res0', [0, 0]);
setParameter('W_out', [1, -2]); setParameter('b_out', [0]);
close(residual.predict([.2, -.4]), 1, 'zero residual preserves hidden input');
setParameter('W_res0', [.4, -.2, .1, .3]); setParameter('b_res0', [-.1, .2]);
close(residual.residualScale, .3, 'documented residual scale');
close(residual.predict([.2, -.4]), .958, 'hand-computed scaled residual output');
console.log('PASS residual fixture: identity route and 0.3 branch scale');

function checkOptimizerSteps() {
  const first = [.4, -.3], second = [-.2, .6], initial = [1, -.2], lr = .05;
  for (const optimizer of ['sgd', 'momentum', 'adam']) {
    const net = new M.DenseNet({input: 1, depth: 0, task: 'regression'});
    net.params.forEach((parameter, index) => { parameter.data[0] = initial[index]; });
    net.step(first.map(value => [value]), lr, optimizer);
    net.step(second.map(value => [value]), lr, optimizer);
    for (let index = 0; index < initial.length; index++) {
      const g1 = first[index], g2 = second[index];
      let totalDirection;
      if (optimizer === 'sgd') totalDirection = g1 + g2;
      else if (optimizer === 'momentum') totalDirection = 1.9 * g1 + g2;
      else {
        // Closed forms after one/two bias-corrected steps, not an update loop.
        const m2 = (.9 * g1 + g2) / 1.9;
        const v2 = (.999 * g1 * g1 + g2 * g2) / 1.999;
        totalDirection = g1 / (Math.abs(g1) + 1e-8) + m2 / (Math.sqrt(v2) + 1e-8);
      }
      close(net.params[index].data[0], initial[index] - lr * totalDirection,
        `${optimizer}: two-step closed form`, 2e-12);
    }
  }
  console.log('PASS optimizers: two-step SGD, momentum, bias-corrected Adam');
}
checkOptimizerSteps();

const image = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];
arrayClose(M.convolution(image, [[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
  [[-6, -6], [-6, -6]], 'known 3x3 windows');
arrayClose(M.convolution([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2], [3, 4]]),
  [[37, 47], [67, 77]], 'asymmetric kernel: cross-correlation, not flipped');
arrayClose(M.convolution([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 1], [1, 1]], 2, 1),
  [[1, 5], [11, 28]], 'stride 2 and zero padding');
console.log('PASS convolution: known windows, kernel orientation, stride/padding');

const batch = [[1, 3, 2], [2, 5, 4], [3, 7, 6], [4, 9, 8]];
const bn = M.batchNorm(batch);
arrayClose(bn.mean, [2.5, 6, 5], 'BN column means');
arrayClose(bn.variance, [1.25, 5, 5], 'BN population variances');
arrayClose(M.batchNorm(batch.map(row => row.map(value => value + 7))).normalized,
  bn.normalized, 'training BN translation invariance');
arrayClose(M.dropout([.2, -.5, 1], 0, 9).values, [.2, -.5, 1], 'zero-probability dropout');
console.log('PASS regularization: BN feature-axis statistics, inverted-dropout identity');

// The prose files are browser data declarations; inspect them without a DOM.
const declarations = {window: {}};
for (const filename of ['foundations-lessons.js', 'lessons.js']) {
  vm.runInNewContext(fs.readFileSync(path.join(interactive, filename), 'utf8'), declarations,
    {filename: `interactive/${filename}`});
}
const lessons = [...declarations.window.FOUNDATIONLESSONS, ...declarations.window.NNLESSONS];
assert.equal(lessons.length, 18, 'current interactive lesson count');
assert.equal(new Set(lessons.map(lesson => lesson.id)).size, lessons.length, 'unique lesson anchors');
for (const lesson of lessons) {
  assert.ok(lesson.body && lesson.task && lesson.quiz, `${lesson.id}: complete declaration`);
  assert.ok(Number.isInteger(lesson.quiz.correct) && lesson.quiz.correct >= 0 &&
    lesson.quiz.correct < lesson.quiz.options.length, `${lesson.id}: valid quiz answer index`);
}
console.log(`PASS declarations: ${lessons.length} lessons, unique anchors and valid quiz indices`);
console.log(`All interactive math checks passed (${comparisons} numerical comparisons).`);

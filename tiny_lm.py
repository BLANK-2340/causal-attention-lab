#!/usr/bin/env python3
"""Causal Attention Lab: an original, deliberately small CPU language model.

Repository: https://github.com/BLANK-2340/causal-attention-lab
Created public 2026-11-18 under BLANK-2340 (owner ID 105676435).

Quick start (Python 3.10+, NumPy):
    python -m pip install numpy
    python -W error tiny_lm.py test
    OPENBLAS_NUM_THREADS=1 python tiny_lm.py experiment
    OPENBLAS_NUM_THREADS=1 python tiny_lm.py generate --prompt 'the red robot'

Architecture: token + learned absolute position embeddings -> one single-head
scaled dot-product CAUSAL attention -> residual addition -> vocabulary logits.
All forward and backward equations are implemented here using NumPy float64;
there is no autodiff framework. This is NOT a full Transformer: no feed-forward
block, normalization, multiple heads, dropout or encoder. Masking includes the
current token; targets are shifted one token ahead. Loss is mean next-token
cross entropy over all batch/time positions, including EOS, excluding BOS as a
target. The vocabulary still permits BOS as an output, without suppressing it.
Training uses explicit Adam, fixed 600 updates / batch 32 / rate .015, no decay,
no gradient clipping, no validation/test-based stopping or hyperparameter search.

Question: can attention recover an earlier subject colour when the immediately
previous word ('stays') cannot distinguish it? Original grammar (256 sentences):
    <bos> the COLOR SUBJECT near the COLOR LANDMARK stays COLOR . <eos>
The final COLOR must repeat the subject colour, not the distractor colour.
Four colours x four subjects x four distractor colours x four landmarks.
A fixed permutation (seed 90210) separates 192 training / 64 held-out COMPLETE
sentences before model fitting. Vocabulary and grammar are specified a priori,
not learned from the held-out set. Each sentence has 11 next-token targets;
agreement is target index 8. Held-out combinations share words and grammar with
training, so this measures narrow compositional interpolation, not natural
language generalization. No private or external text/data/assets are used.

Comparisons: separately trained token+position-only ablation (attention branch
removed), and add-0.5 unigram/bigram distributions fitted ONLY on training target
counts. The ablation has unused Q/K/V parameters for paired initialization but
zero gradients for them; it cannot use preceding tokens. Same batch streams and
steps across neural variants, seeds 7/42/2026. Reports nats/token, perplexity,
token accuracy and subject-colour agreement accuracy. Sampling is seeded and
unconstrained; grammar failures and repetitions are possible and informative.
Exact floating-point bytes can differ across NumPy/BLAS versions; no GPU claims.

Attribution: causal scaled dot-product attention follows the mathematical idea
in Vaswani et al., Attention Is All You Need (2017), sections 3.1/3.2:
https://arxiv.org/abs/1706.03762 . This simplified original implementation is
not a reproduction of their architecture/results. No paper text/code/figures
are redistributed. NumPy is an external BSD-licensed dependency, not vendored:
https://numpy.org/doc/stable/license.html . Adam equations are implemented here;
reference Kingma/Ba (2014): https://arxiv.org/abs/1412.6980 .

Task continuity, 2026-11-18: fourth new November project; first implementation
scope is CPU model, independent gradient/causality/metric/split tests, fixed
paired experiment and CLI generation. No public browser demo, README, separate
license, checkpoint or CI yet. Do not call the repository fully complete until
reproducibility/results documentation, persistence/demo and CI are verified.
Next: add strict checkpoint round-trip, source-bound recorded experiment,
README/architecture/results/limitations, separate license, CI and small demo.
Fifth CUDA fusion project remains separate; GPU checks require real execution.

Initial CPU observations (Python 3.12.14 / NumPy 2.3.5), fixed protocol above:
attention held-out agreement 64/64 in each seed; token-only 14/64,13/64,13/64.
Attention held-out nats/token .538263/.539656/.530634; token-only
.661639/.661858/.642432. Bigram 1.004598 nats/token,13/64 agreement; unigram
2.609949 nats/token,0/64 agreement (its globally most frequent word is 'the').
These three initialization/batch seeds use ONE split, not three independent
corpora and not significance evidence. Attention has 1538 active parameters;
the ablation has 770 active parameters plus 768 unused stored Q/K/V entries.
Lower loss is not proof of interpretability or a competitive language model.
Validation in the initial environment: 11 warning-strict unittest groups passed,
including central finite differences for every one of 60 small-model parameters
in BOTH variants, a scalar-loop forward oracle, first-step Adam algebra,
future-token/prefix invariance, full-budget training, independently counted
baselines and held-out-token poisoning. Two fresh experiment subprocesses
produced identical JSON bytes; CLI generation was executed successfully.

MIT License
Copyright (c) 2026 BLANK-2340
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import itertools
import json
import math
import platform
import unittest

import numpy as np


COLORS = ("red", "blue", "green", "amber")
SUBJECTS = ("robot", "drone", "rover", "sensor")
LANDMARKS = ("lamp", "tower", "gate", "bridge")
VOCAB = ("<bos>", "<eos>", ".", "the", "near", "stays") + COLORS + SUBJECTS + LANDMARKS
TOKEN = {word: i for i, word in enumerate(VOCAB)}
CONTEXT = 11
STEPS = 600
SEEDS = (7, 42, 2026)


def corpus():
    """Disjoint full sentences; no overlapping token-window splitting."""
    sentences = [
        [TOKEN[w] for w in ("<bos>", "the", c, s, "near", "the", d, n,
                           "stays", c, ".", "<eos>")]
        for c, s, d, n in itertools.product(COLORS, SUBJECTS, COLORS, LANDMARKS)
    ]
    order = np.random.default_rng(90210).permutation(len(sentences))
    rows = np.asarray(sentences, dtype=np.int64)[order]
    return rows[:192].copy(), rows[192:].copy()


def softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def token_ids(value, vocab, context):
    ids = np.asarray(value)
    if (ids.ndim != 2 or ids.size == 0 or ids.shape[1] > context
            or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= vocab)):
        raise ValueError("expected a nonempty [batch,time] integer token array within bounds")
    return ids


class Model:
    """Single-head residual attention with a fully explicit backward pass."""

    def __init__(self, seed=7, dim=16, vocab=len(VOCAB), context=CONTEXT, attention=True):
        if any(type(n) is not int or n <= 0 for n in (dim, vocab, context)):
            raise ValueError("dimensions must be positive integers")
        if type(attention) is not bool:
            raise ValueError("attention must be boolean")
        self.dim, self.vocab, self.context, self.attention = dim, vocab, context, attention
        rng = np.random.default_rng(seed)
        self.p = {"embed": rng.normal(0, .2, (vocab, dim)),
                  "position": rng.normal(0, .2, (context, dim))}
        for name in ("query", "key", "value"):
            self.p[name] = rng.normal(0, 1 / math.sqrt(dim), (dim, dim))
        self.p["output"] = rng.normal(0, 1 / math.sqrt(dim), (dim, vocab))
        self.p["bias"] = np.zeros(vocab, dtype=np.float64)

    def forward(self, inputs):
        ids = token_ids(inputs, self.vocab, self.context)
        p, t = self.p, ids.shape[1]
        x = p["embed"][ids] + p["position"][:t]
        q, k, v = (x @ p[name] for name in ("query", "key", "value"))
        scores = q @ k.swapaxes(-1, -2) / math.sqrt(self.dim)
        mask = np.triu(np.ones((t, t), dtype=bool), 1)
        # -inf is intentional; at least the diagonal is always unmasked.
        a = softmax(np.where(mask, -np.inf, scores))
        h = x + a @ v if self.attention else x
        logits = h @ p["output"] + p["bias"]
        return logits, (ids, x, q, k, v, a, h)

    def loss(self, inputs, targets, backward=False):
        logits, cache = self.forward(inputs)
        y = token_ids(targets, self.vocab, self.context)
        if y.shape != logits.shape[:2]:
            raise ValueError("targets must have the same shape as inputs")
        z = logits - logits.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        loss = float(-np.take_along_axis(logp, y[..., None], axis=-1).mean())
        if not backward:
            return loss
        ids, x, q, k, v, a, h = cache
        dz = np.exp(logp)
        np.put_along_axis(dz, y[..., None],
                          np.take_along_axis(dz, y[..., None], axis=-1) - 1, axis=-1)
        dz /= y.size
        g = {name: np.zeros_like(p) for name, p in self.p.items()}
        g["output"] = h.reshape(-1, self.dim).T @ dz.reshape(-1, self.vocab)
        g["bias"] = dz.sum(axis=(0, 1))
        dh = dz @ self.p["output"].T
        dx = dh.copy()  # residual branch
        if self.attention:
            da = dh @ v.swapaxes(-1, -2)
            dv = a.swapaxes(-1, -2) @ dh
            ds = a * (da - (da * a).sum(axis=-1, keepdims=True))
            dq = ds @ k / math.sqrt(self.dim)
            dk = ds.swapaxes(-1, -2) @ q / math.sqrt(self.dim)
            for name, derivative in (("query", dq), ("key", dk), ("value", dv)):
                g[name] = x.reshape(-1, self.dim).T @ derivative.reshape(-1, self.dim)
                dx += derivative @ self.p[name].T
        # add.at accumulates repeated tokens instead of overwriting their gradients.
        np.add.at(g["embed"], ids, dx)
        g["position"][:ids.shape[1]] = dx.sum(axis=0)
        return loss, g


def fit(seed=7, attention=True, steps=STEPS):
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    train, _ = corpus()
    model = Model(seed=seed, attention=attention)
    rng = np.random.default_rng(seed + 10000)
    first = model.loss(train[:, :-1], train[:, 1:])
    moment = {name: np.zeros_like(p) for name, p in model.p.items()}
    variance = {name: np.zeros_like(p) for name, p in model.p.items()}
    for step in range(1, steps + 1):
        batch = train[rng.choice(len(train), size=32, replace=False)]
        _, gradients = model.loss(batch[:, :-1], batch[:, 1:], backward=True)
        for name, grad in gradients.items():
            if not np.isfinite(grad).all():
                raise FloatingPointError("nonfinite gradient")
            moment[name] = .9 * moment[name] + .1 * grad
            variance[name] = .999 * variance[name] + .001 * grad * grad
            corrected = moment[name] / (1 - .9 ** step)
            scale = np.sqrt(variance[name] / (1 - .999 ** step)) + 1e-8
            model.p[name] -= .015 * corrected / scale
    return model, first


def metrics(logp, targets):
    """Independent of the model and optimizer; caller supplies log probabilities."""
    logp, y = np.asarray(logp), np.asarray(targets)
    if (logp.ndim != 3 or y.shape != logp.shape[:2] or y.size == 0
            or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= logp.shape[-1])
            or not np.isfinite(logp).all()
            or not np.allclose(np.exp(logp).sum(axis=-1), 1, atol=1e-10, rtol=0)):
        raise ValueError("expected normalized finite log probabilities and valid targets")
    loss = float(-np.take_along_axis(logp, y[..., None], axis=-1).mean())
    pred = logp.argmax(axis=-1)
    result = {"nats_per_token": loss, "perplexity": math.exp(loss),
              "token_accuracy": float((pred == y).mean()), "target_count": int(y.size)}
    if y.shape[1] == CONTEXT:
        result["agreement_accuracy"] = float((pred[:, 8] == y[:, 8]).mean())
    return result


def evaluate(model, rows):
    logits, _ = model.forward(rows[:, :-1])
    z = logits - logits.max(axis=-1, keepdims=True)
    return metrics(z - np.log(np.exp(z).sum(axis=-1, keepdims=True)), rows[:, 1:])


def count_baselines(train, held):
    v = len(VOCAB)
    unigram = np.full(v, .5)
    bigram = np.full((v, v), .5)
    np.add.at(unigram, train[:, 1:].ravel(), 1)
    np.add.at(bigram, (train[:, :-1].ravel(), train[:, 1:].ravel()), 1)
    uni = np.broadcast_to(np.log(unigram / unigram.sum()), (len(held), CONTEXT, v))
    bi = np.log(bigram / bigram.sum(axis=1, keepdims=True))[held[:, :-1]]
    return {"unigram": metrics(uni, held[:, 1:]), "bigram": metrics(bi, held[:, 1:])}


def generate(model, prompt="the red robot", seed=123, temperature=.8, max_new=20):
    if (not math.isfinite(temperature) or temperature <= 0
            or type(max_new) is not int or not 0 <= max_new <= 100):
        raise ValueError("positive finite temperature and max_new in [0,100] required")
    words = prompt.split()
    if any(w not in TOKEN or w in ("<bos>", "<eos>") for w in words):
        raise ValueError("prompt must contain known words without BOS/EOS")
    ids = [TOKEN["<bos>"]] + [TOKEN[w] for w in words]
    if len(ids) > model.context:
        raise ValueError("prompt exceeds model context")
    rng = np.random.default_rng(seed)
    for _ in range(max_new):
        # Stop at learned context rather than inventing extrapolated positions.
        if len(ids) > model.context:
            break
        logits, _ = model.forward(np.array([ids], dtype=np.int64))
        next_id = int(rng.choice(model.vocab, p=softmax(logits[0, -1] / temperature)))
        ids.append(next_id)
        if next_id == TOKEN["<eos>"]:
            break
    return " ".join(VOCAB[i] for i in ids)


def experiment():
    train, held = corpus()
    runs = []
    for seed in SEEDS:
        for attention in (False, True):
            model, initial = fit(seed, attention)
            runs.append({"seed": seed, "attention": attention, "initial_train_nats": initial,
                         "train": evaluate(model, train), "held_out": evaluate(model, held),
                         "sample": generate(model),
                         "stored_parameters": sum(p.size for p in model.p.values()),
                         "active_parameters": sum(p.size for n, p in model.p.items()
                                                  if attention or n not in ("query", "key", "value"))})
    return {"scope": "original finite synthetic grammar, not competitive natural-language modeling",
            "python": platform.python_version(), "numpy": np.__version__,
            "train_sentences": len(train), "held_out_sentences": len(held),
            "split_seed": 90210, "steps": STEPS, "batch": 32, "rate": .015,
            "baseline_held_out": count_baselines(train, held), "runs": runs}


class Tests(unittest.TestCase):
    def test_forward_scalar_reference(self):
        model = Model(seed=29, dim=2, vocab=3, context=3)
        x = np.array([[1, 1, 2]])
        actual, _ = model.forward(x)
        p = model.p
        # Scalar loops do not share the batched matmul or vectorized softmax path.
        embeddings = [[float(p["embed"][int(token), d] + p["position"][t, d])
                       for d in range(2)] for t, token in enumerate(x[0])]
        projected = {}
        for name in ("query", "key", "value"):
            projected[name] = [[sum(row[i] * p[name][i, j] for i in range(2))
                                for j in range(2)] for row in embeddings]
        for t in range(3):
            scores = [sum(projected["query"][t][d] * projected["key"][s][d]
                          for d in range(2)) / math.sqrt(2) for s in range(t + 1)]
            weights = [math.exp(score - max(scores)) for score in scores]
            total = sum(weights)
            hidden = [embeddings[t][d] + sum(weights[s] / total * projected["value"][s][d]
                                            for s in range(t + 1)) for d in range(2)]
            for word in range(3):
                expected = sum(hidden[d] * p["output"][d, word] for d in range(2)) + p["bias"][word]
                self.assertAlmostEqual(actual[0, t, word], expected, delta=1e-14)

    def test_all_parameter_gradients(self):
        x = np.array([[1, 1, 2, 0], [2, 0, 1, 1]])
        y = np.array([[1, 2, 0, 2], [0, 1, 1, 2]])
        for attention in (False, True):
            model = Model(seed=13, dim=3, vocab=3, context=4, attention=attention)
            _, grad = model.loss(x, y, backward=True)
            for name, param in model.p.items():
                for index in np.ndindex(param.shape):
                    old, epsilon = param[index], 1e-5
                    param[index] = old + epsilon
                    plus = model.loss(x, y)
                    param[index] = old - epsilon
                    minus = model.loss(x, y)
                    param[index] = old
                    numeric = (plus - minus) / (2 * epsilon)
                    self.assertAlmostEqual(numeric, grad[name][index], delta=2e-8,
                                           msg=f"{attention} {name}{index}")

    def test_mask_and_prefix_invariance(self):
        model = Model()
        x = np.array([[0, 3, 6, 10, 4]])
        a, cache = model.forward(x)
        attention = cache[5][0]
        self.assertTrue(np.array_equal(np.triu(attention, 1), np.zeros((5, 5))))
        np.testing.assert_allclose(attention.sum(axis=-1), 1, atol=1e-15)
        self.assertEqual(attention[0, 0], 1)
        changed = x.copy()
        changed[0, 3:] = [9, 12]
        b, _ = model.forward(changed)
        np.testing.assert_array_equal(a[:, :3], b[:, :3])
        prefix, _ = model.forward(x[:, :3])
        np.testing.assert_allclose(a[:, :3], prefix, atol=1e-15, rtol=0)

    def test_loss_alignment_and_extreme_logits(self):
        model = Model(dim=2, vocab=3, context=2)
        for p in model.p.values():
            p.fill(0)
        x, y = np.array([[0, 1]]), np.array([[1, 2]])
        loss, grad = model.loss(x, y, backward=True)
        self.assertAlmostEqual(loss, math.log(3))
        np.testing.assert_allclose(grad["bias"], [1/3, -1/6, -1/6], atol=1e-15)
        model.p["bias"][:] = [1000, 0, -1000]
        loss, grad = model.loss(x, y, backward=True)
        self.assertEqual(loss, 1500)
        self.assertTrue(all(np.isfinite(g).all() for g in grad.values()))

    def test_original_split_and_next_token_protocol(self):
        train, held = corpus()
        a, b = set(map(tuple, train)), set(map(tuple, held))
        self.assertEqual((len(a), len(b), len(a | b), len(a & b)), (192, 64, 256, 0))
        for row in np.concatenate([train, held]):
            self.assertEqual((row[0], row[-1]), (TOKEN["<bos>"], TOKEN["<eos>"]))
            self.assertEqual(row[2], row[9])
            self.assertEqual(row[8], TOKEN["stays"])
            self.assertEqual(len(row[1:]), CONTEXT)
        train[0, 0] = 99
        self.assertNotEqual(corpus()[0][0, 0], 99)  # caller cannot mutate corpus

    def test_metrics_and_count_reference(self):
        probabilities = np.array([[[.8, .2], [.4, .6]]])
        actual = metrics(np.log(probabilities), np.array([[0, 0]]))
        self.assertAlmostEqual(actual["nats_per_token"], -math.log(.32) / 2)
        self.assertAlmostEqual(actual["perplexity"], 1 / math.sqrt(.32))
        self.assertEqual(actual["token_accuracy"], .5)
        train, held = corpus()
        measured = count_baselines(train, held)
        counts = {w: .5 for w in range(len(VOCAB))}
        pairs = {(a, b): .5 for a in counts for b in counts}
        for row in train:
            for a, b in zip(row[:-1], row[1:]):
                counts[int(b)] += 1
                pairs[int(a), int(b)] += 1
        uni_loss, bi_loss = 0., 0.
        for row in held:
            for a, b in zip(row[:-1], row[1:]):
                uni_loss -= math.log(counts[int(b)] / sum(counts.values()))
                bi_loss -= math.log(pairs[int(a), int(b)] /
                                    sum(pairs[int(a), c] for c in counts))
        self.assertAlmostEqual(measured["unigram"]["nats_per_token"], uni_loss / (64*11))
        self.assertAlmostEqual(measured["bigram"]["nats_per_token"], bi_loss / (64*11))

    def test_training_determinism_and_heldout_isolation(self):
        # Fitting's data path only selects corpus()[0]; alter ALL held-out tokens.
        original = globals()["corpus"]
        a, initial = fit(seed=19, steps=30)
        def poison():
            train, held = original()
            held.fill(TOKEN["red"])
            return train, held
        try:
            globals()["corpus"] = poison
            b, _ = fit(seed=19, steps=30)
        finally:
            globals()["corpus"] = original
        for name in a.p:
            np.testing.assert_array_equal(a.p[name], b.p[name])
        train, _ = corpus()
        self.assertLess(evaluate(a, train)["nats_per_token"], initial - .5)

    def test_adam_first_step(self):
        seed = 31
        model = Model(seed=seed)
        train, _ = corpus()
        indices = np.random.default_rng(seed + 10000).choice(192, size=32, replace=False)
        batch = train[indices]
        _, gradient = model.loss(batch[:, :-1], batch[:, 1:], backward=True)
        fitted, _ = fit(seed=seed, steps=1)
        for name, before in model.p.items():
            # At step 1, bias correction cancels (1-beta) in both moments.
            expected = before - .015 * gradient[name] / (np.abs(gradient[name]) + 1e-8)
            np.testing.assert_allclose(fitted.p[name], expected, atol=1e-15, rtol=0)

    def test_fixed_budget_end_to_end(self):
        train, held = corpus()
        model, initial = fit(seed=7)
        result = evaluate(model, held)
        self.assertLess(evaluate(model, train)["nats_per_token"], initial - 1)
        self.assertLess(result["nats_per_token"], count_baselines(train, held)["bigram"]["nats_per_token"])
        self.assertGreaterEqual(result["agreement_accuracy"], .9)
        self.assertEqual(result["target_count"], 704)
        sample = generate(model)
        self.assertEqual(sample.split()[0], "<bos>")
        self.assertTrue(all(word in TOKEN for word in sample.split()))
        self.assertLessEqual(len(sample.split()), CONTEXT + 1)

    def test_ablation_context_and_gradients(self):
        model = Model(attention=False)
        a, _ = model.forward(np.array([[0, 3, 6, 5]]))
        b, _ = model.forward(np.array([[7, 8, 9, 5]]))
        np.testing.assert_array_equal(a[:, -1], b[:, -1])
        _, gradients = model.loss(np.array([[0, 0]]), np.array([[0, 1]]), backward=True)
        for name in ("query", "key", "value"):
            self.assertTrue(np.array_equal(gradients[name], np.zeros_like(gradients[name])))

    def test_generation_and_invalid_inputs(self):
        model = Model()
        self.assertEqual(generate(model, seed=91), generate(model, seed=91))
        self.assertEqual(generate(model, max_new=0), "<bos> the red robot")
        for x in ([], [[-1]], [[len(VOCAB)]], [[1.5]], [1], [[True]], [[0]*12]):
            with self.assertRaises(ValueError):
                model.forward(x)
        with self.assertRaises(ValueError):
            model.loss([[0]], [[0, 1]])
        for kwargs in ({"temperature": 0}, {"temperature": float("nan")},
                       {"prompt": "unknown"}, {"prompt": "<eos>"}, {"max_new": -1}):
            with self.assertRaises(ValueError):
                generate(model, **kwargs)
        with self.assertRaises(ValueError):
            metrics(np.array([[[0., 0.]]]), np.array([[0]]))
        with self.assertRaises(ValueError):
            Model(dim=0)
        with self.assertRaises(ValueError):
            fit(steps=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("test", "experiment", "generate"))
    parser.add_argument("--prompt", default="the red robot")
    args = parser.parse_args()
    if args.command == "test":
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        raise SystemExit(not result.wasSuccessful())
    if args.command == "experiment":
        print(json.dumps(experiment(), indent=2, sort_keys=True, allow_nan=False))
    else:
        model, _ = fit()
        print(generate(model, prompt=args.prompt))


if __name__ == "__main__":
    main()

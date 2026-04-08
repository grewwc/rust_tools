#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("\r", "").split())


def extract_char_ngrams(text: str, min_n: int, max_n: int):
    padded = f"^{normalize_text(text)}$"
    chars = list(padded)
    features = Counter()
    for n in range(min_n, max_n + 1):
        if len(chars) < n:
            continue
        for i in range(len(chars) - n + 1):
            token = "".join(chars[i : i + n])
            if token.strip():
                features[token] += 1
    return features


def build_dataset(corpus: dict):
    cfg = corpus["feature_config"]
    min_n = int(cfg["char_ngram_min"])
    max_n = int(cfg["char_ngram_max"])
    labels = corpus["labels"]
    samples = corpus["samples"]

    doc_freq = Counter()
    docs = []
    for sample in samples:
        feats = extract_char_ngrams(sample["text"], min_n, max_n)
        docs.append({"label": sample["core"], "features": feats, "text": sample["text"]})
        for token in feats.keys():
            doc_freq[token] += 1

    vocab = [
        token
        for token, _ in sorted(doc_freq.items(), key=lambda item: (-item[1], item[0]))[
            : int(cfg["max_features"])
        ]
    ]
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    num_docs = len(docs)
    idf = {
        token: math.log((1.0 + num_docs) / (1.0 + doc_freq[token])) + 1.0 for token in vocab
    }

    xs = []
    ys = []
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for doc in docs:
        total = 0.0
        vec = {}
        for token, count in doc["features"].items():
            if token not in vocab_index:
                continue
            vec[token] = float(count)
            total += float(count)
        if total > 0.0:
            for token in list(vec.keys()):
                vec[token] = (vec[token] / total) * idf[token]
        xs.append(vec)
        ys.append(label_to_idx[doc["label"]])
    return labels, vocab, idf, xs, ys


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]


def train_softmax_regression(labels, vocab, xs, ys, epochs, learning_rate, l2):
    num_classes = len(labels)
    weights = [[0.0 for _ in labels] for _ in vocab]
    bias = [0.0 for _ in labels]
    order = list(range(len(xs)))

    for epoch in range(epochs):
        random.shuffle(order)
        for idx in order:
            vec = xs[idx]
            target = ys[idx]
            logits = bias[:]
            for token_idx, token in enumerate(vocab):
                value = vec.get(token)
                if not value:
                    continue
                w = weights[token_idx]
                for cls in range(num_classes):
                    logits[cls] += value * w[cls]

            probs = softmax(logits)
            for cls in range(num_classes):
                grad = probs[cls] - (1.0 if cls == target else 0.0)
                bias[cls] -= learning_rate * grad

            for token_idx, token in enumerate(vocab):
                value = vec.get(token)
                if not value:
                    continue
                for cls in range(num_classes):
                    grad = (probs[cls] - (1.0 if cls == target else 0.0)) * value
                    weights[token_idx][cls] -= learning_rate * (grad + l2 * weights[token_idx][cls])

        if epoch in {0, epochs // 2, epochs - 1}:
            acc = evaluate(labels, vocab, weights, bias, xs, ys)
            print(f"[epoch {epoch + 1}] train_acc={acc:.4f}")

    return weights, bias


def predict_one(labels, vocab, weights, bias, vec):
    logits = bias[:]
    for token_idx, token in enumerate(vocab):
        value = vec.get(token)
        if not value:
            continue
        for cls in range(len(labels)):
            logits[cls] += value * weights[token_idx][cls]
    probs = softmax(logits)
    best = max(range(len(labels)), key=lambda i: probs[i])
    return best


def evaluate(labels, vocab, weights, bias, xs, ys):
    correct = 0
    for vec, target in zip(xs, ys):
        if predict_one(labels, vocab, weights, bias, vec) == target:
            correct += 1
    return correct / max(len(xs), 1)


def build_model_json(corpus, labels, vocab, idf, weights, bias):
    features = []
    for token_idx, token in enumerate(vocab):
        features.append(
            {
                "token": token,
                "idf": idf[token],
                "weights": [round(x, 8) for x in weights[token_idx]],
            }
        )
    return {
        "version": corpus["version"],
        "labels": labels,
        "feature_config": {
            "char_ngram_min": corpus["feature_config"]["char_ngram_min"],
            "char_ngram_max": corpus["feature_config"]["char_ngram_max"],
        },
        "runtime_rules": corpus["runtime_rules"],
        "bias": [round(x, 8) for x in bias],
        "features": features,
    }


def main():
    parser = argparse.ArgumentParser(description="Train local TF-IDF + Logistic Regression intent model")
    parser.add_argument(
        "--corpus",
        default="config/intent/training_corpus.json",
        help="Path to training corpus json",
    )
    parser.add_argument(
        "--output",
        default="config/intent/intent_model.json",
        help="Path to output model json",
    )
    parser.add_argument("--epochs", type=int, default=220, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.55, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0.0002, help="L2 regularization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    labels, vocab, idf, xs, ys = build_dataset(corpus)
    weights, bias = train_softmax_regression(
        labels, vocab, xs, ys, args.epochs, args.learning_rate, args.l2
    )
    accuracy = evaluate(labels, vocab, weights, bias, xs, ys)
    print(f"[final] train_acc={accuracy:.4f}, features={len(vocab)}, samples={len(xs)}")

    model = build_model_json(corpus, labels, vocab, idf, weights, bias)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {output_path}")


if __name__ == "__main__":
    main()

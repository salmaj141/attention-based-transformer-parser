from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from chu_liu_edmonds import cle_min

Feature = Tuple[str, str, str]
Tree = Dict[int, int]  # child -> parent

ROOT_WORD = "ROOT"
ROOT_POS = "ROOT"


@dataclass
class SentenceExample:
    words: List[str]
    pos: List[str]
    gold_heads: Tree


def maybe_download_dependency_treebank() -> None:
    """Download the NLTK dependency treebank if it is missing."""
    import nltk

    try:
        from nltk.corpus import dependency_treebank

        _ = dependency_treebank.parsed_sents()[0]
    except LookupError:
        nltk.download("dependency_treebank")


def load_dependency_treebank(download: bool = True) -> Tuple[List[SentenceExample], List[SentenceExample]]:
    """Load NLTK dependency_treebank; last 10% of sentences are the test set."""
    if download:
        maybe_download_dependency_treebank()
    from nltk.corpus import dependency_treebank

    examples: List[SentenceExample] = []
    for graph in dependency_treebank.parsed_sents():
        addresses = sorted(idx for idx in graph.nodes if idx != 0)
        words, pos, heads = [], [], {}
        for new_idx, old_idx in enumerate(addresses, start=1):
            node = graph.nodes[old_idx]
            words.append(str(node.get("word", "")))
            pos.append(str(node.get("tag", "")))
            old_head = node.get("head")
            if old_head is None or old_head == 0:
                heads[new_idx] = 0
            else:
                heads[new_idx] = addresses.index(old_head) + 1
        examples.append(SentenceExample(words=words, pos=pos, gold_heads=heads))

    split = int(0.9 * len(examples))
    return examples[:split], examples[split:]


def node_word(sent: SentenceExample, idx: int) -> str:
    return ROOT_WORD if idx == 0 else sent.words[idx - 1]


def node_pos(sent: SentenceExample, idx: int) -> str:
    return ROOT_POS if idx == 0 else sent.pos[idx - 1]


def edge_features(sent: SentenceExample, head: int, dep: int) -> List[Feature]:
    """Boolean features for a candidate arc head -> dep."""
    return [
        ("WORD", node_word(sent, head), node_word(sent, dep)),
        ("POS", node_pos(sent, head), node_pos(sent, dep)),
    ]


def score_edge(weights: Dict[Feature, float], sent: SentenceExample, head: int, dep: int) -> float:
    return sum(weights.get(feat, 0.0) for feat in edge_features(sent, head, dep))


def tree_features(sent: SentenceExample, tree: Tree) -> Dict[Feature, int]:
    feats: Dict[Feature, int] = defaultdict(int)
    for dep, head in tree.items():
        for feat in edge_features(sent, head, dep):
            feats[feat] += 1
    return feats


def predict_mst(sent: SentenceExample, weights: Dict[Feature, float]) -> Tree:
    """Predict the highest-scoring tree; cle_min minimizes, so negate edge scores."""
    n = len(sent.words)
    scores = {}
    for head in range(0, n + 1):
        for dep in range(1, n + 1):
            if head == dep:
                continue
            scores[(head, dep)] = -score_edge(weights, sent, head, dep)
    return cle_min(scores, n + 1)


def _lazy_average_step(
    weights: Dict[Feature, float],
    totals: Dict[Feature, float],
    last_update: Dict[Feature, int],
    step: int,
) -> None:
    """Accumulate the current weights into the running average (lazy update)."""
    for feat, value in weights.items():
        totals[feat] += value * (step - last_update[feat])
        last_update[feat] = step


def train_averaged_perceptron(
    train_data: List[SentenceExample],
    iterations: int = 2,
    lr: float = 1.0,
    seed: int = 13,
) -> Dict[Feature, float]:
    """Averaged structured perceptron for MST parsing."""
    rng = random.Random(seed)
    weights: Dict[Feature, float] = {}
    totals: Dict[Feature, float] = defaultdict(float)
    last_update: Dict[Feature, int] = defaultdict(int)
    step = 0

    for _ in range(iterations):
        order = list(range(len(train_data)))
        rng.shuffle(order)
        for idx in order:
            step += 1
            sent = train_data[idx]
            _lazy_average_step(weights, totals, last_update, step)

            pred = predict_mst(sent, weights)
            gold = sent.gold_heads
            if pred != gold:
                gold_feats = tree_features(sent, gold)
                pred_feats = tree_features(sent, pred)
                for feat in set(gold_feats) | set(pred_feats):
                    delta = lr * (gold_feats.get(feat, 0) - pred_feats.get(feat, 0))
                    if delta:
                        weights[feat] = weights.get(feat, 0.0) + delta
                        if abs(weights[feat]) < 1e-12:
                            del weights[feat]

    if step == 0:
        return {}

    step += 1
    _lazy_average_step(weights, totals, last_update, step)
    return {feat: value / step for feat, value in totals.items() if abs(value) > 1e-12}


def evaluate_parser(data: List[SentenceExample], parse_fn) -> float:
    """Mean unlabeled attachment score (UAS) over all sentences."""
    total_correct = 0
    total_words = 0
    for sent in data:
        pred = parse_fn(sent)
        for dep, gold_head in sent.gold_heads.items():
            total_correct += int(pred.get(dep) == gold_head)
        total_words += len(sent.words)
    return total_correct / total_words if total_words else 0.0


def attn_to_arc_scores(attn_matrix: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Convert an attention matrix to scores for cle_min.

    attn_matrix[u, v] is the attention weight for arc u -> v (head u, dependent v).
    cle_min finds a minimum arborescence, so we negate the attention values.
    ROOT (index 0) may never be a dependent.
    """
    n_plus_1 = attn_matrix.shape[0]
    scores = {}
    for head in range(n_plus_1):
        for dep in range(1, n_plus_1):
            if head == dep:
                continue
            scores[(head, dep)] = -float(attn_matrix[head, dep])
    return scores


def parse_with_attention(
    sent: SentenceExample,
    tokenizer,
    model,
    layer: int,
    head_mode: str = "mean",
) -> Tree:
    from transformer_parser_utils import get_attention_matrix

    attn = get_attention_matrix(sent.words, tokenizer, model, layer=layer, head_mode=head_mode)
    scores = attn_to_arc_scores(attn)
    return cle_min(scores, len(sent.words) + 1)


def evaluate_attention_layers(
    test_data: List[SentenceExample],
    layers: Tuple[int, ...] = (0, 5, 11),
    head_mode: str = "mean",
) -> Dict[str, float]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
    model.eval()

    results = {}
    for layer in layers:
        score = evaluate_parser(
            test_data,
            lambda sent, layer=layer: parse_with_attention(sent, tokenizer, model, layer, head_mode),
        )
        results[f"layer_{layer}"] = score
    return results


def run_all() -> Dict[str, Any]:
    """Train/evaluate the perceptron MST parser and the BERT attention parser."""
    train, test = load_dependency_treebank(download=True)
    weights = train_averaged_perceptron(train, iterations=2, lr=1.0, seed=13)
    mst_uas = evaluate_parser(test, lambda sent: predict_mst(sent, weights))

    results: Dict[str, Any] = {
        "num_train_sentences": len(train),
        "num_test_sentences": len(test),
        "num_features_nonzero": len(weights),
        "mst_perceptron_uas": mst_uas,
    }

    attn_results = evaluate_attention_layers(test, layers=(0, 5, 11), head_mode="mean")
    results["attention"] = attn_results
    best_layer, best_val = max(attn_results.items(), key=lambda kv: kv[1])
    results["best_attention_setting"] = f"{best_layer}, head_mode=mean"
    results["best_attention_uas"] = best_val

    return results


def main() -> None:
    results = run_all()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

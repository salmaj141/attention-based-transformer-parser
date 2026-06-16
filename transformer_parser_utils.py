"""
transformer_parser_utils.py
===========================
Utility functions provided for NLP Exercise 4, Section 2.

You do NOT need to modify this file.

Provided function
-----------------
  get_attention_matrix(words, tokenizer, model, layer=-1, head_mode="mean")
      Extract a word-level attention matrix from a pretrained BERT model.

The Chu-Liu/Edmonds algorithm (cle_min) is in chu_liu_edmonds.py,
the same file used in Section 1.

Typical usage
-------------
  from transformers import AutoTokenizer, AutoModel
  from chu_liu_edmonds import cle_min
  from transformer_parser_utils import get_attention_matrix

  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
  model.eval()

  words  = ["John", "likes", "coffee"]
  attn   = get_attention_matrix(words, tokenizer, model)   # (4, 4) numpy array
  # attn[u, v] = attention weight for arc u -> v  (index 0 = ROOT)
"""

import numpy as np


def get_attention_matrix(words, tokenizer, model, layer=-1, head_mode="mean"):
    """
    Run BERT on `words` and return a word-level attention matrix.

    Parameters
    ----------
    words     : list[str]
        The sentence tokens (already split into words).
    tokenizer : BertTokenizerFast (or compatible)
        Loaded with AutoTokenizer.from_pretrained(...).
    model     : BertModel (or compatible) with output_attentions=True
        Loaded with AutoModel.from_pretrained(..., output_attentions=True).
        Must be in eval mode (model.eval()).
    layer     : int
        Which BERT layer's attention to use.  0 is the first layer,
        -1 (default) is the last layer.
    head_mode : str
        How to combine the multiple attention heads in the chosen layer:
          "first" -- use only the first head (head index 0)
          "mean"  -- average over all heads (default)
          "last"  -- use only the last head

    Returns
    -------
    numpy array, shape (n+1, n+1)
        Entry [u, v] is the attention weight for the arc u -> v
        (u = potential head, v = potential dependent).
        Index 0 is an artificial ROOT row; indices 1..n are the words.
        The ROOT row is constructed from the attention that the [CLS] token
        gives to each word, serving as a proxy for "rootness".
    """
    import torch

    n = len(words)

    # Tokenise, keeping a mapping from sub-token positions to original words.
    # is_split_into_words=True tells the tokenizer that `words` is already
    # a list of tokens, so it should not split on whitespace itself.
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
    )
    # word_ids(): for each position in the sub-token sequence, the index of
    # the original word it came from, or None for special tokens ([CLS]/[SEP]).
    word_ids = encoding.word_ids()

    with torch.no_grad():
        outputs = model(**encoding)

    # outputs.attentions is a tuple of length num_layers.
    # Each element has shape [batch=1, num_heads, seq_len, seq_len].
    attn_layer = outputs.attentions[layer][0]   # [num_heads, seq_len, seq_len]

    if head_mode == "first":
        attn = attn_layer[0].numpy()
    elif head_mode == "last":
        attn = attn_layer[-1].numpy()
    else:   # "mean"
        attn = attn_layer.mean(dim=0).numpy()   # [seq_len, seq_len]

    # Build a mapping from each original word index to the list of sub-token
    # positions that correspond to it.
    word_to_subtokens = {}
    for pos, wid in enumerate(word_ids):
        if wid is not None:
            word_to_subtokens.setdefault(wid, []).append(pos)

    # Construct the (n+1) x (n+1) word-level attention matrix.
    # mat[u, v] = average sub-token attention from word u to word v.
    # Indices are shifted by 1 (word 0 in `words` -> index 1 in mat).
    mat = np.zeros((n + 1, n + 1))
    for wi in range(n):
        for wj in range(n):
            if wi == wj:
                continue
            rows = word_to_subtokens.get(wi, [])
            cols = word_to_subtokens.get(wj, [])
            if rows and cols:
                mat[wi + 1, wj + 1] = attn[np.ix_(rows, cols)].mean()

    # ROOT row (index 0): use the attention that [CLS] (position 0 in the
    # sub-token sequence) gives to each word, as a proxy for how "root-like"
    # each word is.
    cls_pos = 0
    for wj in range(n):
        cols = word_to_subtokens.get(wj, [])
        if cols:
            mat[0, wj + 1] = attn[cls_pos, cols].mean()

    return mat

# attention-based-transformer-parser
Extracted and analyzed hidden structural attributes from pretrained BERT layers, leveraging self-attention weights directly as arc scores. Implemented the Chu-Liu-Edmonds (MST) graph algorithm for structural inference and developed an averaged Perceptron parser from scratch, optimizing feature extraction pipelines and algorithmic complexity

Running
-------
python ex4_solution.py

The script:
1. Loads dependency_treebank and splits the last 10% of sentences for testing.
2. Trains an averaged perceptron MST parser (word + POS bigram features,
   2 iterations, learning rate 1).
3. Evaluates mean UAS on the test set.
4. Evaluates BERT attention-based parsing for layers 0, 5, and 11
   (head_mode="mean").
5. Prints evaluation results (UAS) to the console.

Implementation notes
--------------------
Section 1 (MST Parser):
- Feature function: Boolean word-bigram and POS-bigram indicators.
- Inference: Chu-Liu/Edmonds via cle_min on negated edge scores.
- Learning: Averaged structured perceptron with random sentence order.

Section 2 (Attention-Based Parsing):
- attn_to_arc_scores converts attention matrix entries to cle_min scores.
- Attention weights are negated because cle_min finds a minimum arborescence.
- ROOT (index 0) is never allowed as a dependent.

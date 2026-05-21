# Self-Attention: From Input to Prediction

A from-scratch numpy implementation of the full transformer inference pipeline — no frameworks, no magic.

![](https://lilianweng.github.io/posts/2018-06-24-attention/sentence-example-attention.png)

## What This Is

`attention.py` walks through **every step** a language model takes to turn an input sentence into a next-word prediction:

1. **Embeddings** — words become vectors
2. **Self-Attention** — vectors become context-aware
3. **Feed-Forward + Activation** — nonlinear transformation amplifies signal
4. **Language Modeling Head** — projection into vocabulary space + softmax
5. **Loss Function** — measuring how wrong the prediction is
6. **RLHF / Reward Shaping** — closed-form policy update as a logit shift, leashed by `β`
7. **Temperature** — controlling confidence at inference time

Each step prints exactly what's happening so you can see the math in action.

## Run It

```bash
python attention.py
```

No dependencies beyond numpy.

---

## The Pipeline, Step by Step

### 1. Embeddings

Each word gets a vector in `d_model`-dimensional space. These are randomly initialized here — in a real model, they're learned during training and encode semantic relationships.

```python
embeddings = np.random.randn(vocab_size, d_model) * init_scale
X = np.array([embeddings[word_to_index[w]] for w in words])
```

The `init_scale` matters. Too small and every word looks the same — the entire system produces uniform noise. Training is the process of pushing these vectors apart until they carry meaning.

### 2. Self-Attention

The core mechanism from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). Each word looks at every other word and decides how much to attend to it.

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    context_aware = np.dot(attention_weights, V)
    return context_aware, attention_weights
```

The input embeddings are projected through learned matrices `W_Q`, `W_K`, `W_V` into Query, Key, and Value spaces:

- **Query**: "what am I looking for?"
- **Key**: "what do I contain?"
- **Value**: "what information do I carry?"

The dot product between Query and Key gives a similarity score. Softmax normalizes these into attention weights (each row sums to 1.0). The weighted sum of Values produces a new **context-aware embedding** for each position — a vector that encodes not just the word itself, but its relationship to every other word in the sequence.

```
Attention weights (each row sums to 1.0):
          the    cat    sat     on    the    mat
   the  0.243  0.112  0.165  0.136  0.243  0.101
   cat  0.062  0.197  0.367  0.117  0.062  0.194
   sat  0.106  0.179  0.237  0.151  0.106  0.219
    on  0.137  0.172  0.230  0.176  0.137  0.147
   the  0.243  0.112  0.165  0.136  0.243  0.101
   mat  0.049  0.294  0.238  0.153  0.049  0.216
```

Notice: "cat" attends most strongly to "sat" (0.367). "mat" attends most strongly to "cat" (0.294). These asymmetric relationships are what allow the model to understand structure.

> **The same operation powers RAG and in-context learning.** `Q · K^T` is a similarity search. When the keys are the prompt's own tokens, you get self-attention. When the keys are few-shot examples you pasted, you get in-context learning. When the keys are pre-computed embeddings of documents in a vector database, you get retrieval-augmented generation. The math is identical — only *whose* keys and values are eligible changes. A vector DB is just a `K` matrix too big to keep on the GPU. Full breakdown in [`blog.md`](./blog.md) Stage 2.

### 3. Feed-Forward Network with GeLU Activation

After attention, each position passes through a feed-forward network with a nonlinear activation function:

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def feed_forward(x, W1, b1, W2, b2):
    hidden = gelu(np.dot(x, W1) + b1)
    return np.dot(hidden, W2) + b2
```

GeLU (Gaussian Error Linear Unit) is the activation function used in GPT-2/3/4. It does something crucial: **it suppresses weak signals and amplifies strong ones**.

```
Before activation (sample from 'cat' position):
  Linear:  [ 0.627   0.0529 -0.729   0.2436] ...
  After GeLU: [ 0.4606  0.0276 -0.1699  0.1453] ...
```

Negative values get crushed toward zero. Positive values pass through (slightly reduced). This nonlinearity is why the model can learn complex patterns rather than just linear combinations — and why the relationship between training frequency and output probability is not a straight line.

### 4. Language Modeling Head (W_vocab + Softmax)

This is where a context vector becomes an actual word prediction. The feed-forward output gets projected from embedding space into **vocabulary space** — one number (logit) per word in the vocabulary:

```python
def predict_next_token(context_vector, W_vocab):
    logits = np.dot(context_vector, W_vocab)
    probs = softmax(logits, axis=-1)
    return logits, probs
```

We take the last token's representation (autoregressive: predict what comes next) and project it:

```
Token       Logit  Probability
------------------------------
the       -0.8443       0.0899
cat        0.7124       0.4264 <-- predicted
sat       -0.2205       0.1678
on         0.1315       0.2385
mat       -0.9943       0.0774
```

The logit difference between "cat" (0.71) and "on" (0.13) is only 0.58 — but after softmax, "cat" gets 1.8x the probability. **Softmax exponentially amplifies small differences.** This is why a modest training-data advantage compounds into a dominant recommendation.

### 5. Cross-Entropy Loss

During training, the loss function measures how wrong the prediction was:

```python
def cross_entropy_loss(probs, target_index):
    return -np.log(probs[target_index] + 1e-12)
```

If the training data says the next word should be "the" but the model assigned it only 9% probability:

```
Cross-entropy loss = -log(0.0899) = 2.4090
```

High loss means the model was confidently wrong. Backpropagation then adjusts **every weight in the entire network** — embeddings, W_Q, W_K, W_V, feed-forward weights, W_vocab — to make "the" more probable in this context next time.

### 6. Temperature

At inference time, temperature controls how sharp the probability distribution is:

```
T=0.5: predicted='cat', top_prob=0.647, entropy=1.020
T=1.0: predicted='cat', top_prob=0.426, entropy=1.419
T=2.0: predicted='cat', top_prob=0.307, entropy=1.559
```

Low temperature (0.5): the model is more decisive — its existing biases are amplified.
High temperature (2.0): the distribution flattens — less probable options get a chance.

Temperature divides the logits before softmax. It doesn't change what the model "thinks" — it changes how aggressively the winner takes all.

---

## Key Insight: Differentiation Is Everything

All weights in this demo are **randomly initialized**. The output is meaningless — the model has no knowledge. But it's not *uniformly* meaningless. Even random weights create some differentiation, and the pipeline (attention + activation + softmax) amplifies that differentiation into a decisive prediction.

Training is the process of replacing random differentiation with *meaningful* differentiation. Every gradient update moves the weights further from uniform, encoding patterns like "when the context mentions web development, PostgreSQL is more likely than CockroachDB."

If you initialize the weights too small (close to zero), the entire system collapses into uniformity — every word gets equal probability, attention is flat, temperature has no effect. The system is technically functional but produces nothing useful. **Signal requires asymmetry. Knowledge is differentiation.**

---

## The Architecture

```
Input Sentence
      │
      ▼
┌─────────────┐
│  Embeddings │  words → vectors (d_model-dimensional)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ W_Q, W_K,   │  project into Query, Key, Value spaces
│    W_V      │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│    Self-     │  Attention(Q,K,V) = softmax(QK^T/√d_k)V
│  Attention   │  → context-aware embeddings
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Feed-Forward │  GeLU activation: amplify signal, suppress noise
│   Network    │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│   W_vocab    │  project to vocabulary space → logits
│  (LM Head)  │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│   Softmax    │  logits → probability distribution
│ (÷ temperature)│
└─────┬───────┘
      │
      ▼
  Predicted Token (argmax)
```

In a real transformer, there are multiple attention heads, layer normalization, residual connections, and dozens of stacked layers. But the core path from input to prediction is what you see above — and what `attention.py` implements.

---

## Resources

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Self-Attention Explained](https://medium.com/@ramendrakumar/self-attention-d8196b9e9143)
- [Eugene Yan on Attention](https://eugeneyan.com/writing/attention/)
- [An illustration of next word prediction (BERT, GPT, XLNet)](https://ajay-arunachalam08.medium.com/an-illustration-of-next-word-prediction-with-state-of-the-art-network-architectures-like-bert-gpt-c0af02921f17)

import numpy as np

# =============================================================================
# FULL PIPELINE DEMO: From Self-Attention to Next-Word Prediction
#
# This script walks through the same stages described in blog.md, in
# data-flow order:
#   STEP 1. Embeddings                  (Stage 1 in blog)
#   STEP 2. Self-Attention              (Stage 2 in blog)
#   STEP 3. Feed-Forward + GeLU         (Stage 3 in blog)
#   STEP 4. Language Modeling Head      (Stage 4 in blog)
#   STEP 5. Cross-Entropy Loss          (Stage 5 in blog)
#   STEP 6. RLHF / Reward Shaping       (Stage 6 in blog)
#   STEP 7. Temperature Sampling        (Stage 7 in blog)
#
# STEP 6 demonstrates Reinforcement Learning from Human Feedback as the
# closed-form logit shift it actually is: new_logit = base_logit + R/β.
# SFT (the supervised half of Stage 6) is structurally identical to
# STEP 5's cross-entropy loss against a curated target token and is not
# separately demonstrated.
#
# Everything is from scratch with numpy — no frameworks, no magic.
# =============================================================================


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """Gaussian Error Linear Unit — the activation used in GPT-2/3/4."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def scaled_dot_product_attention(Q, K, V):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V"""
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    context_aware = np.dot(attention_weights, V)
    return context_aware, attention_weights


def feed_forward(x, W1, b1, W2, b2):
    """Position-wise feed-forward network with GeLU activation (Stage 3 in blog)."""
    hidden = gelu(np.dot(x, W1) + b1)
    return np.dot(hidden, W2) + b2


def predict_next_token(context_vector, W_vocab):
    """Language modeling head: project into vocab space, softmax (Stage 4 in blog)."""
    logits = np.dot(context_vector, W_vocab)
    probs = softmax(logits, axis=-1)
    return logits, probs


def cross_entropy_loss(probs, target_index):
    """Cross-entropy loss for a single target token (Stage 5 in blog)."""
    return -np.log(probs[target_index] + 1e-12)


# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    sentence = "the cat sat on the mat"
    words = sentence.split()
    unique_words = list(dict.fromkeys(words))
    vocab_size = len(unique_words)

    word_to_index = {w: i for i, w in enumerate(unique_words)}
    index_to_word = {i: w for w, i in word_to_index.items()}

    # --- Hyperparameters ---
    d_model = 8       # embedding dimension
    d_k = 6           # query/key/value dimension
    d_ff = 16         # feed-forward hidden dimension
    init_scale = 0.5  # weight initialization scale (larger = more differentiation)

    # =========================================================================
    # STEP 1: EMBEDDINGS (Blog Stage 1)
    # =========================================================================
    print("=" * 60)
    print("STEP 1: EMBEDDINGS")
    print("=" * 60)
    embeddings = np.random.randn(vocab_size, d_model) * init_scale
    X = np.array([embeddings[word_to_index[w]] for w in words])
    print(f"Sentence: '{sentence}'")
    print(f"Vocabulary: {unique_words} ({vocab_size} tokens)")
    print(f"Embedding matrix shape: ({vocab_size}, {d_model})")
    print(f"Input sequence shape: {X.shape}  (6 tokens x {d_model}-dim)")
    print()

    # =========================================================================
    # STEP 2: SELF-ATTENTION (Blog Stage 2)
    # =========================================================================
    print("=" * 60)
    print("STEP 2: SELF-ATTENTION")
    print("=" * 60)

    W_Q = np.random.randn(d_model, d_k) * init_scale
    W_K = np.random.randn(d_model, d_k) * init_scale
    W_V = np.random.randn(d_model, d_k) * init_scale

    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)

    context_aware, attn_weights = scaled_dot_product_attention(Q, K, V)

    print("Attention weights (each row sums to 1.0):")
    print(f"{'':>6} " + " ".join(f"{w:>6}" for w in words))
    for i, w in enumerate(words):
        row = " ".join(f"{v:6.3f}" for v in attn_weights[i])
        print(f"{w:>6} {row}")
    print(f"\nContext-aware embeddings shape: {context_aware.shape}")
    print()

    # =========================================================================
    # STEP 3: FEED-FORWARD + ACTIVATION (Blog Stage 3)
    # =========================================================================
    print("=" * 60)
    print("STEP 3: FEED-FORWARD NETWORK WITH GeLU ACTIVATION")
    print("=" * 60)

    W1 = np.random.randn(d_k, d_ff) * init_scale
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_k) * init_scale
    b2 = np.zeros(d_k)

    ff_output = feed_forward(context_aware, W1, b1, W2, b2)

    print(f"Feed-forward hidden dim: {d_ff}")
    print(f"Activation function: GeLU (nonlinear amplifier)")
    print(f"Output shape: {ff_output.shape}")
    print()
    print("Before activation (sample from 'cat' position):")
    raw = np.dot(context_aware[1], W1) + b1
    print(f"  Linear:  {raw[:4].round(4)} ...")
    print(f"  After GeLU: {gelu(raw)[:4].round(4)} ...")
    print("  Notice: negative values get suppressed, positive amplified.")
    print()

    # =========================================================================
    # STEP 4: LANGUAGE MODELING HEAD (Blog Stage 4)
    # =========================================================================
    print("=" * 60)
    print("STEP 4: LANGUAGE MODELING HEAD (W_vocab projection + softmax)")
    print("=" * 60)

    # Use the last token position (autoregressive: predict what comes after "mat")
    last_token_repr = ff_output[-1]  # shape: (d_k,)

    W_vocab = np.random.randn(d_k, vocab_size) * init_scale
    logits, probs = predict_next_token(last_token_repr, W_vocab)

    predicted_index = int(np.argmax(probs))
    predicted_word = index_to_word[predicted_index]

    print(f"Predicting what comes after: '{words[-1]}'")
    print(f"W_vocab shape: ({d_k}, {vocab_size}) — projects from embedding space to vocabulary")
    print()
    print(f"{'Token':<8} {'Logit':>8} {'Probability':>12}")
    print("-" * 30)
    for i in range(vocab_size):
        marker = " <-- predicted" if i == predicted_index else ""
        print(f"{index_to_word[i]:<8} {logits[i]:>8.4f} {probs[i]:>12.4f}{marker}")
    print()
    print(f"Predicted next token: '{predicted_word}' (greedy argmax)")
    print()

    # Show softmax amplification
    print("--- Softmax amplification demo ---")
    top = np.max(logits)
    second = np.sort(logits)[-2]
    print(f"Top logit: {top:.4f}, Runner-up: {second:.4f}, Diff: {top - second:.4f}")
    print(f"But after softmax: top prob is {np.max(probs)/np.sort(probs)[-2]:.1f}x the runner-up")
    print("Small logit differences become large probability gaps.")
    print()

    # =========================================================================
    # STEP 5: LOSS FUNCTION (Blog Stage 5)
    # =========================================================================
    print("=" * 60)
    print("STEP 5: CROSS-ENTROPY LOSS")
    print("=" * 60)

    # Pretend the "correct" next token is "the" (as if training data said so)
    target_word = "the"
    target_idx = word_to_index[target_word]
    loss = cross_entropy_loss(probs, target_idx)

    print(f"Suppose training data says the correct next token is: '{target_word}'")
    print(f"Model assigned P('{target_word}') = {probs[target_idx]:.4f}")
    print(f"Cross-entropy loss = -log({probs[target_idx]:.4f}) = {loss:.4f}")
    print()
    print("If this loss is high, backpropagation would adjust ALL weights")
    print("(embeddings, W_Q, W_K, W_V, W1, W2, W_vocab) to make")
    print(f"'{target_word}' more probable in this context next time.")
    print()

    # =========================================================================
    # STEP 6: RLHF / REWARD SHAPING (Blog Stage 6)
    # =========================================================================
    #
    # The KL-regularized RLHF objective
    #     max_pi   E_y[ R(y) ]  -  beta * KL( pi(y) || pi_ref(y) )
    # has a closed-form optimum:
    #     pi*(y)  proportional to  pi_ref(y) * exp( R(y) / beta )
    # In log-space, that is just an additive shift on the base logits:
    #     new_logit_y = base_logit_y + R(y) / beta
    # i.e. a reward model is mechanically a learned logit bias, leashed by 1/beta.
    # =========================================================================
    print("=" * 60)
    print("STEP 6: RLHF — REWARD SHAPING AS A LOGIT SHIFT")
    print("=" * 60)
    print("Base (pretraining) distribution from STEP 4:")
    for i in range(vocab_size):
        print(f"  {index_to_word[i]:<6}  base_logit = {logits[i]:>7.4f}   P = {probs[i]:.4f}")
    print()

    # Pretend a varied preference dataset was collected.
    # Annotators consistently preferred completions ending in an action verb
    # ('sat') and consistently disliked the noun 'cat'. After training a reward
    # model R(y) on those pairs, it assigns the following scalar rewards:
    reward = np.zeros(vocab_size)
    reward[word_to_index["sat"]] = +3.0   # strongly preferred
    reward[word_to_index["cat"]] = -2.0   # mildly disliked

    print("Reward model R(y) trained on annotator preferences:")
    for i in range(vocab_size):
        sign = "+" if reward[i] >= 0 else ""
        print(f"  {index_to_word[i]:<6}  R = {sign}{reward[i]:.2f}")
    print()

    # Sweep the KL-leash coefficient beta. Smaller beta = looser leash =
    # reward dominates the prior. Larger beta = tighter leash = prior wins.
    print("KL-leash sweep — new_logit = base_logit + R / beta:")
    header = f"  {'beta':>5}  {'leash':<11}  {'top':<6}  {'P(sat)':>8}  {'P(cat)':>8}  {'KL(pi||ref)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for beta in [10.0, 1.0, 0.3, 0.1]:
        rl_logits = logits + reward / beta
        rl_probs = softmax(rl_logits, axis=-1)
        top_idx = int(np.argmax(rl_probs))
        kl = float(np.sum(rl_probs * np.log((rl_probs + 1e-12) / (probs + 1e-12))))
        if beta >= 5:
            leash = "very tight"
        elif beta >= 1:
            leash = "tight"
        elif beta >= 0.3:
            leash = "loose"
        else:
            leash = "slack"
        print(
            f"  {beta:>5.2f}  {leash:<11}  {index_to_word[top_idx]:<6}  "
            f"{rl_probs[word_to_index['sat']]:>8.4f}  "
            f"{rl_probs[word_to_index['cat']]:>8.4f}  {kl:>12.4f}"
        )
    print()
    print("Observations:")
    print("  beta=10:  KL leash is tight; reward barely moves the distribution.")
    print("  beta=1:   reward wins enough to flip the top token from 'cat' to 'sat'.")
    print("  beta=0.1: leash slack; policy collapses onto the reward's favorite.")
    print()
    print("RLHF cannot manufacture tokens from nothing. The reward enters as an")
    print("additive shift on the base logit — so a token with logit ~ -infinity")
    print("(never seen in pretraining) cannot be reached for any finite reward.")
    print("RLHF redistributes probability mass within the support of the prior;")
    print("it does not expand the support.")
    print()

    # =========================================================================
    # STEP 7: TEMPERATURE DEMO (Blog Stage 7 — inference-time controls)
    # =========================================================================
    print("=" * 60)
    print("STEP 7: TEMPERATURE EFFECT ON PREDICTIONS")
    print("=" * 60)

    for temp in [0.5, 1.0, 2.0]:
        scaled_logits = logits / temp
        temp_probs = softmax(scaled_logits, axis=-1)
        top_idx = int(np.argmax(temp_probs))
        entropy = -np.sum(temp_probs * np.log(temp_probs + 1e-12))
        print(f"  T={temp:.1f}: predicted='{index_to_word[top_idx]}', "
              f"top_prob={np.max(temp_probs):.3f}, entropy={entropy:.3f}")

    print()
    print("Lower temperature -> sharper distribution (stronger bias)")
    print("Higher temperature -> flatter distribution (more diversity)")
    print()
    print("=" * 60)
    print("COMPLETE: This is the full path from input to prediction.")
    print("Pretraining (STEPS 1-5) shapes the base prior. RLHF (STEP 6)")
    print("re-weights that prior via a closed-form logit shift, leashed by")
    print("beta. Temperature (STEP 7) is the final inference-time tilt.")
    print("=" * 60)

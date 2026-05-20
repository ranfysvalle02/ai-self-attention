import numpy as np

def softmax(x, axis=-1):
    """
    Computes the softmax function along a specified axis, with numerical stability.
    """
    # Subtract max for numerical stability (prevents overflow)
    # keepdims=True ensures the broadcast aligns correctly across the matrix
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # Normalize so all probabilities sum to 1 along the specified axis
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def create_word_representations(sentence, d_model=4):
    """
    Creates a simple vocabulary and random embeddings for a given sentence.
    d_model is the dimensionality of our word embeddings.
    """
    words = sentence.split()
    unique_words = list(dict.fromkeys(words)) # Preserve order, remove duplicates
    
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    
    # Initialize random embeddings for our vocabulary
    vocab_size = len(unique_words)
    np.random.seed(42) # Seeded for reproducible output
    embeddings = np.random.rand(vocab_size, d_model)
    
    # Map the specific sequence of words to their embeddings
    sequence_embeddings = np.array([embeddings[word_to_index[w]] for w in words])
    
    return sequence_embeddings, words

def scaled_dot_product_attention(Q, K, V):
    """
    Calculates the true self-attention mechanism: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    # d_k is the dimension of the keys
    d_k = K.shape[-1]
    
    # 1. Calculate the dot product between Queries and Keys (the "scores")
    # Q is (N x d_k), K.T is (d_k x N) -> scores is (N x N)
    scores = np.dot(Q, K.T)
    
    # 2. Scale the scores by the square root of d_k
    scaled_scores = scores / np.sqrt(d_k)
    
    # 3. Apply softmax to get attention weights (probabilities)
    attention_weights = softmax(scaled_scores, axis=-1)
    
    # 4. Multiply the attention weights by the Values matrix
    # weights is (N x N), V is (N x d_v) -> context_aware_embeddings is (N x d_v)
    context_aware_embeddings = np.dot(attention_weights, V)
    
    return context_aware_embeddings, attention_weights

if __name__ == "__main__":
    sentence = "the cat sat on the mat"
    d_model = 4 # Dimension of our input embeddings
    d_k = 3     # Dimension of our Query/Key/Value vectors
    
    print(f"--- 1. INPUT PROCESSING ---")
    print(f"Sentence: '{sentence}'")
    
    # Get the raw embeddings for the N words in our sequence
    # X shape: (N, d_model) where N=6, d_model=4
    X, words = create_word_representations(sentence, d_model=d_model)
    N = len(words)
    print(f"Sequence length (N): {N} tokens")
    print(f"Input Embeddings (X) shape: {X.shape}\n")
    
    print(f"--- 2. LINEAR PROJECTIONS ---")
    # In a real neural network, these W matrices are the weights the model LEARNS.
    # We initialize them randomly here.
    np.random.seed(42)
    W_Q = np.random.rand(d_model, d_k)
    W_K = np.random.rand(d_model, d_k)
    W_V = np.random.rand(d_model, d_k)
    
    # Project input embeddings into Query, Key, and Value spaces
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)
    
    print(f"Queries (Q) shape: {Q.shape}")
    print(f"Keys (K) shape:    {K.shape}")
    print(f"Values (V) shape:  {V.shape}\n")
    
    print(f"--- 3. CALCULATING SELF-ATTENTION ---")
    context_aware_embeddings, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print("Attention Weights (N x N matrix):")
    print("How much each word attends to every other word (Rows sum to 1.0)")
    
    # Print a formatted table of attention weights
    print(f"{'':>6} " + " ".join([f"{w:>6}" for w in words]))
    for i, row_word in enumerate(words):
        row_str = " ".join([f"{val:6.3f}" for val in attention_weights[i]])
        print(f"{row_word:>6} {row_str}")
        
    print(f"\n--- 4. FINAL OUTPUT ---")
    print(f"Context-Aware Embeddings shape: {context_aware_embeddings.shape}")
    print("Notice how each token's original vector has been replaced by a new vector")
    print("that is a weighted sum of ALL the Values in the sequence.")
    
    # Show the final vector for the word "cat"
    cat_index = words.index("cat")
    print(f"\nOriginal embedding for 'cat': {X[cat_index].round(3)}")
    print(f"New context-aware embedding for 'cat': {context_aware_embeddings[cat_index].round(3)}")
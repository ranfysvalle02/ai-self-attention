### **Basic Word Prediction with Self-Attention**

![](https://lilianweng.github.io/posts/2018-06-24-attention/sentence-example-attention.png)


Imagine you're reading a long novel. You don't read every word with the same level of focus. Instead, you pay more attention to certain parts, like the plot twists or character developments. This is similar to how attention works in a language model.

The model presented in this repository is a type of **language model** that predicts the next word with a **self-attention mechanism**. 

* **Self-Attention:** The self-attention mechanism allows the model to focus on different parts of the input sequence based on their relevance to the current output. This is achieved by assigning weights to each input element, with larger weights indicating greater importance.

**How Does Self-Attention Work?**

1. **Embedding:** Each word in the input sequence is converted into a numerical representation called an embedding. 
2. **Query, Key, and Value:** For each word, three vectors are calculated: a query, a key, and a value.
3. **Attention Scores:** The dot product between the query of one word and the keys of all other words is calculated. This gives a score representing the similarity between the words.
4. **Softmax:** The scores are normalized using the softmax function to obtain attention weights.
5. **Context Vector:** The weighted sum of the value vectors, using the attention weights, creates a context vector. This context vector captures the relevant information from the entire sequence.

**The Code in Action**

* **Word Embeddings:** The code initializes random word embeddings. In practice, pre-trained embeddings like Word2Vec or GloVe can be used for better performance.
* **Context Window:** The code defines a context window, specifying the number of words to consider before the current word.
* **Self-Attention:** The code calculates attention scores, applies softmax, and creates the context vector.
* **Prediction:** The context vector is used to predict the next word, often using a simple linear layer or a more complex model.

![](https://www.mdpi.com/applsci/applsci-12-03846/article_deploy/html/images/applsci-12-03846-g006-550.jpg)

_(Image Credit to article[ Attention Map-Guided Visual Explanations for Deep Neural Networks ](https://www.mdpi.com/2076-3417/12/8/3846) )_

# **We should pay more attention to attention**

It all started with: ["Attention is all you need."](https://arxiv.org/abs/1706.03762)

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*bWhofmIsEaplOkav6FjqpA.png)

In the above image/example, there are 7 sequences in the sentence ‘the train left the station on time’, and we can see a 7x7 attention score matrix.

According to the self-attention scores depicted in the picture, the word ‘train’ pays more attention to the word ‘station’ rather than other words in consideration, such as ‘on’ or ‘the’. Alternatively, we can say the word ‘station’ pays more attention to the word ‘train’ rather than other words in consideration, such as ‘on’ or ‘the’.

[Read more here: Self-Attention](https://medium.com/@ramendrakumar/self-attention-d8196b9e9143)

Attention is a mechanism that allows a language model to focus on different parts of its input sequence based on their relevance to the current output. This is achieved by assigning weights to each input element, with larger weights indicating greater importance. These weights are calculated using a similarity metric, such as the dot product, between the query vector and each key vector in the input sequence.

For instance, in translation, attention helps the model concentrate on words or phrases that are semantically connected, leading to more accurate translations. On the other hand, this same mechanism can be exploited to generate misleading or biased text by directing the model's focus towards specific information.

**Attention and Quality:**

* **Positive impact:** Attention allows LLMs to focus on the most relevant parts of the input sequence when generating a response. This leads to responses that are more coherent, relevant, and grammatically correct.
* **Negative impact:**  
    * **Focus on misleading information:**  If the input contains misleading or irrelevant keywords, the LLM's attention might be drawn to them, resulting in inaccurate, nonsensical or other undesired responses.
    * **Missing key information:**  The LLM might overlook crucial information if the wording is different from what it's trained on. 


## Introduction

In this guide, we will explore the concept of attention through a Python code snippet that uses the self-attention mechanism to predict the next word in a sentence.

We will implement a basic language model that uses **self-attention** to predict the next word in a sentence. 

The core functionality relies on the self-attention mechanism.

## A Deeper Dive into the Process with Examples

### 1. Word Representations: A Visual Analogy

Imagine you're trying to teach a computer about the English language. You start by assigning each word a unique numerical identifier. This is similar to how we create word representations. 

* **Word:** "cat"
* **Representation:** [0.2, 0.5, -0.3]

These numbers, or embeddings, are randomly initialized. Over time, as the model learns from data, these embeddings will adjust to better represent the meaning and context of the words.

### 2. Self-Attention: Focusing on the Right Words

Imagine you're reading a sentence: "The quick brown fox jumps over the lazy dog." To understand the meaning, you focus on certain words more than others. Self-attention mimics this human intuition.

* **Query:** "jumps" (the word we're trying to understand)
* **Keys:** "The," "quick," "brown," "fox," "over," "the," "lazy," "dog" (all the words in the sentence)
* **Values:** The embeddings of these words

The model calculates a similarity score between the query and each key. This score represents how relevant each word is to understanding "jumps." For instance, "fox" and "jumps" might have a high similarity score.

### 3. Attention Weights and Probabilities

The similarity scores are converted into probabilities using the softmax function. This ensures that the probabilities sum to 1.

* **Similarity Scores:** [0.2, 0.5, 0.1, 0.6, 0.3, 0.2, 0.4, 0.1]
* **Probabilities:** [0.12, 0.28, 0.06, 0.32, 0.17, 0.12, 0.23, 0.06]

These probabilities indicate the importance of each word in the context of understanding "jumps."

### 4. Predicting the Next Word

The model first uses the attention probabilities to build a single **context vector** — a weighted sum of the word embeddings that summarizes what the model "knows" about the current position. That context vector then gets compared against the model's entire vocabulary (via a learned projection layer) to choose the actual next word.

* **Weighted Sum (Context Vector):** [0.12 * "The" + 0.28 * "quick" + ...] → a single `d_model`-dimensional vector summarizing the context.
* **Vocabulary Projection:** The context vector is multiplied by a learned matrix `W_vocab` (the language modeling head), producing one **logit** for every word in the vocabulary. A softmax over those logits gives a real probability distribution over all possible next words.
* **Predicted Next Word:** "over" (the word in the vocabulary with the highest projected score — *not* the word with the highest attention weight).

> Heads up: the weighted sum alone does **not** name a word — it just produces a vector. The vocabulary projection step is what turns that vector into an actual next-word prediction. We will revisit this distinction in the "Common Pitfall" callout further down.

### Self-Attention and Word Embeddings

* **Self-Attention**: This is the key concept used in the `calculate_self_attention` function. It allows the model to focus on relevant parts of the input sequence (the sentence) when predicting the next word.

* **Word Embeddings**: The model uses randomly generated embeddings to represent each word. These embeddings are then projected into query, key, and value vectors which are used for calculating the attention weights.

**Overall, the model can be considered a simple language model with a self-attention mechanism for next word prediction.** It demonstrates the core idea of self-attention but lacks the complexity of more advanced models like Transformers, which utilize this mechanism extensively.

```python
import numpy as np
```

We start by importing the numpy library, which provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## The Softmax Function

The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities. If one of the inputs is small or large, the softmax function squashes it, which helps in mitigating the exploding and vanishing gradient problems.

```python
def softmax(x):
  """
  This softmax function is often used in machine learning and deep learning to convert 
  a vector of real numbers into a probability distribution. 
  Each output value is between 0 and 1 (inclusive), and the sum of all output values is 1. 
  """
  # Subtract the max value in the input array from all elements for numerical stability.
  # This ensures that all values in the array are between 0 and 1, which helps prevent potential overflow or underflow issues.
  x -= np.max(x)

  # Apply the exponential function to each element in the array.
  # This transforms each value in the array into a positive value.
  exp_x = np.exp(x)

  # Divide each element in the array by the sum of all elements in the array.
  # This normalizes the values so that they all add up to 1, which is a requirement for a probability distribution.
  softmax_x = exp_x / np.sum(exp_x)

  # Return the resulting array, which represents a probability distribution over the input array.
  return softmax_x
```

## Creating Word Representations

The `create_word_representations` function takes a list of sentences as input and creates a dictionary mapping words to indices and vice versa. It also creates a list of word embeddings, which are randomly initialized.

```python
def create_word_representations(sentences):
    word_to_index = {}
    index_to_word = {}
    word_embeddings = []

    for sentence in sentences:
        for word in sentence.split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word[len(index_to_word)] = word
                word_embeddings.append(np.random.rand(3))  # Random embeddings

    return np.array(word_embeddings), word_to_index, index_to_word
```

## The Impact of Randomly Generated Embeddings

**Randomly generated embeddings** serve as a starting point for the model to learn meaningful representations of words. They are essentially arbitrary numerical vectors assigned to each word.

* **Initialization:** Random embeddings provide a starting point for the model to learn meaningful representations of words. Without them, the model wouldn't know where to begin and its outputs would likely be nonsensical.
* **Exploration:** Randomness encourages the model to explore different directions in the solution space, potentially leading to better performance as it learns from the data.

**Limitations of Random Embeddings:**

* **Arbitrary Starting Point:** Random embeddings are essentially random guesses about how words should be represented. They may not capture any inherent relationships between words initially.
* **Slower Learning:** The model might take longer to converge on optimal word representations if the random starting points are far from the ideal ones.

**Impact on Model Output:**

The quality of the word embeddings directly affects the model's output:

* **Better Embeddings, Better Outputs:** If the model starts with good word representations that capture semantic relationships, it will be better at predicting the next word in a sentence and generating more coherent and relevant outputs.
* **Poor Embeddings, Poor Outputs:** With random embeddings, the model might struggle to understand the context and relationships between words. This can lead to nonsensical or grammatically incorrect outputs. 

**Example:**

Consider the sentence "The quick brown fox jumps over the lazy dog."

* **With good embeddings:** The model might identify the relationship between "fox" and "jumps" and predict "jumps" as the next word.
* **With poor embeddings:** The model might struggle to connect "fox" to any meaningful word and might predict something unrelated, like "The" or "dog."

In summary, while randomly generated embeddings may seem arbitrary at first, they play a crucial role in initializing the model and allowing it to learn meaningful representations of words.

## Calculating Self-Attention

The `calculate_self_attention` function calculates the attention scores for each word in the context. It then computes the attention weights by applying the exponential function to the scores and normalizing them.

```python
def calculate_self_attention(query, keys, values):
    scores = np.dot(query, keys.T) / np.sqrt(keys.shape[1])
    attention_weights = np.empty_like(scores)
    for i in range(len(scores)):
        if len(keys[i].shape) == 1:  # Check if 1D array
            attention_weights[i] = np.exp(scores[i])  # No need to sum for unique words
        else:
            attention_weights[i] = np.exp(scores[i]) / np.sum(np.exp(scores[i]), axis=1, keepdims=True)

    return attention_weights
```

The attention weights show how much importance the model assigns to each word in the context when predicting the next word. Higher weights indicate greater relevance.

- The: 2.1307
- quick: 2.5428
- brown: 1.9087
- fox: 2.6365
- jumps: 2.2119
- over: 1.2500
- the: 2.1166
- lazy: 2.5802
- dog: 1.5677

As you can see, the words "quick," "fox," and "lazy" have the highest weights, suggesting they are the most important for predicting the next word.

## Predicting the Next Word with Self-Attention

> **Important Clarification — A Common Pitfall:** Attention probabilities are **not** a probability distribution over the vocabulary. They are an `N x N` matrix that describes how each token in the input sequence attends to every other token. Taking an `argmax` over attention weights would only ever return a token that is *already in the context window* — it cannot generate a new word.
>
> To predict the next word, a real Transformer follows two distinct steps:
> 1. **Self-Attention** produces a *context-aware embedding* for each position (a vector in `d_model` space).
> 2. A **linear projection layer** (often called the *language modeling head* or `lm_head`) maps that embedding from `d_model` into the full vocabulary space (`vocab_size`), producing **logits**. A final softmax over those logits gives the true next-word probability distribution.

The `predict_next_word_with_self_attention` function below demonstrates this correctly. It computes a context vector from self-attention, then projects it through `W_vocab` into vocabulary space before applying softmax and `argmax`.

```python
def predict_next_word_with_self_attention(
    context_window,
    word_embeddings,
    word_to_index,
    index_to_word,
    W_vocab,
):
    """
    Predicts the next word by:
      1. Using self-attention to build a context-aware vector from the context window.
      2. Projecting that vector into vocabulary space (logits) via a learned matrix W_vocab.
      3. Applying softmax over the vocabulary to get a real probability distribution.
      4. Selecting the most likely token with argmax.
    """
    # 1. Look up embeddings for the context words. Shape: (context_len, d_model)
    context_embeddings = np.array(
        [word_embeddings[word_to_index[w]] for w in context_window]
    )

    # 2. Self-attention over the context. Q = K = V = context_embeddings here for simplicity.
    #    scaled_dot_product_attention returns a context-aware embedding per token.
    context_aware_embeddings, attention_weights = scaled_dot_product_attention(
        context_embeddings, context_embeddings, context_embeddings
    )

    # 3. Pool the contextualized tokens into a single context vector.
    #    Real Transformers (e.g. GPT) instead take the last token's hidden state.
    context_vector = context_aware_embeddings.mean(axis=0)  # shape: (d_model,)

    # 4. Project from embedding space (d_model) into vocabulary space (vocab_size).
    #    W_vocab has shape (d_model, vocab_size). In real models this is a LEARNED matrix
    #    (often weight-tied to the input embedding matrix).
    logits = np.dot(context_vector, W_vocab)  # shape: (vocab_size,)

    # 5. Softmax over the FULL vocabulary -> a true next-word probability distribution.
    vocab_probabilities = softmax(logits)

    # 6. Pick the highest-probability word from the entire vocabulary (greedy decoding).
    predicted_index = int(np.argmax(vocab_probabilities))
    predicted_word = index_to_word[predicted_index]

    return predicted_word, vocab_probabilities, attention_weights
```

Notice the distinction: `attention_weights` describes *how the context attends to itself*, while `vocab_probabilities` is what we actually use to choose the next word. Conflating the two is one of the most common errors when first learning Transformers.

## Breaking Down the Prediction Process

![](https://miro.medium.com/v2/resize:fit:1400/1*kXg3zEXnzRDzSBrYLKlnxA.png)
_(Image Credit to [An illustration of next word prediction with state-of-the-art network architectures like BERT, GPT, and XLNet](https://ajay-arunachalam08.medium.com/an-illustration-of-next-word-prediction-with-state-of-the-art-network-architectures-like-bert-gpt-c0af02921f17) )_

This code is implementing a simple version of the self-attention mechanism, which is a key component in Transformer models used in natural language processing. The self-attention mechanism allows the model to weigh the importance of words in a sentence when predicting the next word.

Here's a breakdown of the code:

1. `create_word_representations(sentences)`: This function takes a list of sentences as input and creates a word-to-index and index-to-word dictionary, and a list of word embeddings. Each unique word in the sentences is assigned a unique index and a random 3-dimensional vector as its embedding.

2. `calculate_self_attention(query, key, value)`: This function calculates the self-attention weights and the output vector. The attention weights are calculated by taking the dot product of the query and key, scaling it, and applying the softmax function. The output vector is the weighted sum of the value vectors, where the weights are the attention weights.

3. `predict_next_word_with_self_attention(context_window, word_embeddings, word_to_index, index_to_word, W_vocab)`: This function predicts the next word given a context window. It (a) runs self-attention over the context to build context-aware embeddings, (b) pools them into a single context vector, (c) projects that vector into vocabulary space using the learned matrix `W_vocab` (the "language modeling head") to produce **logits**, (d) applies softmax over the entire vocabulary, and (e) selects the most likely token. The argmax is taken over the *vocabulary distribution*, **not** over attention weights — attention weights only describe relationships *within the input sequence* and cannot, by themselves, generate a new word.

## Running the Model

Finally, we run the model on a sentence. We initialize a random vocabulary projection matrix `W_vocab` (which a real model would *learn* via backpropagation alongside the embeddings and attention weights), then predict the next word given a two-word context.

```python
if __name__ == "__main__":
    sentences = [
        "The quick brown fox jumps over the lazy dog",
    ]

    word_embeddings, word_to_index, index_to_word = create_word_representations(sentences)
    vocab_size, d_model = word_embeddings.shape

    # W_vocab is the "language modeling head": it projects a d_model-dimensional
    # context vector into a vocab_size-dimensional logit vector.
    # In a real Transformer this matrix is LEARNED during training (and is often
    # weight-tied to the input embedding matrix).
    np.random.seed(0)
    W_vocab = np.random.rand(d_model, vocab_size)

    current_word = "jumps"
    context_window_size = 2  # Considering two words before the current word

    for sentence in sentences:
        words = sentence.split()
        current_word_index = words.index(current_word)
        context_window = words[max(0, current_word_index - context_window_size):current_word_index]

        predicted_word, vocab_probabilities, attention_weights = predict_next_word_with_self_attention(
            context_window,
            word_embeddings,
            word_to_index,
            index_to_word,
            W_vocab,
        )

        print(f"\nGiven the word: {current_word}")
        print(f"Context: {' '.join(context_window)}")
        print(f"Sentence: {sentence}")

        print("\nNext-word probabilities over the VOCABULARY (sums to 1.0):")
        for idx, prob in enumerate(vocab_probabilities):
            print(f"\t{index_to_word[idx]}: {prob:.4f}")

        print(f"\nPredicted next word: {predicted_word}")

    print("""
Note: The word embeddings and W_vocab are randomly initialized here for
illustration. In a real Transformer, BOTH the embedding matrix and W_vocab
are learned parameters, updated via backpropagation on a next-token-prediction
objective. That training is what gives the model its semantic understanding.
""")
```

This code provides a basic model that uses self-attention to predict the next word in a sentence. It demonstrates the core idea of self-attention but lacks the complexity of more advanced models like Transformers, which utilize this mechanism extensively.

## **Additional Considerations:**

- The quality of the word embeddings used can significantly impact the model's performance.
- The size of the vocabulary and the complexity of the language can also affect the model's accuracy.

**Intelligence is a Product of Training**

The "intelligence" of an LLM is directly tied to the quality and diversity of its training data. Here's how:

* **Data Bias:** If the training data is biased, the LLM will also be biased in its outputs. For example, an LLM trained on mostly news articles might struggle to understand sarcasm or humor. 
* **Data Limitedness:** The real world is vast and complex. LLMs can only process what they've been trained on. Limited data can lead to incomplete understanding and difficulty handling unexpected situations.
* **Training Objectives:** Ultimately, LLMs are optimized for the tasks they are trained on. An LLM trained for text summarization may not excel at creative writing tasks, even if the data is vast.

## Exploring Different Types of Attention Mechanisms

While we've discussed the basic concept of attention, it's important to note that there are several types of attention mechanisms used in different models. One of the most notable is multi-head attention, which is a key component of Transformer models. 

**NOTE — Embeddings Are Learned, Not Frozen:**
In nearly all foundational Transformer models (BERT, GPT-2/3/4, T5, and the original *Attention Is All You Need* paper), the input word-embedding matrix is a **learnable parameter**. It is initialized randomly and then actively updated via backpropagation on the training objective, right alongside the attention projections (`W_Q`, `W_K`, `W_V`), the feed-forward weights, and the output projection (`W_vocab` / `lm_head`). This is *how* the model learns semantic relationships between words in the first place — geometric structure in the embedding space (e.g. "king" - "man" + "woman" ≈ "queen") emerges *because* the embeddings move during training.

Embeddings are only held frozen in specific situations, such as:
- Certain **transfer-learning / fine-tuning** setups where you intentionally freeze the embedding layer to preserve pretrained representations and reduce trainable parameters.
- Parameter-efficient methods like **LoRA** or **adapters** that freeze the base model (including embeddings) and only train small added modules.
- When using **pretrained static embeddings** (Word2Vec, GloVe) as fixed features in a downstream model.

In standard end-to-end Transformer pretraining, embeddings are very much *not* constant.

![Transformers in generative models.](https://www.jeremyjordan.me/content/images/2023/05/multi-head-attention.png)

### Multi-Head Attention

Multi-head attention is a type of attention mechanism that allows the model to focus on different parts of the input sequence simultaneously. It does this by splitting the input into multiple "heads" and applying the attention mechanism to each head independently. This allows the model to capture various aspects of the input sequence, such as different levels of abstraction or different types of relationships between words.

In the context of language models, multi-head attention can help the model understand complex sentences where different words have different relationships with each other. For example, in the sentence "The cat sat on the mat," the word "cat" is related to "sat" (the action the cat is performing) and "mat" (the location of the action). Multi-head attention allows the model to capture both of these relationships simultaneously.

## Conclusion

**The Challenges of LLM Intelligence**

The "intelligence" of an LLM heavily depends on the quality and variety of its training data. Biases, limitations in the data itself, and narrow training objectives can all hinder a model's ability to represent the real world's complexities. Just like a student highlighting doesn't guarantee comprehension, attention in LLMs doesn't guarantee true understanding. 

## SCRIPT OUTPUT

```
--- 1. INPUT PROCESSING ---
Sentence: 'the cat sat on the mat'
Sequence length (N): 6 tokens
Input Embeddings (X) shape: (6, 4)

--- 2. LINEAR PROJECTIONS ---
Queries (Q) shape: (6, 3)
Keys (K) shape:    (6, 3)
Values (V) shape:  (6, 3)

--- 3. CALCULATING SELF-ATTENTION ---
Attention Weights (N x N matrix):
How much each word attends to every other word (Rows sum to 1.0)
          the    cat    sat     on    the    mat
   the  0.271  0.073  0.168  0.103  0.271  0.114
   cat  0.227  0.106  0.170  0.132  0.227  0.139
   sat  0.267  0.074  0.167  0.109  0.267  0.116
    on  0.236  0.102  0.172  0.120  0.236  0.133
   the  0.271  0.073  0.168  0.103  0.271  0.114
   mat  0.229  0.106  0.172  0.128  0.229  0.136

--- 4. FINAL OUTPUT ---
Context-Aware Embeddings shape: (6, 3)
Notice how each token's original vector has been replaced by a new vector
that is a weighted sum of ALL the Values in the sequence.

Original embedding for 'cat': [0.156 0.156 0.058 0.866]
New context-aware embedding for 'cat': [1.366 1.387 0.626]
```

## FULL SOURCE CODE

```python
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
```

## Resources

https://eugeneyan.com/writing/attention/




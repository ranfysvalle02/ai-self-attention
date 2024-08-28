import numpy as np

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

def calculate_self_attention(query, keys, values):
    scores = np.dot(query, keys.T) / np.sqrt(keys.shape[1])
    attention_weights = np.empty_like(scores)
    for i in range(len(scores)):
        if len(keys[i].shape) == 1:  # Check if 1D array
            attention_weights[i] = np.exp(scores[i])  # No need to sum for unique words
        else:
            attention_weights[i] = np.exp(scores[i]) / np.sum(np.exp(scores[i]), axis=1, keepdims=True)

    return attention_weights

def predict_next_word_with_self_attention(current_word, context_window, words, word_embeddings, word_to_index, index_to_word):
    context_embeddings = word_embeddings[[word_to_index[word] for word in context_window]]
    query = np.mean(context_embeddings, axis=0)  # Average context embeddings
    keys = values = np.array([word_embeddings[word_to_index[word]] for word in words])
    attention_weights = calculate_self_attention(query, keys, values)
    attention_probabilities = softmax(attention_weights)
    predicted_index = np.argmax(attention_probabilities)  # Select the word with the highest probability
    predicted_word = index_to_word[predicted_index]
    return predicted_word, attention_probabilities

if __name__ == "__main__":
    sentences = [
        "The quick brown fox jumps over the lazy dog",
    ]

    word_embeddings, word_to_index, index_to_word = create_word_representations(sentences)
    current_word = "jumps"
    context_window_size = 2  # Considering two words before the current word

    for sentence in sentences:
        words = sentence.split()
        current_word_index = words.index(current_word)
        context_window = words[max(0, current_word_index - context_window_size):current_word_index]
        predicted_word, attention_probabilities = predict_next_word_with_self_attention(current_word, context_window, words, word_embeddings, word_to_index, index_to_word)
        print(f"\nGiven the word: {current_word}")
print(f"Context: {' '.join(context_window)}")  # Print context window
print(f"Sentence: {sentence}")
print("Attention Probabilities:")
for word, prob in zip(words, attention_probabilities):
    print(f"\t{word}: {prob:.4f}")
print(f"Predicted next word: {predicted_word}\n")
print("""
The word embeddings are initialized randomly in this code. 
This means that the relationships between different words are not captured in the embeddings, 
which could lead to seemingly random attention probabilities.
""")
print("""
The input triggers the attention mechanism which is used to weight 
the importance of different words in the sentence for the prediction of the next word.
""")
print(f"Prediction process: The model uses the context of the given word '{current_word}' to predict the next word. The attention mechanism assigns different weights to the words in the context based on their relevance. The word with the highest weight is considered as the most relevant word for the prediction.")
print(f"Attention Impact: The attention probabilities show the relevance of each word in the context for the prediction. The higher the probability, the more impact the word has on the prediction.\n")

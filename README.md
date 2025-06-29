# Sentiment Analysis Using Deep Learning and Word Embeddings

This project implements sentiment analysis on the IMDb movie review dataset using deep learning models with both Bag of Words (BoW) and word embeddings approaches. The notebook compares the performance of SimpleRNN, GRU, LSTM, and BiLSTM models.

## Project Overview

The notebook `Sentiment_analysis_using_deep_learning_and_word_embeddings.ipynb` performs the following tasks:
- Loads and preprocesses the IMDb dataset (positive and negative reviews).
- Implements sentiment analysis using:
  - Bag of Words (BoW) with CountVectorizer.
  - Word embeddings with Keras Tokenizer.
- Trains and evaluates four deep learning models: SimpleRNN, GRU, LSTM, and BiLSTM.
- Reports performance metrics (Accuracy, Precision, Recall, F1-score) for each model.

## Requirements

To run the notebook, install the following dependencies:

```bash
pip install numpy pandas sklearn nltk keras
```
Additionally, download the required NLTK data:

- `punkt` for tokenization
- `stopwords` for stop word removal
## Dataset
The project uses the IMDb dataset (`aclImdb_v1.tar.gz`). The dataset should be placed in a directory accessible to the notebook (e.g., Google Drive for Colab users).

Directory Structure:
- `aclImdb/train/pos/`: Positive reviews
- `aclImdb/train/neg/`: Negative reviews
**Note:** The notebook processes up to 2000 files from each category to limit computational load.
## Preprocessing
The preprocessing steps include:

- Converting text to lowercase.
- Removing punctuation.
- Tokenizing text using NLTK's `word_tokenize`.
- Removing English stop words.
- Combining and shuffling positive and negative reviews.

## Models
Two sets of models are implemented:

1- **Bag of Words Models:**
- SimpleRNN
- GRU
- LSTM
- BiLSTM
- **Features:** CountVectorizer with a vocabulary size of 10,000 and sequences padded to a length of 100.

2- **Word Embedding Models:**
- SimpleRNN with Embedding
- GRU with Embedding
- LSTM with Embedding
- BiLSTM with Embedding
- **Features:** Keras Tokenizer with a vocabulary size of 10,000 and 64-dimensional embeddings.
- **Note:**Each model is trained with the Adam optimizer, binary cross-entropy loss, and early stopping (for BoW models) to prevent overfitting.


## Results
The notebook outputs performance metrics for each model on the test set. Example results (actual values may vary):

1- **Bag of Words Models**
- `RNN:` Accuracy: 0.562, Precision: 0.559, Recall: 0.682, F1-score: 0.614
- `GRU:` Accuracy: 0.508, Precision: 0.510, Recall: 0.933, F1-score: 0.660
- `LSTM:` Accuracy: 0.519, Precision: 0.523, Recall: 0.666, F1-score: 0.586
- `BiLSTM:` Accuracy: 0.616, Precision: 0.614, Recall: 0.674, F1-score: 0.642


2- **Word Embedding Models**
- `RNN with Embedding`: Accuracy: 0.792, Precision: 0.786, Recall: 0.815, F1-score: 0.800
- `GRU with Embedding:` Accuracy: 0.854, Precision: 0.878, Recall: 0.831, F1-score: 0.854
- `LSTM with Embedding:` Accuracy: 0.870, Precision: 0.886, Recall: 0.857, F1-score: 0.871
- `BiLSTM with Embedding:` Accuracy: 0.860, Precision: 0.853, Recall: 0.878, F1-score: 0.865

 ## Notes
- The notebook assumes access to the IMDb dataset in a tar.gz format. Adjust file paths if using a different setup.
- The BoW models may show lower performance due to the sparse nature of the input. Word embedding models generally perform better.
- Training times vary depending on hardware (GPU recommended for faster computation).
 ## License
- This project is for educational purposes and uses the publicly available IMDb dataset. Ensure compliance with the dataset's usage terms.

  ## Author
  - M Burhan ud din

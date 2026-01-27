import pandas as pd
from IPython.display import display
import re


def display_info(df):
    rows = display(df[:10])
    print("\n")
    info = display(df.info())
    return rows, info


#
def func_clean(text):
    text = str(text).lower()
    pattern = "(\\[.{1,}\\]){1,}"
    text = re.sub(pattern, "", text)
    # Convert to lower case

    # Remove text in square brackets

    # Remove punctuation
    pattern = '[.,":;!-?()]{1,}'
    text = re.sub(pattern, "", text)
    # Remove words with numbers
    pattern = "([A-z]{1,}[0-9]{1,}.{0,}){1,}"
    text = re.sub(pattern, "", text)
    return text


# # Write the function for POS tagging and lemmatization, filtering stopwords and keeping only NN and NNS tags
#
# import nltk
# from nltk.corpus import stopwords
# from tqdm.notebook import tqdm
#
# # Download the stopwords corpus
# nltk.download("stopwords")
#
# # Load spacy model
# nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
# stop_words = set(stopwords.words("english"))
#
#
# def pos_tag_lemmatize(texts):
#     """
#     Apply POS-based lemmatization to a list/series of texts.
#     Only nouns (NN, NNS) are kept, lemmatized and lowercased.
#     """
#     lemmatized_texts = []
#
#     # Batch processing using spaCy's pipe
#     for doc in tqdm(nlp.pipe(texts, batch_size=50, n_process=10), total=len(texts)):
#         tokens = []
#         for token in doc:
#             if (
#                 (token.tag_ in ["NN", "NNS"])
#                 and (token.text.lower() not in stop_words)
#                 and token.is_alpha
#             ):
#                 tokens.append(token.lemma_.lower())
#         lemmatized_texts.append(" ".join(tokens))
#
#     return lemmatized_texts


def todo(df):
    text = df["cleaned_text"].tolist()
    out = []
    for t in text:
        out.extend(t.split())
    out1 = {}
    for item in out:
        if item in out1:
            out1[item] += 1
        else:
            out1[item] = 1

    x = dict(sorted(out1.items(), key=lambda x: x[1], reverse=True))
    print(list(x.items())[:100])


# from gensim.models import Word2Vec
#
# ## Write your code here to extract the vectors from the Word2Vec model for both training and validation data
#
#
# ## Extract the target variable for the training data and validation data target variable for the training data and validation data
#
#
#
#
# # --- Perform the split to get train_data_df and val_data_df (as full DFs) ---
# X = df_clean[['cleaned_text', 'lemmatized_news_text']]
# y = df_clean['news_label']
# X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# train_data_df = pd.concat([X_train_df, y_train], axis=1)
# val_data_df = pd.concat([X_val_df, y_val], axis=1)
# print("train_data_df and val_data_df prepared.")
#
# # --- Handle NaN/Empty values in the text data ---
# train_data_df['lemmatized_news_text'].fillna('', inplace=True)
# val_data_df['lemmatized_news_text'].fillna('', inplace=True)
# train_data_df['lemmatized_news_text'] = train_data_df['lemmatized_news_text'].astype(str).str.strip()
# val_data_df['lemmatized_news_text'] = val_data_df['lemmatized_news_text'].astype(str).str.strip()
# print("NaN/empty string values in text data handled for train and val sets.")
#
#
# # --- Prepare corpus for Word2Vec training ---
# sentences_train = [doc.split() for doc in train_data_df['lemmatized_news_text'].tolist()]
# print("\nPrepared sentences corpus for Word2Vec training.")
#
# # --- Initialize and Train the Custom Word2Vec Model ---
# print("\nInitializing and training custom Word2Vec model...")
# # Using default parameters suitable for demonstration
# custom_word2vec_model = Word2Vec(sentences=sentences_train,
#                                  vector_size=300, # 300 dimensions as requested previously for Google News
#                                  window=5,
#                                  min_count=1, # Keep all words for small dummy dataset
#                                  workers=4,
#                                  sg=1) # Skip-gram
#
# custom_word2vec_model.build_vocab(sentences_train)
# custom_word2vec_model.train(sentences_train,
#                             total_examples=custom_word2vec_model.corpus_count,
#                             epochs=custom_word2vec_model.epochs)
# print("Custom Word2Vec model trained successfully on the training data.")
#
#
# # --- Function to extract document vectors ---
# def get_document_vector(text_tokens, model, vector_size):
#     """
#     Computes the average Word2Vec vector for a given list of tokens.
#
#     Returns:
#         The average vector for the document. Returns a zero vector
#         if no words from the document are found in the model's vocabulary.
#     """
#     vectors = []
#     for word in text_tokens:
#         if word in model.wv: # Checking if word is in the model's vocabulary
#             vectors.append(model.wv[word])
#     if vectors:
#         return np.mean(vectors, axis=0) # Averaging the vectors
#     else:
#         return np.zeros(vector_size) # Returns a zero vector if no words were found
#
# print("\n--- Extracting Word2Vec Vectors for Training Data ---")
# # Apply the function to the 'lemmatized_news_text' column for training data
# X_train_vectors = np.array([
#     get_document_vector(doc.split(), model, model.vector_size)
#     for doc in train_data_df['lemmatized_news_text'].tolist()
# ])
#
# print(f"Shape of X_train_vectors (Word2Vec vectors for training data): {X_train_vectors.shape}")
# print(f"First 5 elements of the first training vector:\n{X_train_vectors[0][:5]}")
#

import numpy as np


def calculate_sentence_vector_word2vec(model, sentence):
    """
    Function to calculate the sentence vector using a trained Word2Vec model.

    Args:
    - model: Trained Word2Vec model.
    - sentence: Input sentence represented as a list of words.

    Returns:
    - Sentence vector.
    """
    word_vectors = [
        model[word] for word in sentence if word in model
    ]  # train model -keyed vectors - vector
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    # Return zero vector if no words are found in the model
    X_train_word2vec = [
        calculate_sentence_vector_word2vec(model, sentence)
        for sentence in X_train["news_text"]
    ]
    X_val_word2vec = [
        calculate_sentence_vector_word2vec(model, sentence)
        for sentence in X_val["news_text"]
    ]


## Extract the target variable for the training data and validation data


import numpy as np
from gensim.utils import simple_preprocess

# Assuming word2vec_model is already loaded
# word2vec_model = api.load("word2vec-google-news-300")


def document_vector(doc, model):
    """
    Create document vectors by averaging word vectors for words in the document.
    Words not in vocabulary are ignored.
    """
    # Tokenize and preprocess document
    words = simple_preprocess(doc)

    # Filter tokens that exist in model vocabulary
    valid_words = [word for word in words if word in model.key_to_index]

    if not valid_words:
        # Return zero vector if no valid words found
        return np.zeros(model.vector_size)

    # Average word vectors
    vectors = [model[word] for word in valid_words]
    print(vectors[:10])
    return np.mean(vectors, axis=0)

    # Print shapes to confirm
    print("X_train_vectors shape:", X_train_vectors.shape)
    print("X_val_vectors shape:", X_val_vectors.shape)
    print("y_train_array shape:", y_train_array.shape)
    print("y_val_array shape:", y_val_array.shape)

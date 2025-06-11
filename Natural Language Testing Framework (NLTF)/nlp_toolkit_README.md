
# 🧠 NLP Toolkit

A modular, extensible Natural Language Processing (NLP) toolkit written in Python, designed to help you **preprocess text**, **extract features**, **evaluate models**, and **run test suites** with ease.

## 📦 Key Features

- **Preprocessing Pipeline**
  - Lowercasing, punctuation & number removal
  - Tokenization using NLTK
  - Stopword removal
  - Stemming and Lemmatization

- **Feature Extraction**
  - Bag-of-Words (BoW) representation
  - TF-IDF vectorization
  - Word Embedding averaging (with external models)

- **Evaluation Metrics**
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix
  - Text Similarity (Jaccard, Cosine, Levenshtein)

- **Test Framework**
  - Easily add unit-like test cases
  - Automatically evaluate predictions
  - Generate test summaries and detailed diagnostics

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nlp-toolkit.git
cd nlp-toolkit
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: First-time users of NLTK may need to download required data.

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🧪 Quick Example

```python
from nlp_toolkit import NLPreprocessor, NLTestSuite

def basic_sentiment(text):
    pos_words = ['great', 'happy', 'excellent']
    neg_words = ['bad', 'awful', 'sad']
    tokens = NLPreprocessor().full_preprocess(text)
    pos, neg = sum(w in pos_words for w in tokens), sum(w in neg_words for w in tokens)
    return "positive" if pos > neg else "negative" if neg > pos else "neutral"

test_suite = NLTestSuite()
test_suite.add_test_case("This is an excellent product!", "positive", description="Positive sentiment")
test_suite.add_test_case("The service was awful.", "negative", description="Negative sentiment")
test_suite.run_tests(basic_sentiment)
```

## 🧹 Preprocessing Pipeline

```python
pre = NLPreprocessor()
text = "This product is AMAZING and arrived on 12/12/2024!"
cleaned_tokens = pre.full_preprocess(text)
print(cleaned_tokens)
```

**Output:**
```
['product', 'amazing', 'arrived']
```

## 🔎 Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_toolkit import NLFeatureExtractor

texts = ["the cat sat on the mat", "the dog barked at the cat"]
vectorizer = TfidfVectorizer()
vectorizer.fit(texts)

tokens = ["cat", "sat", "mat"]
vector = NLFeatureExtractor.tfidf_vectorize(tokens, vectorizer)
```

## 🧪 Test Framework

```python
test_suite = NLTestSuite()
test_suite.add_test_case("I am happy with this.", "positive", description="Positive case")
test_suite.add_test_case("Very bad experience", "negative", description="Negative case")
results = test_suite.run_tests(basic_sentiment)
```

## 📊 Evaluation Metrics

```python
from nlp_toolkit import NLEvaluator

y_true = ["positive", "negative", "neutral"]
y_pred = ["positive", "negative", "positive"]

metrics = NLEvaluator.calculate_metrics(y_true, y_pred)
print(metrics)
```

**Text Similarity Example:**
```python
sim = NLEvaluator.calculate_similarity("The dog barks", "A dog is barking", method="jaccard")
print(f"Jaccard Similarity: {sim:.2f}")
```

## 📁 File Structure

```
nlp_toolkit/
├── nlp_toolkit.py           # Main codebase
├── README.md
├── requirements.txt
├── examples/
│   └── test_sentiment_model.py
```

## 🧑‍💻 Contribution

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Open a Pull Request

## 🌟 Acknowledgements

- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)

## 📬 Contact

Created by Sachin Kamal - feel free to reach out via GitHub issues or pull requests.

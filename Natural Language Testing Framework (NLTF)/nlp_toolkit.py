#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Toolkit: Preprocessing, Feature Extraction, Testing, & Evaluation
Author: Sachin Kamal
Date: 2025-06-07
Purpose:  Toolkit General Outline
Description:
    - Preprocessing tools using NLTK
    - Feature extraction: BoW, TF-IDF, embeddings
    - Testing framework for the NLP models
    - Evaluation metrics and text similarity functions
"""

import re
import string
from typing import List, Dict, Tuple, Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# --------------------- Preprocessing ----------------------

class NLPreprocessor:
    """
    Natural Language Preprocessing class for text cleaning and normalization
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        return ' '.join(text.split())

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def full_preprocess(self, text: str) -> List[str]:
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return self.lemmatize(tokens)


# --------------------- Feature Extraction ---------------------

class NLFeatureExtractor:
    """
    Feature extraction from natural language text
    """

    @staticmethod
    def bag_of_words(tokens: List[str], vocab: Dict[str, int]) -> np.ndarray:
        vector = np.zeros(len(vocab))
        for token in tokens:
            if token in vocab:
                vector[vocab[token]] += 1
        return vector

    @staticmethod
    def tfidf_vectorize(tokens: List[str], tfidf_vectorizer: Any) -> np.ndarray:
        return tfidf_vectorizer.transform([' '.join(tokens)]).toarray()

    @staticmethod
    def word_embeddings(tokens: List[str], embedding_model: Any) -> np.ndarray:
        embeddings = [embedding_model[token] for token in tokens if token in embedding_model]
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(embedding_model.vector_size)


# --------------------- Test Suite ---------------------

class NLTestSuite:
    """
    Test suite for NLP models
    """

    def __init__(self):
        self.preprocessor = NLPreprocessor()
        self.test_cases = []

    def add_test_case(self, input_text: str, expected_output: Any, processor: Callable = None, description: str = ""):
        self.test_cases.append({
            'input': input_text,
            'expected': expected_output,
            'processor': processor,
            'description': description
        })

    def run_tests(self, model: Callable, verbose: bool = True) -> Dict[str, Any]:
        results = {
            'total': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'details': [],
            'accuracy': 0
        }

        y_true = []
        y_pred = []

        for case in self.test_cases:
            try:
                prediction = model(case['input'])
                if case['processor']:
                    prediction = case['processor'](prediction)

                is_correct = prediction == case['expected'] if not isinstance(case['expected'], (list, np.ndarray)) \
                    else np.array_equal(prediction, case['expected'])

                results['passed'] += int(is_correct)
                results['failed'] += int(not is_correct)

                y_true.append(case['expected'])
                y_pred.append(prediction)

                results['details'].append({
                    'description': case['description'],
                    'input': case['input'],
                    'expected': case['expected'],
                    'prediction': prediction,
                    'passed': is_correct
                })

                if verbose:
                    print(f"{'PASSED' if is_correct else 'FAILED'} - {case['description']}")
                    if not is_correct:
                        print(f"  Input: {case['input']}")
                        print(f"  Expected: {case['expected']}")
                        print(f"  Got: {prediction}\n")

            except Exception as e:
                results['failed'] += 1
                if verbose:
                    print(f"ERROR - {case['description']}")
                    print(f"  Exception: {str(e)}\n")

        if y_true:
            try:
                results.update({
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted'),
                    'recall': recall_score(y_true, y_pred, average='weighted'),
                    'f1': f1_score(y_true, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
                })
            except:
                pass

        if verbose:
            print("\nTest Summary:")
            print(f"Total: {results['total']}")
            print(f"Passed: {results['passed']}")
            print(f"Failed: {results['failed']}")
            if 'accuracy' in results:
                print(f"Accuracy: {results['accuracy']:.2f}")

        return results


# --------------------- Evaluation ---------------------

class NLEvaluator:
    """
    Evaluation metrics and similarity computation
    """

    @staticmethod
    def calculate_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

    @staticmethod
    def calculate_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
        preprocessor = NLPreprocessor()
        tokens1 = preprocessor.full_preprocess(text1)
        tokens2 = preprocessor.full_preprocess(text2)

        if method == 'jaccard':
            set1, set2 = set(tokens1), set(tokens2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union else 0

        elif method == 'cosine':
            vocab = list(set(tokens1 + tokens2))
            vec1 = np.array([1 if word in tokens1 else 0 for word in vocab])
            vec2 = np.array([1 if word in tokens2 else 0 for word in vocab])
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return dot_product / norm_product if norm_product else 0

        elif method == 'levenshtein':
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if not s2:
                    return len(s1)

                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                return previous_row[-1]

            distance = levenshtein_distance(text1, text2)
            return 1 - (distance / max(len(text1), len(text2))) if max(len(text1), len(text2)) else 0

        else:
            raise ValueError(f"Unsupported method: {method}")


# --------------------- Example Usage ---------------------

if __name__ == "__main__":
    def example_sentiment_model(text: str) -> str:
        positive_words = ['good', 'great', 'excellent', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'sad']
        tokens = NLPreprocessor().full_preprocess(text)
        pos, neg = sum(w in positive_words for w in tokens), sum(w in negative_words for w in tokens)
        return "positive" if pos > neg else "negative" if neg > pos else "neutral"

    suite = NLTestSuite()
    suite.add_test_case("This product is great! I'm very happy with it.", "positive", description="Positive test")
    suite.add_test_case("The service was terrible and the food was awful.", "negative", description="Negative test")
    suite.add_test_case("The item arrived on time.", "neutral", description="Neutral test")
    suite.run_tests(example_sentiment_model)

    evaluator = NLEvaluator()
    sim = evaluator.calculate_similarity("The cat sat on the mat", "The kitten sat on the rug", method='jaccard')
    print(f"\nJaccard similarity: {sim:.2f}")

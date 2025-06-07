
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

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class NLPreprocessor:
    """
    Natural Language Preprocessing class for text cleaning and normalization
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from token list"""
        return [word for word in tokens if word not in self.stop_words]
    
    def stem(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        return [self.stemmer.stem(word) for word in tokens]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def full_preprocess(self, text: str) -> List[str]:
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens

class NLFeatureExtractor:
    """
    Feature extraction from natural language text
    """
    @staticmethod
    def bag_of_words(tokens: List[str], vocab: Dict[str, int]) -> np.ndarray:
        """Convert tokens to bag-of-words vector"""
        vector = np.zeros(len(vocab))
        for token in tokens:
            if token in vocab:
                vector[vocab[token]] += 1
        return vector
    
    @staticmethod
    def tfidf_vectorize(tokens: List[str], tfidf_vectorizer: Any) -> np.ndarray:
        """Convert text to TF-IDF vector (requires pre-fitted vectorizer)"""
        return tfidf_vectorizer.transform([' '.join(tokens)]).toarray()
    
    @staticmethod
    def word_embeddings(tokens: List[str], embedding_model: Any) -> np.ndarray:
        """Get average word embeddings for tokens"""
        embeddings = []
        for token in tokens:
            if token in embedding_model:
                embeddings.append(embedding_model[token])
        if len(embeddings) > 0:
            return np.mean(embeddings, axis=0)
        return np.zeros(embedding_model.vector_size)

class NLTestSuite:
    """
    Test suite for natural language processing models
    """
    def __init__(self):
        self.preprocessor = NLPreprocessor()
        self.test_cases = []
    
    def add_test_case(self, 
                     input_text: str, 
                     expected_output: Any, 
                     processor: Callable = None,
                     description: str = ""):
        """
        Add a test case to the suite
        
        Args:
            input_text: Input text to test
            expected_output: Expected output from the model
            processor: Optional function to process model output before comparison
            description: Description of the test case
        """
        self.test_cases.append({
            'input': input_text,
            'expected': expected_output,
            'processor': processor,
            'description': description
        })
    
    def run_tests(self, model: Callable, verbose: bool = True) -> Dict[str, Any]:
        """
        Run all test cases against the provided model
        
        Args:
            model: Callable that takes text input and returns predictions
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with test results and metrics
        """
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
                # Get model prediction
                prediction = model(case['input'])
                
                # Process prediction if processor provided
                if case['processor'] is not None:
                    prediction = case['processor'](prediction)
                
                # Compare with expected output
                if isinstance(case['expected'], (list, np.ndarray)):
                    is_correct = np.array_equal(prediction, case['expected'])
                else:
                    is_correct = prediction == case['expected']
                
                # Record results
                if is_correct:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                
                # For metrics calculation
                y_true.append(case['expected'])
                y_pred.append(prediction)
                
                # Store details
                details = {
                    'description': case['description'],
                    'input': case['input'],
                    'expected': case['expected'],
                    'prediction': prediction,
                    'passed': is_correct
                }
                results['details'].append(details)
                
                if verbose:
                    status = "PASSED" if is_correct else "FAILED"
                    print(f"{status} - {case['description']}")
                    if not is_correct:
                        print(f"  Input: {case['input']}")
                        print(f"  Expected: {case['expected']}")
                        print(f"  Got: {prediction}")
                        print()
            
            except Exception as e:
                results['failed'] += 1
                if verbose:
                    print(f"ERROR - {case['description']}")
                    print(f"  Exception: {str(e)}")
                    print()
        
        # Calculate metrics if possible
        if len(y_true) > 0:
            try:
                results['accuracy'] = accuracy_score(y_true, y_pred)
                results['precision'] = precision_score(y_true, y_pred, average='weighted')
                results['recall'] = recall_score(y_true, y_pred, average='weighted')
                results['f1'] = f1_score(y_true, y_pred, average='weighted')
                results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
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

class NLEvaluator:
    """
    Evaluation metrics for natural language models
    """
    @staticmethod
    def calculate_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
        """
        Calculate standard classification metrics
        
        Args:
            y_true: List of true labels
            y_pred: List of predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        return metrics
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
        """
        Calculate text similarity between two strings
        
        Args:
            text1: First text string
            text2: Second text string
            method: Similarity method ('jaccard', 'cosine', 'levenshtein')
            
        Returns:
            Similarity score between 0 and 1
        """
        preprocessor = NLPreprocessor()
        tokens1 = preprocessor.full_preprocess(text1)
        tokens2 = preprocessor.full_preprocess(text2)
        
        if method == 'jaccard':
            set1 = set(tokens1)
            set2 = set(tokens2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union != 0 else 0
        
        elif method == 'cosine':
            # For simplicity, using binary presence here
            # In practice, you'd use TF-IDF or word embeddings
            vocab = list(set(tokens1 + tokens2))
            vec1 = [1 if word in tokens1 else 0 for word in vocab]
            vec2 = [1 if word in tokens2 else 0 for word in vocab]
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2) if (norm1 * norm2) != 0 else 0
        
        elif method == 'levenshtein':
            # Using simple ratio for demonstration
            # For better results, consider python-Levenshtein package
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if len(s2) == 0:
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
            max_len = max(len(text1), len(text2))
            return 1 - (distance / max_len) if max_len != 0 else 0
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")

# Example usage
if __name__ == "__main__":
    # Example sentiment analysis test
    def example_sentiment_model(text: str) -> str:
        """Example dummy sentiment analysis model"""
        positive_words = ['good', 'great', 'excellent', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'sad']
        
        tokens = NLPreprocessor().full_preprocess(text)
        
        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count = sum(1 for word in tokens if word in negative_words)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"
    
    # Create test suite
    test_suite = NLTestSuite()
    
    # Add test cases
    test_suite.add_test_case(
        "This product is great! I'm very happy with it.",
        "positive",
        description="Positive sentiment test"
    )
    
    test_suite.add_test_case(
        "The service was terrible and the food was awful.",
        "negative",
        description="Negative sentiment test"
    )
    
    test_suite.add_test_case(
        "The item arrived on time.",
        "neutral",
        description="Neutral sentiment test"
    )
    
    # Run tests
    results = test_suite.run_tests(example_sentiment_model)
    
    # Evaluate similarity
    evaluator = NLEvaluator()
    similarity = evaluator.calculate_similarity(
        "The cat sat on the mat",
        "The kitten sat on the rug",
        method='jaccard'
    )
    print(f"\nText similarity: {similarity:.2f}")

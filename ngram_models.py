"""
N-Gram Language Modeling Implementation
DATA641 Homework 2

This module implements various N-gram language models including:
- Maximum Likelihood Estimation (MLE) models
- Add-1 (Laplace) smoothing
- Linear interpolation
- Stupid backoff
- Text generation

Author: Student
Course: DATA641
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import random

class NGramLanguageModel:
    """
    Base N-gram language model using Maximum Likelihood Estimation.
    
    This class implements the fundamental N-gram model that computes
    probabilities using relative frequencies from training data.
    """
    
    def __init__(self, n: int):
        """
        Initialize N-gram model.
        
        Args:
            n: Order of the N-gram model (1 for unigram, 2 for bigram, etc.)
        """
        self.n = n
        self.ngram_counts = defaultdict(int)  # Count of each N-gram
        self.context_counts = defaultdict(int)  # Count of each context (N-1 words)
        self.vocab = set()  # Vocabulary seen during training
        self.total_words = 0  # Total word count for unigram normalization
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing and adding sentence boundaries.
        
        Args:
            text: Raw input sentence
            
        Returns:
            List of tokens with appropriate sentence markers
        """
        text = text.lower().strip()
        tokens = text.split()
        
        # Add sentence boundary markers for N > 1
        if self.n > 1:
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        
        return tokens
    
    def get_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """Extract all N-grams from tokenized sentence."""
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            ngrams.append(ngram)
        return ngrams
    
    def train(self, corpus: List[str]):
        """
        Train the N-gram model on a corpus.
        
        Args:
            corpus: List of sentences to train on
        """
        for sentence in corpus:
            tokens = self.preprocess_text(sentence)
            self.vocab.update(tokens)
            
            ngrams = self.get_ngrams(tokens)
            
            # Count N-grams and their contexts
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                if self.n > 1:
                    context = ngram[:-1]  # First N-1 words
                    self.context_counts[context] += 1
        
        # Store total words for unigram normalization
        self.total_words = sum(self.ngram_counts.values()) if self.n == 1 else None
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        if self.n == 1:
            return self.ngram_counts[ngram] / sum(self.ngram_counts.values())
        else:
            context = ngram[:-1]
            if self.context_counts[context] == 0:
                return 0.0
            return self.ngram_counts[ngram] / self.context_counts[context]
    
    def calculate_perplexity(self, test_corpus: List[str]) -> float:
        log_prob_sum = 0
        total_words = 0
        
        for sentence in test_corpus:
            tokens = self.preprocess_text(sentence)
            ngrams = self.get_ngrams(tokens)
            
            for ngram in ngrams:
                prob = self.get_probability(ngram)
                if prob == 0:
                    return float('inf')
                log_prob_sum += math.log(prob)
                total_words += 1
        
        if total_words == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / total_words
        perplexity = math.exp(-avg_log_prob)
        return perplexity

class AddOneSmoothedNGramModel(NGramLanguageModel):
    def __init__(self, n: int):
        super().__init__(n)
        self.vocab_size = 0
    
    def train(self, corpus: List[str]):
        super().train(corpus)
        self.vocab_size = len(self.vocab)
    
    def get_probability(self, ngram: Tuple[str, ...]) -> float:
        if self.n == 1:
            return (self.ngram_counts[ngram] + 1) / (sum(self.ngram_counts.values()) + self.vocab_size)
        else:
            context = ngram[:-1]
            numerator = self.ngram_counts[ngram] + 1
            denominator = self.context_counts[context] + self.vocab_size
            return numerator / denominator

class LinearInterpolationModel:
    def __init__(self, lambdas: List[float]):
        assert len(lambdas) == 3, "Need exactly 3 lambda values for unigram, bigram, trigram"
        assert abs(sum(lambdas) - 1.0) < 1e-6, "Lambda values must sum to 1"
        
        self.lambdas = lambdas
        self.unigram_model = NGramLanguageModel(1)
        self.bigram_model = NGramLanguageModel(2)
        self.trigram_model = NGramLanguageModel(3)
        self.vocab = set()
    
    def train(self, corpus: List[str]):
        self.unigram_model.train(corpus)
        self.bigram_model.train(corpus)
        self.trigram_model.train(corpus)
        
        for sentence in corpus:
            tokens = self.unigram_model.preprocess_text(sentence)
            self.vocab.update(tokens)
    
    def get_probability(self, word: str, context: Tuple[str, ...]) -> float:
        if len(context) >= 2:
            trigram = context[-2:] + (word,)
            trigram_prob = self.trigram_model.get_probability(trigram)
        else:
            trigram_prob = 0.0
        
        if len(context) >= 1:
            bigram = context[-1:] + (word,)
            bigram_prob = self.bigram_model.get_probability(bigram)
        else:
            bigram_prob = 0.0
        
        unigram_prob = self.unigram_model.get_probability((word,))
        
        interpolated_prob = (self.lambdas[0] * unigram_prob + 
                           self.lambdas[1] * bigram_prob + 
                           self.lambdas[2] * trigram_prob)
        
        return interpolated_prob
    
    def calculate_perplexity(self, test_corpus: List[str]) -> float:
        log_prob_sum = 0
        total_words = 0
        
        for sentence in test_corpus:
            tokens = self.unigram_model.preprocess_text(sentence)
            
            for i in range(2, len(tokens)):
                word = tokens[i]
                context = tuple(tokens[max(0, i-2):i])
                
                prob = self.get_probability(word, context)
                if prob == 0:
                    return float('inf')
                
                log_prob_sum += math.log(prob)
                total_words += 1
        
        if total_words == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / total_words
        perplexity = math.exp(-avg_log_prob)
        return perplexity

class StupidBackoffModel:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.unigram_model = NGramLanguageModel(1)
        self.bigram_model = NGramLanguageModel(2)
        self.trigram_model = NGramLanguageModel(3)
        self.vocab = set()
    
    def train(self, corpus: List[str]):
        self.unigram_model.train(corpus)
        self.bigram_model.train(corpus)
        self.trigram_model.train(corpus)
        
        for sentence in corpus:
            tokens = self.unigram_model.preprocess_text(sentence)
            self.vocab.update(tokens)
    
    def get_score(self, word: str, context: Tuple[str, ...]) -> float:
        if len(context) >= 2:
            trigram = context[-2:] + (word,)
            if self.trigram_model.ngram_counts[trigram] > 0:
                trigram_context = trigram[:-1]
                return self.trigram_model.ngram_counts[trigram] / self.trigram_model.context_counts[trigram_context]
        
        if len(context) >= 1:
            bigram = context[-1:] + (word,)
            if self.bigram_model.ngram_counts[bigram] > 0:
                bigram_context = bigram[:-1]
                return self.alpha * (self.bigram_model.ngram_counts[bigram] / self.bigram_model.context_counts[bigram_context])
        
        return self.alpha * self.alpha * (self.unigram_model.ngram_counts[(word,)] / sum(self.unigram_model.ngram_counts.values()))
    
    def calculate_perplexity(self, test_corpus: List[str]) -> float:
        log_score_sum = 0
        total_words = 0
        
        for sentence in test_corpus:
            tokens = self.unigram_model.preprocess_text(sentence)
            
            for i in range(2, len(tokens)):
                word = tokens[i]
                context = tuple(tokens[max(0, i-2):i])
                
                score = self.get_score(word, context)
                if score == 0:
                    return float('inf')
                
                log_score_sum += math.log(score)
                total_words += 1
        
        if total_words == 0:
            return float('inf')
        
        avg_log_score = log_score_sum / total_words
        perplexity = math.exp(-avg_log_score)
        return perplexity

def optimize_interpolation_weights(train_corpus: List[str], dev_corpus: List[str]) -> Tuple[List[float], float]:
    best_lambdas = None
    best_perplexity = float('inf')
    
    lambda_combinations = [
        [0.1, 0.3, 0.6],
        [0.2, 0.3, 0.5],
        [0.3, 0.3, 0.4],
        [0.4, 0.3, 0.3],
        [0.5, 0.3, 0.2]
    ]
    
    for lambdas in lambda_combinations:
        model = LinearInterpolationModel(lambdas)
        model.train(train_corpus)
        perplexity = model.calculate_perplexity(dev_corpus)
        
        print(f"Lambdas {lambdas}: Perplexity = {perplexity:.2f}")
        
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_lambdas = lambdas
    
    return best_lambdas, best_perplexity

class TextGenerator:
    def __init__(self, model):
        self.model = model
    
    def generate_sentence(self, max_length: int = 20) -> str:
        if isinstance(self.model, StupidBackoffModel):
            return self._generate_backoff_sentence(max_length)
        elif isinstance(self.model, LinearInterpolationModel):
            return self._generate_interpolated_sentence(max_length)
        elif isinstance(self.model, NGramLanguageModel):
            return self._generate_ngram_sentence(max_length)
    
    def _generate_backoff_sentence(self, max_length: int) -> str:
        context = ['<s>', '<s>']
        words = []
        
        for _ in range(max_length):
            possible_words = []
            context_tuple = tuple(context[-2:])
            
            for word in self.model.vocab:
                if word not in ['<s>', '</s>']:
                    score = self.model.get_score(word, context_tuple)
                    if score > 0:
                        possible_words.append((word, score))
            
            if not possible_words:
                break
            
            possible_words.sort(key=lambda x: x[1], reverse=True)
            next_word = self._sample_from_distribution(possible_words[:100])
            
            if next_word == '</s>':
                break
            
            words.append(next_word)
            context.append(next_word)
        
        return ' '.join(words)
    
    def _generate_interpolated_sentence(self, max_length: int) -> str:
        context = ['<s>', '<s>']
        words = []
        
        for _ in range(max_length):
            possible_words = []
            context_tuple = tuple(context[-2:])
            
            for word in self.model.vocab:
                if word not in ['<s>', '</s>']:
                    prob = self.model.get_probability(word, context_tuple)
                    if prob > 0:
                        possible_words.append((word, prob))
            
            if not possible_words:
                break
            
            possible_words.sort(key=lambda x: x[1], reverse=True)
            next_word = self._sample_from_distribution(possible_words[:100])
            
            if next_word == '</s>':
                break
            
            words.append(next_word)
            context.append(next_word)
        
        return ' '.join(words)
    
    def _generate_ngram_sentence(self, max_length: int) -> str:
        if self.model.n == 1:
            words = []
            for _ in range(max_length):
                word_probs = []
                for word in self.model.vocab:
                    if word not in ['<s>', '</s>']:
                        prob = self.model.get_probability((word,))
                        word_probs.append((word, prob))
                
                if not word_probs:
                    break
                
                word_probs.sort(key=lambda x: x[1], reverse=True)
                next_word = self._sample_from_distribution(word_probs[:100])
                words.append(next_word)
                
                if next_word == '</s>':
                    break
            
            return ' '.join(words).replace('</s>', '').strip()
        
        else:
            context = ['<s>'] * (self.model.n - 1)
            words = []
            
            for _ in range(max_length):
                possible_words = []
                context_tuple = tuple(context[-(self.model.n-1):])
                
                for word in self.model.vocab:
                    if word != '<s>':
                        ngram = context_tuple + (word,)
                        prob = self.model.get_probability(ngram)
                        if prob > 0:
                            possible_words.append((word, prob))
                
                if not possible_words:
                    break
                
                possible_words.sort(key=lambda x: x[1], reverse=True)
                next_word = self._sample_from_distribution(possible_words[:100])
                
                if next_word == '</s>':
                    break
                
                words.append(next_word)
                context.append(next_word)
            
            return ' '.join(words)
    
    def _sample_from_distribution(self, word_probs: List[Tuple[str, float]]) -> str:
        if not word_probs:
            return '</s>'
        
        total_prob = sum(prob for _, prob in word_probs)
        if total_prob == 0:
            return word_probs[0][0]
        
        normalized_probs = [(word, prob / total_prob) for word, prob in word_probs]
        
        r = random.random()
        cumulative = 0
        for word, prob in normalized_probs:
            cumulative += prob
            if r <= cumulative:
                return word
        
        return normalized_probs[-1][0]

def load_data(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    train_corpus = load_data('ptbdataset/ptb.train.txt')
    dev_corpus = load_data('ptbdataset/ptb.valid.txt')
    test_corpus = load_data('ptbdataset/ptb.test.txt')
    
    print("Training and evaluating N-gram models...")
    print("=" * 50)
    
    results = {}
    
    for n in [1, 2, 3, 4]:
        print(f"\nTraining {n}-gram model (MLE)...")
        model = NGramLanguageModel(n)
        model.train(train_corpus)
        perplexity = model.calculate_perplexity(test_corpus)
        results[f'{n}-gram MLE'] = perplexity
        print(f"{n}-gram MLE Perplexity: {perplexity:.2f}")
    
    print(f"\nTraining Trigram with Add-1 Smoothing...")
    add1_model = AddOneSmoothedNGramModel(3)
    add1_model.train(train_corpus)
    add1_perplexity = add1_model.calculate_perplexity(test_corpus)
    results['Trigram Add-1'] = add1_perplexity
    print(f"Trigram Add-1 Perplexity: {add1_perplexity:.2f}")
    
    print(f"\nOptimizing Linear Interpolation weights...")
    best_lambdas, dev_perplexity = optimize_interpolation_weights(train_corpus, dev_corpus)
    print(f"Best lambdas: {best_lambdas}, Dev Perplexity: {dev_perplexity:.2f}")
    
    interpolation_model = LinearInterpolationModel(best_lambdas)
    interpolation_model.train(train_corpus)
    interpolation_perplexity = interpolation_model.calculate_perplexity(test_corpus)
    results['Linear Interpolation'] = interpolation_perplexity
    print(f"Linear Interpolation Test Perplexity: {interpolation_perplexity:.2f}")
    
    print(f"\nTraining Stupid Backoff model...")
    backoff_model = StupidBackoffModel(alpha=0.4)
    backoff_model.train(train_corpus)
    backoff_perplexity = backoff_model.calculate_perplexity(test_corpus)
    results['Stupid Backoff'] = backoff_perplexity
    print(f"Stupid Backoff Perplexity: {backoff_perplexity:.2f}")
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)
    for model_name, perplexity in results.items():
        print(f"{model_name:20s}: {perplexity:.2f}")
    
    best_model_name = min(results.keys(), key=lambda k: results[k])
    print(f"\nBest model: {best_model_name} (Perplexity: {results[best_model_name]:.2f})")
    
    if best_model_name == 'Stupid Backoff':
        best_model = backoff_model
    elif best_model_name == 'Linear Interpolation':
        best_model = interpolation_model
    elif best_model_name == 'Trigram Add-1':
        best_model = add1_model
    else:
        n = int(best_model_name.split('-')[0])
        best_model = NGramLanguageModel(n)
        best_model.train(train_corpus)
    
    print(f"\nGenerating text with best model ({best_model_name}):")
    print("=" * 50)
    generator = TextGenerator(best_model)
    for i in range(5):
        sentence = generator.generate_sentence()
        print(f"{i+1}. {sentence}")

if __name__ == "__main__":
    main()
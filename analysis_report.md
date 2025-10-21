# N-Gram Language Modeling and Evaluation - Analysis Report

## Overview
This report presents the implementation and evaluation of various N-gram language models trained on the Penn Treebank dataset. The models include Maximum Likelihood Estimation (MLE) N-grams, Add-1 smoothing, Linear Interpolation, and Stupid Backoff techniques.

## Dataset and Preprocessing

### Dataset Statistics
- **Training Data**: 42,068 sentences
- **Validation Data**: 3,370 sentences  
- **Test Data**: 3,761 sentences
- **Source**: Penn Treebank dataset from Kaggle

### Preprocessing Strategy
The preprocessing pipeline includes:
1. **Tokenization**: Simple whitespace-based tokenization
2. **Lowercasing**: All text converted to lowercase for consistency
3. **Sentence Boundaries**: Added `<s>` start tokens and `</s>` end tokens for N > 1
4. **Unknown Words**: The dataset already contains `<unk>` tokens for out-of-vocabulary words
5. **Numbers**: Normalized with `N` tokens in the original dataset

## Experimental Results

### Model Performance Summary

| Model                | Perplexity | Status |
|---------------------|------------|---------|
| 1-gram MLE          | 716.90     | ✓ |
| 2-gram MLE          | ∞ (inf)    | Zero probabilities |
| 3-gram MLE          | ∞ (inf)    | Zero probabilities |
| 4-gram MLE          | ∞ (inf)    | Zero probabilities |
| Trigram Add-1       | 3308.23    | ✓ |
| Linear Interpolation| 217.17     | ✓ |
| **Stupid Backoff**  | **214.40** | **Best Model** |

### Linear Interpolation Optimization
The linear interpolation weights were optimized on the validation set:

| λ₁ (Unigram) | λ₂ (Bigram) | λ₃ (Trigram) | Dev Perplexity |
|--------------|-------------|--------------|----------------|
| 0.1          | 0.3         | 0.6          | 294.98         |
| 0.2          | 0.3         | 0.5          | 253.83         |
| 0.3          | 0.3         | 0.4          | 236.55         |
| **0.4**      | **0.3**     | **0.3**      | **230.02**     |
| 0.5          | 0.3         | 0.2          | 231.83         |

**Optimal weights**: λ₁=0.4, λ₂=0.3, λ₃=0.3

## Analysis and Discussion

### 4.1 Pre-processing and Vocabulary Decisions

Our preprocessing strategy was designed to balance simplicity with effectiveness for N-gram modeling:

**Tokenization Strategy**:
- **Whitespace-based tokenization**: We used simple `text.split()` which works well for the Penn Treebank dataset since it's already well-formatted
- **Lowercasing**: All text converted to lowercase (`text.lower()`) to reduce vocabulary size and treat "The" and "the" as the same token
- **No additional normalization**: We preserved punctuation and special characters as they appeared in the original dataset

**Sentence Boundary Handling**:
- **Start tokens**: Added `<s>` tokens at sentence beginning (N-1 tokens for N-gram models where N>1)
- **End tokens**: Added `</s>` token at sentence end to properly model sentence termination
- **Context padding**: This ensures the first and last words have proper context for probability estimation

**Vocabulary Decisions**:
- **Unknown word handling**: The dataset already contained `<unk>` tokens for out-of-vocabulary words, which we preserved
- **Number normalization**: The dataset used `N` tokens for numbers, which we maintained
- **No vocabulary pruning**: We kept all tokens that appeared in the training data
- **Vocabulary size**: Our final vocabulary contained all unique tokens seen during training

**Rationale**:
This preprocessing approach prioritizes consistency with the dataset's existing format while ensuring that N-gram contexts are properly defined. The sentence boundary markers are crucial for N>1 models to avoid cross-sentence dependencies that would be linguistically invalid.

### 4.2 Impact of N-gram Order

The results clearly demonstrate the **curse of dimensionality** in language modeling:

- **Unigram (N=1)**: Achieves finite perplexity (716.90) but lacks contextual information
- **Higher-order N-grams (N=2,3,4)**: Suffer from severe data sparsity, resulting in infinite perplexity due to zero probabilities for unseen N-grams

This phenomenon illustrates the fundamental trade-off in N-gram modeling:
- **Lower N**: Better coverage but less context
- **Higher N**: More context but severe sparsity issues

The **Markov Assumption** becomes increasingly problematic as N increases because:
1. The number of possible N-gram combinations grows exponentially
2. Even large corpora like Penn Treebank cannot provide sufficient coverage
3. Many valid N-gram sequences in the test set were never observed during training

### 4.2 Comparison of Smoothing/Backoff Strategies

#### Why Unsmoothed Models Fail
The infinite perplexity scores for unsmoothed N-gram models (N>1) occur because:
1. **Zero Probability Problem**: When the model encounters an unseen N-gram during testing, it assigns probability 0
2. **Perplexity Calculation**: Since perplexity involves taking the logarithm of probabilities, log(0) = -∞, leading to infinite perplexity
3. **Data Sparsity**: Even with 42K training sentences, the coverage of possible bigrams, trigrams, and 4-grams is insufficient

#### Smoothing Effectiveness

**Add-1 (Laplace) Smoothing**:
- **Perplexity**: 3308.23
- **Pros**: Eliminates zero probabilities by adding 1 to all counts
- **Cons**: Overly aggressive smoothing that assigns too much probability mass to unseen events
- **Result**: High perplexity due to poor probability distribution

**Linear Interpolation**:
- **Perplexity**: 217.17
- **Strategy**: Combines unigram, bigram, and trigram probabilities with weighted average
- **Advantage**: Gracefully falls back to lower-order models when higher-order context is unavailable
- **Optimal weights favor unigrams** (λ₁=0.4), suggesting the importance of word frequency information

**Stupid Backoff**:
- **Perplexity**: 214.40 (Best performance)
- **Strategy**: Uses highest-order available N-gram, backing off with penalty factor α=0.4
- **Alpha Choice**: We used α=0.4, a common choice recommended in the literature (Brants et al., 2007), rather than optimizing on dev data due to its proven effectiveness
- **Advantage**: More aggressive use of available context while maintaining smoothness
- **Success factors**: 
  - Preserves the maximum available context
  - Simple and effective backoff mechanism
  - Less parameter sensitivity than interpolation

### 4.3 Best Model Analysis

**Stupid Backoff emerged as the best model** because:
1. **Context Preservation**: Uses trigram information when available
2. **Graceful Degradation**: Backs off systematically to bigrams then unigrams
3. **Computational Efficiency**: Simpler than interpolation methods
4. **Practical Effectiveness**: Well-suited for the data sparsity characteristics of the Penn Treebank

## 4.4 Qualitative Analysis - Generated Text

### Generated Sentences (Stupid Backoff Model):

1. "but mr. bush 's joining what they know the daily tribune in the case by the new york court 's"
2. "like lebanon and the <unk> also are lots of equipment to make a transaction work with p&g 's international signal"
3. "the $ 300-a-share bid if they are <unk> to mr. cray founded their computer company said export <unk> is scheduled"
4. "it 's not that means goods could be a fairly soft year in addition to <unk> him exclusive spokesman for"
5. "and the national association of manufacturers hanover said that to its houston natural gas industry how to use as supplier"

### Fluency Assessment

**Strengths**:
- **Syntactic Structure**: Sentences generally follow English grammatical patterns
- **Domain Coherence**: Content reflects the business/financial domain of Penn Treebank
- **Local Coherence**: Short phrases are often semantically reasonable
- **Vocabulary**: Uses appropriate business terminology and proper nouns

**Limitations**:
- **Long-range Dependencies**: Sentences lose coherence over longer spans
- **Semantic Consistency**: Global meaning often unclear or contradictory
- **Unknown Token Handling**: Frequent `<unk>` tokens disrupt readability
- **Discourse Flow**: Lacks overall narrative or argumentative structure

### Generation Mechanism

The text generation process works by:
1. **Context Initialization**: Starting with `<s> <s>` tokens
2. **Probabilistic Sampling**: At each step, computing scores for all vocabulary words given current context
3. **Weighted Selection**: Sampling from top-100 candidates based on Stupid Backoff scores
4. **Context Update**: Appending selected word and shifting context window
5. **Termination**: Stopping at `</s>` token or maximum length

The **Markov assumption** limits the model's ability to maintain long-range semantic coherence, but local fluency is achieved through learned patterns from the training corpus.

## Conclusions

1. **N-gram Order**: Higher-order N-grams require sophisticated smoothing techniques to handle data sparsity
2. **Smoothing Necessity**: Unsmoothed models are impractical for real applications due to zero probability issues
3. **Backoff vs. Interpolation**: Stupid Backoff slightly outperformed Linear Interpolation, suggesting that preserving maximum context is beneficial
4. **Text Generation**: While locally fluent, generated text lacks global coherence due to the limited context window of N-gram models
5. **Practical Performance**: Perplexity around 214 represents reasonable performance for a trigram model on this dataset

The results demonstrate the fundamental limitations of N-gram models and highlight why modern neural language models have largely superseded them for most applications.
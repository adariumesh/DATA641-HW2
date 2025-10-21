# N-Gram Language Modeling and Evaluation

This project implements and evaluates various N-gram language models on the Penn Treebank dataset as part of DATA641 Homework 2.

## Project Structure

```
HW2/
├── ngram_models.py           # Main implementation file
├── analysis_report.md        # Detailed analysis and results
├── README.md                # This file
└── ptbdataset/              # Penn Treebank dataset
    ├── ptb.train.txt        # Training data (42,068 sentences)
    ├── ptb.valid.txt        # Validation data (3,370 sentences)
    └── ptb.test.txt         # Test data (3,761 sentences)
```

## Implementation Overview

The project implements the following models:
1. **Maximum Likelihood Estimation (MLE) N-grams** (N=1,2,3,4)
2. **Add-1 (Laplace) Smoothing** for trigrams
3. **Linear Interpolation** with automatic weight optimization
4. **Stupid Backoff** algorithm
5. **Text Generation** using the best performing model

## Requirements

- Python 3.7+
- Standard library modules only (no external dependencies)

## How to Run

### Basic Execution
```bash
python ngram_models.py
```

This will:
1. Load the Penn Treebank dataset
2. Train all N-gram models (MLE for N=1,2,3,4)
3. Train Add-1 smoothed trigram model
4. Optimize linear interpolation weights on validation data
5. Train Stupid Backoff model
6. Evaluate all models on test data using perplexity
7. Generate 5 sample sentences using the best model

### Expected Output

The program will display:
- Training progress for each model
- Perplexity scores on test data
- Linear interpolation weight optimization results
- Final performance comparison
- Generated text samples

Example output:
```
Training and evaluating N-gram models...
==================================================

Training 1-gram model (MLE)...
1-gram MLE Perplexity: 716.90

Training 2-gram model (MLE)...
2-gram MLE Perplexity: inf

...

==================================================
FINAL RESULTS:
==================================================
1-gram MLE          : 716.90
2-gram MLE          : inf
3-gram MLE          : inf
4-gram MLE          : inf
Trigram Add-1       : 3308.23
Linear Interpolation: 217.17
Stupid Backoff      : 214.40

Best model: Stupid Backoff (Perplexity: 214.40)

Generating text with best model (Stupid Backoff):
==================================================
1. but mr. bush 's joining what they know the daily tribune...
2. like lebanon and the <unk> also are lots of equipment...
...
```

## Code Architecture

### Key Classes

1. **`NGramLanguageModel`**: Base class for MLE N-gram models
   - Handles tokenization and preprocessing
   - Computes N-gram and context counts
   - Calculates probabilities and perplexity

2. **`AddOneSmoothedNGramModel`**: Extends base class with Add-1 smoothing
   - Modifies probability calculation to add 1 to all counts

3. **`LinearInterpolationModel`**: Implements linear interpolation
   - Combines unigram, bigram, and trigram probabilities
   - Supports automatic weight optimization

4. **`StupidBackoffModel`**: Implements Stupid Backoff algorithm
   - Uses highest-order available N-gram with backoff penalty

5. **`TextGenerator`**: Generates text using trained models
   - Supports all model types
   - Uses probabilistic sampling for word selection

### Key Functions

- **`load_data()`**: Loads text files and returns list of sentences
- **`optimize_interpolation_weights()`**: Finds optimal λ values for interpolation
- **`main()`**: Orchestrates the entire evaluation pipeline

## Model Details

### Preprocessing
- Lowercase normalization
- Sentence boundary markers (`<s>`, `</s>`)
- Whitespace tokenization
- Preserves existing `<unk>` tokens

### Evaluation Metric
All models are evaluated using **perplexity** on the test set:
```
Perplexity = exp(-1/N * Σ log P(w_i | context))
```

### Text Generation
The text generator uses:
- Probabilistic sampling from top-K candidates (K=100)
- Context window management
- Proper handling of sentence boundaries

## Results Summary

| Model | Perplexity | Notes |
|-------|------------|-------|
| Unigram MLE | 716.90 | No context information |
| Bigram+ MLE | ∞ | Zero probability issues |
| Add-1 Trigram | 3308.23 | Over-smoothing |
| Linear Interpolation | 217.17 | Optimal λ=[0.4,0.3,0.3] |
| **Stupid Backoff** | **214.40** | **Best performance** |

## Analysis

See `analysis_report.md` for detailed analysis including:
- Impact of N-gram order on performance
- Comparison of smoothing techniques
- Qualitative analysis of generated text
- Discussion of data sparsity issues

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure the `ptbdataset/` directory and files exist
2. **Memory issues**: The code is optimized for efficiency but large vocabularies may require more RAM
3. **Infinite perplexity**: This is expected for unsmoothed models with N>1

### Performance Notes

- Training time: ~30-60 seconds on modern hardware
- Memory usage: ~500MB peak for all models
- The code includes progress indicators for long-running operations

## Academic Integrity

This implementation is for educational purposes as part of DATA641 coursework. The code demonstrates fundamental concepts in statistical language modeling and should be understood conceptually rather than copied.

## References

- Penn Treebank Dataset: https://www.kaggle.com/datasets/aliakay8/penn-treebank-dataset
- Stupid Backoff: Brants et al. (2007)
- Linear Interpolation: Jelinek & Mercer (1980)
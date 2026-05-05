# Meta-Learning Implementation with MAML

A comprehensive implementation of Model-Agnostic Meta-Learning (MAML) for few-shot learning, featuring strong baselines, advanced meta-learning algorithms, and interactive demonstrations.

## ⚠️ Safety & Ethics Disclaimer

**This is a research and educational project. NOT FOR PRODUCTION USE.**

- This implementation is for research and educational purposes only
- Results should not be used for critical decision-making without human oversight
- Meta-learning models may exhibit unexpected behaviors on out-of-distribution data
- Always validate results with domain experts before applying to real-world scenarios
- Consider ethical implications when applying meta-learning to sensitive domains

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Meta-Learning-Implementation-with-MAML.git
cd Meta-Learning-Implementation-with-MAML

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Train MAML on few-shot classification
python scripts/train.py --config configs/maml_mnist.yaml

# Evaluate on test tasks
python scripts/evaluate.py --config configs/maml_mnist.yaml --checkpoint checkpoints/best_model.pth

# Run interactive demo
streamlit run demo/app.py
```

## Dataset Schema

### MNIST Few-Shot Tasks
- **Format**: 28x28 grayscale images
- **Classes**: 10 digits (0-9)
- **Task Structure**: N-way K-shot classification
- **Splits**: Train (50k), Val (5k), Test (5k)
- **License**: Public domain

### Custom Dataset Support
- CSV/JSON metadata files
- Image folders organized by class
- Automatic few-shot task generation

## Architecture

### Models Implemented

#### Baselines
- **Logistic Regression**: Simple linear baseline
- **Simple CNN**: Basic convolutional network
- **ResNet-18**: Standard deep learning baseline

#### Meta-Learning Methods
- **MAML**: Model-Agnostic Meta-Learning (primary)
- **MAML++**: Improved MAML with better optimization
- **ProtoNet**: Prototypical Networks for comparison
- **Matching Networks**: Memory-augmented approach

### Training Pipeline

1. **Task Sampling**: Generate N-way K-shot tasks from dataset
2. **Meta-Training**: Train on multiple tasks simultaneously
3. **Meta-Validation**: Evaluate adaptation speed on held-out tasks
4. **Meta-Testing**: Final evaluation on unseen task distributions

## Evaluation Metrics

### Few-Shot Learning Metrics
- **Accuracy**: Classification accuracy after adaptation
- **Adaptation Speed**: Performance vs. gradient steps
- **Generalization**: Performance on unseen task distributions
- **Sample Efficiency**: Performance vs. number of support examples

### Comparative Analysis
- **Baseline Comparison**: MAML vs. standard supervised learning
- **Ablation Studies**: Effect of inner loop steps, learning rates, etc.
- **Task Difficulty**: Performance across different task complexities

## Expected Performance

### MNIST 5-way Classification
- **1-shot**: ~85-90% accuracy
- **5-shot**: ~95-98% accuracy
- **10-shot**: ~98-99% accuracy

*Note: Results may vary based on hyperparameters and random seeds*

## Interactive Demo

The Streamlit demo (`demo/app.py`) provides:

- **Task Visualization**: View support and query examples
- **Adaptation Process**: Watch model adapt in real-time
- **Performance Comparison**: Compare different meta-learning methods
- **Parameter Tuning**: Adjust hyperparameters interactively

## Configuration

Configuration files in `configs/` control:

- **Model Architecture**: Network depth, width, activation functions
- **Training Parameters**: Learning rates, batch sizes, epochs
- **Task Generation**: N-way K-shot settings, task sampling
- **Evaluation**: Metrics, logging, checkpointing

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model definitions
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── train/             # Training loops
│   ├── eval/              # Evaluation scripts
│   ├── viz/               # Visualization utilities
│   └── utils/             # General utilities
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── assets/                # Generated plots and results
├── tests/                 # Unit tests
├── scripts/               # Training and evaluation scripts
├── demo/                  # Interactive demonstrations
└── notebooks/             # Jupyter notebooks for analysis
```

## Development

### Code Quality
```bash
# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ scripts/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## References

- Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks.
- Antoniou, A., et al. (2019). MAML++: Better meta-learning algorithms.
- Snell, J., et al. (2017). Prototypical networks for few-shot learning.

## Author

**kryptologyst**  
GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

## License

MIT License - see LICENSE file for details.

---

*This project is part of the 1000 AI Projects series, focusing on meta-learning and few-shot adaptation techniques.*
# Meta-Learning-Implementation-with-MAML

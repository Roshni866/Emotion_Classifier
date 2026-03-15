# Emotion Classification using Deep Learning

A comprehensive deep learning project that compares multiple neural network architectures (CNN, ANN, RNN/LSTM) for text-based emotion classification.

## 📋 Project Overview

This project implements and evaluates three different deep learning models to classify text comments into emotion categories (Joy, Fear, Anger). The project includes complete data exploration, preprocessing, model development, training, and comparative performance analysis.

**Key Achievement:** Achieved 96.14% accuracy with CNN architecture on binary classification (Joy vs Fear) and 92% accuracy with RNN/LSTM on multi-class classification.

## 🎯 Objectives

- Perform exploratory data analysis on emotion classification dataset
- Implement multiple neural network architectures for text classification
- Compare model performance across different architectures
- Develop preprocessing pipeline for text data
- Build Word2Vec embeddings for semantic text representation

## 📊 Dataset

- **Source:** Emotion Classification Dataset (`Emotion_classify_Data.csv`)
- **Total Samples:** ~5,937 comments
- **Emotion Categories:** Anger, Joy, Fear (with mixed multi-class implementation)
- **Features:** Text comments with emotion labels
- **Train-Test Split:** 70-30 split with stratification

### Data Statistics

| Metric | Value |
|--------|-------|
| Total Comments | 5,937 |
| Joy | 2,000 |
| Fear | 1,937 |
| Anger | 2,000 |
| Avg. Comment Length | ~14 words |
| Max Word Count | 60+ words |

## 🔧 Technologies & Libraries

```
Python 3.12
TensorFlow/Keras - Deep Learning Framework
scikit-learn - Machine Learning utilities
pandas - Data manipulation
NumPy - Numerical computing
Matplotlib & Seaborn - Data visualization
NLTK - Natural Language Processing
Gensim - Word2Vec embeddings
```

## 📈 Models Implemented

### 1. **Convolutional Neural Network (CNN)**
- **Architecture:** Embedding → Conv1D (2 layers) → MaxPooling → Flatten → Dense layers
- **Parameters:** 128 filters, kernel size 5, embedding dimension 200
- **Test Accuracy:** 96.14%
- **Best for:** Feature extraction from local text patterns

### 2. **Artificial Neural Network (ANN)**
- **Architecture:** Embedding → Flatten → Dense layers (256 → 128 → 64 neurons)
- **Parameters:** ReLU activation, Sigmoid output for binary classification
- **Test Accuracy:** 94.01%
- **Best for:** Baseline architecture with lower computational cost

### 3. **Recurrent Neural Network (RNN/LSTM)**
- **Architecture:** Embedding → LSTM (32 units) → GlobalMaxPool → Dense
- **Parameters:** Embedding dimension 50, 3 categories output
- **Test Accuracy:** 92% (Multi-class, 3 emotions)
- **Best for:** Sequential pattern learning and long-range dependencies

## 🚀 Pipeline Overview

### 1. Data Exploration
- Distribution analysis of emotion labels
- Word and character count distributions
- Average word length analysis
- Top 20 most common words visualization

### 2. Data Preprocessing
- Special character removal using regex
- Text lowercasing
- Tokenization with NLTK
- Stopword removal
- Sequence padding (max_length = 100)

### 3. Feature Engineering
- Tokenization with Keras Tokenizer
- Word2Vec embeddings (200-dimensional vectors)
- Sequence padding to fixed length
- Vocabulary size: 7,273 unique words

### 4. Model Training
- Epochs: 30
- Batch size: 32
- Optimizer: Adam
- Loss functions: Binary/Sparse Categorical Crossentropy
- Validation split: 30% test data

## 📊 Results & Performance

### CNN Model Performance
```
Binary Classification (Joy vs Fear)
Test Accuracy: 96.14%
Final Training Accuracy: 100%
Convergence: Rapid (5-6 epochs)
```

### ANN Model Performance
```
Binary Classification (Joy vs Fear)
Test Accuracy: 94.01%
Final Training Accuracy: 100%
Best validation accuracy: 94.52%
```

### RNN/LSTM Model Performance
```
Multi-class Classification (3 emotions)
Test Accuracy: 92%
Per-class metrics:
  - Anger (Class 0): Precision 0.91, Recall 0.92, F1 0.91
  - Fear (Class 1): Precision 0.92, Recall 0.92, F1 0.92
  - Joy (Class 2): Precision 0.93, Recall 0.93, F1 0.93
```

## 📁 Project Structure

```
emotion-classification/
├── README.md
├── dataset/
│   └── Emotion_classify_Data.csv
├── notebooks/
│   └── emotion_classification.ipynb
├── images/
│   ├
│   │── emotion_distribution.png
│   │── cnn_model_accuracy_plot.png
│   │── ann_model_loss_plot.png
│   │── rnn_model_accuracy_plot.png
│   ├── cnn_model.h5
│   ├── ann_model.h5
│   └── rnn_model.h5
└── requirements.txt
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- pip or conda

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/emotion-classification.git
cd emotion-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 📚 Usage

### Running the Complete Pipeline

```python
# Load and preprocess data
import pandas as pd
from src.data_preprocessing import preprocess_text

df = pd.read_csv('dataset/Emotion_classify_Data.csv')
df['processed_comment'] = df['Comment'].apply(preprocess_text)

# Train CNN model
from src.models import build_cnn_model
model = build_cnn_model(vocab_size=7273, embedding_dim=200)
history = model.fit(X_train, y_train, epochs=30, batch_size=32)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
```

### Making Predictions

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Preprocess input text
test_text = "I feel very happy today"
processed = preprocess_text(test_text)
sequence = tokenizer.texts_to_sequences([processed])
padded = pad_sequences(sequence, maxlen=100)

# Predict emotion
prediction = model.predict(padded)
emotion = ['Fear', 'Joy'][int(prediction[0] > 0.5)]
print(f"Predicted Emotion: {emotion}")
```

## 🔍 Key Findings

1. **CNN superiority for text:** CNN achieved 96.14% accuracy, suggesting convolutional layers effectively capture local contextual patterns in text.

2. **Quick convergence:** All models converged within 5-6 epochs, indicating good data-to-model fit.

3. **Low overfitting:** Test accuracy remained high despite 100% training accuracy after epoch 5, suggesting good generalization.

4. **Class balance matters:** Despite initial class imbalance, stratified splitting and LSTM with 3 classes achieved balanced per-class performance (91-93%).

5. **Embedding quality:** Word2Vec embeddings with 200 dimensions provided sufficient semantic representation for emotion classification.

## 💡 Model Comparison

| Metric | CNN | ANN | RNN/LSTM |
|--------|-----|-----|----------|
| Test Accuracy | 96.14% | 94.01% | 92% |
| Convergence Speed | Fast | Fast | Medium |
| Computational Cost | Medium | Low | High |
| Sequence Understanding | Good | Fair | Excellent |
| Best Use Case | Local patterns | Baseline | Long dependencies |

## 🎓 Learning Outcomes

- **Deep Learning Fundamentals:** Implemented and compared multiple architectures
- **Text Processing:** Built complete NLP pipeline with tokenization, embedding, and padding
- **Model Evaluation:** Used stratified splits and multiple metrics for robust evaluation
- **Visualization:** Created comprehensive training curves and performance plots
- **Best Practices:** Applied modern deep learning conventions (padding, embedding, dropout considerations)

## 📝 Hyperparameter Details

```
Tokenization:
  - num_words: 2000 (RNN), vocabulary size: 7273 unique
  - max_length: 100 padding

Embedding:
  - embedding_dim: 200 (CNN/ANN), 50 (RNN)
  - trainable: True

CNN:
  - conv1d_filters: 128, kernel_size: 5
  - pool_size: 5
  - dense_units: [256, 128]

ANN:
  - dense_units: [256, 128, 64]

RNN:
  - lstm_units: 32
  - embedding_dim: 50
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.




## 📚 References

1. [TensorFlow Documentation](https://www.tensorflow.org/)
2. [Keras API Reference](https://keras.io/)
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
4. [Word2Vec Paper](https://arxiv.org/pdf/1301.3781.pdf)
5. [Text Classification with CNNs](https://arxiv.org/pdf/1408.5882.pdf)

---

**Last Updated:** March 15, 2026  


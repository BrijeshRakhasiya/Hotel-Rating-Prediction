# 🏨 TripAdvisor Hotel Review Rating Prediction

This project focuses on predicting hotel review ratings using NLP techniques on the [TripAdvisor Hotel Reviews Dataset](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews). We aim to classify user reviews into one of the 5 rating categories (1 to 5 stars) using machine learning and deep learning models.

---

## 📂 Dataset

- **Source:** [Kaggle - TripAdvisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)
- **Description:** This dataset contains 20,000 hotel reviews along with corresponding ratings from 1 to 5.
- **Columns:**
  - `Review`: Textual review of a hotel
  - `Rating`: Integer value from 1 to 5 representing the review score

---

## ⚙️ Project Workflow

###  🧹 Data Preprocessing
- Removed null and duplicate reviews
- Converted ratings to integers
- Applied text cleaning:
  - Lowercasing
  - Punctuation and stopword removal using **NLTK**
  - Lemmatization

###  🧠 Model Training & Evaluation

#### 🤖 Deep Learning Model (GRU-based)
| Model                 | Accuracy | AUC    | Loss   |
|----------------------|----------|--------|--------|
| Bidirectional GRU    | 77.36%   | 0.8564 | 0.4893 |

- **Text Vectorization:** `Tokenizer` + `pad_sequences`
- **Embedding Dimension:** 128
- **Model Architecture:**
  - Input Layer
  - Embedding Layer
  - Bidirectional GRU (128 units, return sequences)
  - Flatten Layer
  - Dense Output Layer (Sigmoid)

```python
embedding_dim = 128
inputs = tf.keras.Input(shape=(max_seq_length,))

embedding = tf.keras.layers.Embedding(
    input_dim=num_words,
    output_dim=embedding_dim,
    input_length=max_seq_length
)(inputs)

gru = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(128, return_sequences=True)
)(embedding)

flatten = tf.keras.layers.Flatten()(gru)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)

model = tf.keras.Model(inputs, outputs)
```

## 📌 Requirements

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow keras wordcloud
```


# 🙋‍♂️ Author
Brijesh Rakhasiya
AI/ML Enthusiast | Data Scientist | Problem Solver


## 📄 License

This project is licensed under the MIT License.

---
**Made ❤️ by Brijesh Rakhasiya**.

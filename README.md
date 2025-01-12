# Sentiment Analysis of Uber Reviews Using NLP Techniques

This project performs sentiment analysis on Uber reviews dataset using Natural Language Processing (NLP) techniques. The primary goal is to predict the sentiment (Positive, Neutral, or Negative) of user reviews based on the content of the review. We use machine learning models such as RandomForestClassifier to analyze and classify the reviews.

## Project Structure

- **Step 1: Importing Libraries**: The necessary libraries for data manipulation, visualization, and machine learning are imported.
- **Step 2: Data Loading and Exploration**: The dataset is loaded, and basic exploration is performed to understand the data.
- **Step 3: Data Cleaning**: Irrelevant columns are dropped, missing values are handled, and date columns are converted to datetime format.
- **Step 4: Data Preprocessing for NLP**: The review content is cleaned by removing non-alphabetic characters, converting to lowercase, and lemmatizing the text.
- **Step 5: Sentiment Labeling**: Sentiment labels (Positive, Neutral, Negative) are assigned based on the review score.
- **Step 6: Data Visualization**: Sentiment distribution is visualized using Seaborn and a Word Cloud is generated for positive reviews.
- **Step 7: Model Building**: A RandomForestClassifier model is trained on the processed data, evaluated, and performance is analyzed with classification metrics including confusion matrix.

## Libraries Used

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib**: For visualizations.
- **seaborn**: For statistical data visualization.
- **nltk**: For natural language processing tasks like tokenization, stopwords removal, and lemmatization.
- **spacy**: For advanced NLP tasks like tokenization and lemmatization.
- **scikit-learn**: For machine learning, including model building, evaluation, and feature extraction.
- **wordcloud**: For creating word clouds.

## Dataset

The dataset consists of Uber user reviews, with the following columns:
- `userName`: Name of the user.
- `content`: Review text content.
- `score`: Rating of the review (integer between 1 and 5).
- `thumbsUpCount`: Number of thumbs-up votes.
- `reviewCreatedVersion`: The app version when the review was created.
- `at`: Timestamp of the review.
- `appVersion`: The app version at the time of review.

## Steps Involved

### 1. Data Exploration and Cleaning

We load the dataset, inspect its contents, and clean it by:
- Dropping irrelevant columns (`userImage`, `replyContent`, `repliedAt`).
- Filling missing values in the review content column.
- Converting date columns to appropriate datetime format.

### 2. Data Preprocessing for NLP

The text data is preprocessed to improve the performance of the NLP model:
- Non-word characters are removed.
- The text is converted to lowercase.
- Tokenization is performed using SpaCy.
- Stopwords are removed, and lemmatization is applied.

### 3. Sentiment Labeling

Sentiments are assigned based on the review score:
- Reviews with a score of 4 or 5 are labeled "Positive".
- Reviews with a score of 3 are labeled "Neutral".
- Reviews with a score of 1 or 2 are labeled "Negative".

### 4. Data Visualization

- The sentiment distribution is visualized using a **countplot**.
- A **Word Cloud** for positive reviews is generated to visualize frequently used words.

### 5. Model Building

A **RandomForestClassifier** is trained on the preprocessed text data:
- Text data is converted to numerical form using TF-IDF vectorization.
- The dataset is split into training and testing sets.
- The model is evaluated using **accuracy score** and **classification report**.
- A **confusion matrix** is generated to assess the classification performance in more detail.

## Example Output

### Sentiment Distribution

The sentiment distribution of reviews is visualized with a **countplot**. This helps in understanding the balance between Positive, Negative, and Neutral reviews.

### Word Cloud for Positive Reviews

A **Word Cloud** is generated for the Positive reviews, which displays the most frequently mentioned words in a visually appealing way.

### Model Evaluation

The model’s performance is evaluated using the following metrics:
- **Classification Report**: Provides precision, recall, and F1-score for each class (Positive, Neutral, Negative).
- **Confusion Matrix**: Helps in understanding how well the model predicts each class.

Example Output:
```
              precision    recall  f1-score   support
    Negative       0.81      0.83      0.82       554
    Neutral        1.00      0.00      0.00        80
    Positive       0.93      0.97      0.95      1766

    accuracy                           0.90      2400
   macro avg       0.91      0.60      0.59      2400
weighted avg       0.91      0.90      0.89      2400
```

## How to Run the Project

1. Clone this repository to your local machine:
    ```
    git clone https://github.com/yourusername/uber-sentiment-analysis.git
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Place the `uber_reviews.csv` dataset in the project folder.

4. Run the code:
    ```
    python sentiment_analysis.py
    ```

## Conclusion

This project demonstrates how to analyze the sentiment of Uber reviews using machine learning and NLP techniques. It includes data preprocessing, model building, and evaluation, making it a comprehensive analysis pipeline for text-based sentiment classification.

---

Make sure to replace `yourusername` with your actual GitHub username in the cloning instructions. Let me know if you need further modifications!

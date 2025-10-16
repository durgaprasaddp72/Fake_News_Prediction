# Fake_News_Prediction
This project aims to detect fake news articles using Natural Language Processing (NLP) and supervised machine learning techniques. It demonstrates how text preprocessing, feature extraction, and classification models can be combined to build a robust fake news classifier.
📊 Problem Statement
With the rise of misinformation online, it's crucial to develop automated systems that can classify news articles as real or fake based on their content. This project uses labeled news data to train models that can make such predictions.
🧠 Techniques Used
- Text Preprocessing: Tokenization, stopword removal, stemming, and TF-IDF vectorization
- Classification Models: Logistic Regression, Naive Bayes, Random Forest
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curve
- Libraries: scikit-learn, pandas, NumPy, matplotlib, seaborn
📁 Dataset
- The dataset contains labeled news articles with binary classification: REAL or FAKE.
- Source: Kaggle Fake News Dataset
🚀 Results
- Achieved up to 96% accuracy using Logistic Regression and TF-IDF features
- Visualized model performance using confusion matrices and ROC curves
- Demonstrated the effectiveness of NLP in detecting misinformation
📦 How to Run
# Clone the repository
git clone https://github.com/durgaprasaddp72/Fake_News_Prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the main script
python fake_news_classifier.py


📌 Future Improvements
- Integrate deep learning models (e.g., LSTM, BERT) for contextual understanding
- Deploy the model using FastAPI or Streamlit for real-time predictions
- Add explainability using SHAP or LIME
📬 Contact
For questions or collaboration, reach out at durgaprasaddp72@gmail.com

Let me know if you’d like help writing the requirements.txt, adding a sample prediction script, or deploying this as a web app!





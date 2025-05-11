import joblib
import nltk
from nltk.corpus import stopwords
import string

# Скачиваем стоп-слова
#nltk.download('stopwords')

# Предобработка текста
def preprocess_text(text):
    stop_words = set(stopwords.words('russian'))
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Функция для предсказания тональности нового текста
def predict_sentiment(text):
    # Загрузка модели
    pipeline = joblib.load('sentiment_model.pkl')
    text = preprocess_text(text)
    probas = pipeline.predict_proba([text])[0]
    sentiment = pipeline.predict([text])[0]
    return sentiment, probas

def get_result_predict(text):
    sentiment, probas = predict_sentiment(text)
    print(f'Sentiment: {"Positive" if sentiment == 1 else "Negative"}')
    print(f'Probability: {probas[1] if sentiment == 1 else probas[0]}')

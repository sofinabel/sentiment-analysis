import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import joblib

# Скачиваем стоп-слова
nltk.download('stopwords')

n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
pos_df = pd.read_csv('positive (1).csv', sep=';', on_bad_lines='skip', names=n, usecols=['text'])
neg_df = pd.read_csv('negative (1).csv', sep=';', on_bad_lines='skip', names=n, usecols=['text'])

# Добавляем метки
neg_df['label'] = 0
pos_df['label'] = 1

# Объединяем данные
df = pd.concat([neg_df, pos_df])

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Предобработка текста
def preprocess_text(text):
    stop_words = set(stopwords.words('russian'))
    translator = str.maketrans('', '', string.punctuation) # Удаление пунктуации
    text = text.translate(translator)
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

print('Preprocessing texts...')
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Создание пайплайна
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Обучение модели
print('Starting training...')
pipeline.fit(X_train, y_train)

# Оценка модели
y_pred = pipeline.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Сохранение модели
joblib.dump(pipeline, 'sentiment_model.pkl')
print('Model saved!')


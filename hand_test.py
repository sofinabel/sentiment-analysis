import testingNB
import testingLSTM
import pandas as pd

tests = [line.encode('cp1251').decode('utf8') for line in open('test.txt')]
results = ["Positive", "Negative", "Positive", "Negative", "Negative", "Negative"]

df = pd.DataFrame({'text': tests})
def predict_sentiment_nb(text):
    sentiment, probability = testingNB.predict_sentiment(text)
    sentiment = "Positive" if sentiment == 1 else "Negative"
    probability = probability[0] if sentiment == "Negative" else probability[1]
    return sentiment + ' ' + str(round(probability, 4))

def predict_sentiment_lstm(text):
    probability = testingLSTM.predict_sentiment(text)
    sentiment = "Positive" if probability >= 0.5 else "Negative"
    probability = 1 - probability if sentiment == "Negative" else probability
    return sentiment + ' ' + str(round(probability, 4))


df['NB'] = df['text'].apply(predict_sentiment_nb)
df['LSTMN'] = df['text'].apply(predict_sentiment_lstm)
df['My_opinion'] = results

df.to_csv('output.csv', index=False)
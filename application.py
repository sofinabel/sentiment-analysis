from flask import Flask, request, render_template
import testingNB
import testingLSTM

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    probability = None
    model_type = 'nb'
    if request.method == 'POST':
        text = request.form['text']
        model_type = request.form['model']

        if model_type == 'nb':
            sentiment, probability = testingNB.predict_sentiment(text)
            sentiment = "Positive" if sentiment == 1 else "Negative"
            probability = probability[0] if sentiment == "Negative" else probability[1]
        elif model_type == 'lstm':
            probability = testingLSTM.predict_sentiment(text)
            sentiment = "Positive" if probability >= 0.5 else "Negative"
            probability = 1 - probability if sentiment == "Negative" else probability

    return render_template('index.html', sentiment=sentiment, probability=probability, model_type=model_type)


if __name__ == '__main__':
    app.run(debug=True)

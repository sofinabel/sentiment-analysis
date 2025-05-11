import numpy as np
from LSTMN import *

# Функция для предсказания тональности текста
def predict_sentiment(text):
    # Загрузка модели и словаря
    model = LSTMNet(vocab_size=164192, output_size=1, embedding_dim=100, hidden_dim=128, n_layers=2)
    model.load_state_dict(torch.load("model.pth"))
    vocab_to_int = np.load("vocab_to_int.npy", allow_pickle=True).item()

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    with torch.no_grad():
        # Предобработка текста
        punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
        text = ''.join([c for c in text.lower() if c not in punctuation])
        text_ints = [vocab_to_int.get(word, 0) for word in text.split()]
        text_ints = torch.tensor(text_ints).unsqueeze(0)
        if torch.cuda.is_available():
            text_ints = text_ints.cuda()
        h = model.init_hidden(1)
        output, _ = model(text_ints, h)
        return output.item()

import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.utils import shuffle
from LSTMN import *

# Считывание данных
n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
data_positive = pd.read_csv('positive (1).csv', sep=';', on_bad_lines='skip', names=n, usecols=['text'])
data_negative = pd.read_csv('negative (1).csv', sep=';', on_bad_lines='skip', names=n, usecols=['text'])

# Формирование сбалансированного датасета
sample_size = 40000
reviews = np.concatenate((data_positive['text'].values[:sample_size],
                          data_negative['text'].values[:sample_size]), axis=0)
labels = np.asarray([1] * sample_size + [0] * sample_size)
reviews, labels = shuffle(reviews, labels, random_state=0)

# Токенизация и создание словаря
def tokenize_and_create_vocab(texts):
    punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
    all_text = ' '.join(texts).lower()
    all_text = ''.join([c for c in all_text if c not in punctuation])
    words = all_text.split()
    vocab = Counter(words)
    vocab_to_int = {word: idx for idx, (word, _) in enumerate(vocab.items(), 1)}
    return vocab_to_int

vocab_to_int = tokenize_and_create_vocab(reviews)

# Преобразование текстов в числовые представления
def texts_to_ints(texts, vocab_to_int):
    reviews_ints = []
    for text in texts:
        reviews_ints.append([vocab_to_int.get(word, 0) for word in text.split()])
    return reviews_ints

reviews_ints = texts_to_ints(reviews, vocab_to_int)

# Добавление паддингов (векторы равной длины)
def add_padding(reviews_ints, seq_length=30):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

features = add_padding(reviews_ints)

# Разделение данных на обучающую, валидационную и тестовую выборки
split_frac = 0.8
split_idx = int(len(features) * split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = labels[:split_idx], labels[split_idx:]
val_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:val_idx], remaining_x[val_idx:]
val_y, test_y = remaining_y[:val_idx], remaining_y[val_idx:]

# Создание DataLoader
batch_size = 50
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

# Инициализация модели, критерия и оптимизатора
vocab_size = len(vocab_to_int) + 1
output_size = 1
embedding_dim = 100
hidden_dim = 128
n_layers = 2
model = LSTMNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
criterion = nn.BCELoss() # Бинарная кросс-энтропия
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Определение режима: GPU или CPU
if torch.cuda.is_available():
    model.cuda()

# Обучение модели
epochs = 4
clip = 5
model.train()
for e in range(epochs):
    h = model.init_hidden(batch_size)
    for inputs, labels in train_loader:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.data for each in h])
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        print(f'Epoch: {e + 1}/{epochs}, Loss: {loss.item()}')

# Сохранение модели и словаря
torch.save(model.state_dict(), "model.pth")
np.save("vocab_to_int.npy", vocab_to_int)

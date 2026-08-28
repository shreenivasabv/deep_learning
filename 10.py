from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)

X_train = pad_sequences(
    X_train,
    maxlen=200
)

X_test = pad_sequences(
    X_test,
    maxlen=200
)

model = Sequential([
    Embedding(10000, 128),
    Bidirectional(LSTM(64)),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32
)

print(
    "Accuracy:",
    model.evaluate(X_test, y_test)[1]
)

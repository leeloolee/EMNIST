from model import create_cnn_model
from data_load import data_loader

def train(DATA_URL):
    x_train, y_train, x_test = data_loader(DATA_URL)
    model = create_cnn_model(x_train)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20)
    return model
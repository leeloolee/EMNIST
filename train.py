from model import create_cnn_model
from data_load import data_loader
from tqdm.keras import TqdmCallback
import os

def train(DATA_URL, SAVE_URL):
    x_train, y_train, x_test = data_loader(DATA_URL)
    model = create_cnn_model(x_train)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
    checkpoint_path = os.path.join(SAVE_URL,"my_model.h5")
    # checkpoint_dir = SAVE_URL

    model.fit(x_train, y_train, epochs=20, verbose=0, callbacks=[TqdmCallback(verbose=2)])
    model.save(checkpoint_path)
    return model
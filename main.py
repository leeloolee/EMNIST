import json
from train import train

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)


    data_url = config["DATA"]
    save_url = config["TRAIN"]["CKPT_SAVE_DIR"]
    train(data_url, save_url)

    #TODO predict 추가하기 -> 파일로로

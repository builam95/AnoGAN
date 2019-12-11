from dcgan import DCGAN
import numpy as np
from keras.optimizers import Adam
from data_loader import load_keys
import datetime
import os
from plot_loss import loss_plotter
import glob
import cv2
import sys

def normalize(X):
    return (X - 127.5) / 127.5


def denormalize(X):
    return ((X + 1.0) / 2.0 * 255.0).astype(dtype=np.uint8)


def log_plotter(train_directories, g_optim, d_optim, epochs, ETA, save_dir):
    save_path = os.path.join(save_dir, "log.txt")
    with open(save_path, 'a') as f:
        for train_path in train_directories:
            f.write("train paht: {}".format(train_path) + "\n")
        f.write("epoch　は {}".format(epochs) + "\n")
        f.write("処理にかかった時間は {}".format(str(ETA)) + "[sec]" + '\n')
        f.write("g_optim: {}".format(g_optim.get_config()) + "\n")
        f.write("d_optim: {}".format(d_optim.get_config()) + "\n")


def normalize_brightness(img, ave, std):
    """
    画像の明るさを与えられた平均輝度, 標準偏差の値に正規化
    :param img:
    :param: ave: 画像群の輝度平均
    :param: std: 画像軍の標準偏差
    :return: normBrightImg: 正規化した画像
    """
    normBrightImg = (img - np.mean(img)) / np.std(img) * std + ave
    return normBrightImg


def calc_brightness(imgPaths):
    """
    画像群の明るさの平均輝度と標準偏差を計算てその値に画像の輝度を正規化
    :param imgPaths: 画像のパスリスト
    :return: ave: 画像の平均
    :return: std: 画像の標準偏差
    """
    # 平均・標準偏差の配列を用意
    ave_list = []
    std_list = []
    for imgPath in imgPaths:
        # 画像の読み込み
        img = cv2.imread(imgPath)
        # 画像一枚あたりの輝度平均・標準偏差を求める
        img_ave = np.mean(img)
        img_std = np.std(img)
        # 画像一枚の輝度平均と・標準偏差を配列に追加
        ave_list.append(img_ave)
        std_list.append(img_std)

    # 画像全体の輝度平均と標準偏差をそれぞれの画像の平均の平均と、標準偏差の平均とする
    ave = sum(ave_list) // len(ave_list)
    std = sum(std_list) // len(std_list)
    print("#### 画像軍の平均輝度は　{}　です。".format(ave))
    print("#### 画像軍の輝度の標準偏差は {} です".format(std))
    # import pdb
    # pdb.set_trace()
    return ave, std


def load_train(train_directories):
    """
    フォルダ内にある画像を使って、trainデータにする

    input
    -----
    train_dir: str
    画像フォルダまでのパス

    output
    -----
    X_train: ndarray
    DCGANに入力するtrainデータ
    """
    X_train = []
    for train_dir in train_directories:
        train_path_list = glob.glob(train_dir + "/*")
        print(train_path_list)

        # ave, std = calc_brightness(train_path_list)
        
        for train_img_path in train_path_list:
            train_img = cv2.imread(train_img_path)
            # 画像の明るさを正規化
            # train_img = normalize_brightness(train_img, ave=ave, std=std)
            # 明るさを正規化した画像を保存
            # save_path = os.path.join("/home/mizuno/PycharmProjects/AnoGAN_keys/Dataset/N021_N030/calc_kido_train",
            #                          os.path.basename(train_img_path))
            # cv2.imwrite(save_path, train_img)
            X_train.append(train_img)
    X_train = np.array(X_train)
    print('画像枚数 :', len(X_train))
    # import pdb
    # pdb.set_trace()
    return X_train


if __name__ == '__main__':
    EPOCHS = 1500
    FOLDERS = ['N001_N010','N011_N020','N001_N035','N021_N030']
    TRPATH = []
    for folder in FOLDERS :
        TRPATH.append("../../../Dataset/2019_June/"+folder+"/train")
    settings = {"input_dim": 64, "batch_size": 16, "epochs": EPOCHS, 
                "train_path": TRPATH,
                "save_folder": "epochs_" + str(EPOCHS)+'('+'_'.join(FOLDERS)+')',
                "g_optim": Adam(lr=0.001,beta_1 = 0.5,beta_2 = 0.9),
                "d_optim": Adam(lr=0.001, beta_1 = 0.5,beta_2 = 0.9),
                
                }

    start_time = datetime.datetime.now()
    # 設定を代入

    ### 0. 学習データの用意
    X_train = load_train(settings["train_path"])
    X_train = normalize(X_train)
    input_shape = X_train[0].shape
    print(input_shape)
    # X_test_original = X_test.copy()

    ### 1. DCGANの学習
    dcgan = DCGAN(settings["input_dim"], input_shape)
    dcgan.compile(settings["g_optim"], settings["d_optim"])

    # 保存先rootフォルダの作成
    save_root = os.path.join("../train_result", settings["save_folder"])
    # 間違えて上書きしそうになった時用
    if os.path.isdir(save_root):
        print("{}　は既に存在しています。別のフォルダ名を入力してください。".format(save_root))
        settings["save_folder"] = input("新しいフォルダ名: ")
        save_root = os.path.join("../train_result", settings["save_folder"])
        # sys.exit()



    save_result_dir = os.path.join(save_root, "result")
    save_weight_dir = os.path.join(save_root, "weights")
    os.makedirs(save_result_dir, exist_ok=True)
    os.makedirs(save_weight_dir, exist_ok=True)
    g_losses, d_losses = dcgan.train(settings["epochs"], settings["batch_size"], X_train, save_result_dir, save_weight_dir)

    # lossをcsvに書き込む
    # result, weightsと同じようにファイル名を日付で指定する。
    loss_dir = os.path.join(save_root, "loss")
    os.makedirs(loss_dir, exist_ok=True)
    loss_path = os.path.join(loss_dir, "loss.csv")
    with open(loss_path, 'w') as f:
        f.write("g_loss,d_loss" + '\n')
        for g_loss, d_loss in zip(g_losses, d_losses):
            f.write(str(g_loss) + ',' + str(d_loss) + '\n')

    # lossのプロット
    loss_plotter(loss_csv_path=loss_path, EPOCHS=EPOCHS)
    ETA = datetime.datetime.now() - start_time
    print("処理にかかった時間は {} ".format(ETA))
    # 学習に関する記録を保存
    log_plotter(train_directories=settings["train_path"], g_optim=settings["g_optim"], d_optim=settings["d_optim"], epochs=settings["epochs"], ETA=ETA, save_dir=save_root)

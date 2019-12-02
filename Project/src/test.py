import os, cv2
from dcgan import DCGAN
from anogan import ANOGAN
import numpy as np
from keras.optimizers import Adam
from data_loader import load_keys
from train import normalize, denormalize
# from score_plot import create_hist
import datetime
import time
import glob


def log_plotter(g_weight_path, d_weight_path, iterations, ETA, save_dir):
    save_path = os.path.join(save_dir, "log.txt")
    with open(save_path, 'a') as f:
        f.write("iteration: {}".format(iterations) + '\n')
        f.write("Gの重み {}".format(g_weight_path) + '\n')
        f.write("Dの重み {}".format(d_weight_path) + '\n')
        f.write("処理にかかった時間は {}".format(str(ETA)) + "[sec]" + '\n')


def load_data(path_list):
    X_test = []
    y_test = []
    for img_path in path_list:
        test_img = cv2.imread(img_path)
        X_test.append(test_img)
        y_test.append(0)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_test, y_test



def normalize_brightness(img):
    """
    画像の明るさを平均輝度64, 標準偏差に正規化
    :param img:
    :return: normBrightImg
    """
    normBrightImg = (img - np.mean(img)) / np.std(img) * 16 + 64
    return normBrightImg


if __name__ == '__main__':
    # 変更すべき箇所⇓実験ごとに変える。
    settings = {"input_dim": 64, # 入力画像の一辺のpx。特にいじる必要なし
                "g_weight_path": "../train_result/sample/weights/generator_49.h5", # Gの学習weight
                "d_weight_path": "../train_result/sample/weights/descriminator_49.h5", # Dの学習weight
                "test_dir": "../../../Dataset/2019_June/N001_N010/test/learned_imgs", # テストデータのフォルダ
                "save_root": "../test_result/sample", # 結果を保存するフォルダ
                "csv_name": "Normal100_epoch4000" # スコアを記録するcsvファイルのファイル名
                }
    anogan_optim = Adam(lr=0.001, amsgrad=True)

    os.makedirs(settings["save_root"], exist_ok=False)
    # 画像のフォルダのパスを入れる
    test_path_list = glob.glob(settings["test_dir"] + "/*.jpg")
    save_score_path = os.path.join(settings["save_root"], settings["csv_name"] + ".csv")
    # 使用したい重みのパスを入れる
    weight_epoch = int(os.path.basename(settings["g_weight_path"]).replace("generator_", "").replace(".h5", "")) + 1
    print("使用するGeneratorの重みは　{} ".format(settings["g_weight_path"]))
    print()
    print("使用するDiscriminatorの重みは　{}".format(settings["d_weight_path"]))

    # データ・セットの用意
    X_test, y_test = load_data(test_path_list)

    # データの正規化
    X_test = normalize(X_test)
    input_shape = X_test[0].shape

    # dcganを用意する
    dcgan = DCGAN(settings["input_dim"], input_shape)
    dcgan.load_weights(
        d_weight=settings["d_weight_path"],
        g_weight=settings["g_weight_path"])

    # 保存先rootフォルダの指定


    start = time.time()
    # AnoGANインスタンスの生成
    anogan = ANOGAN(settings["input_dim"], dcgan.g)
    anogan.compile(anogan_optim)

    with open(save_score_path, "w") as f:
        f.write("filenam,anomaly_score" + "\n")
    for i, (test_img, test_path) in enumerate(zip(X_test, test_path_list)):
        filename = os.path.basename(test_path)[:-4]
        loop_start = time.time()
        test_img = test_img[np.newaxis, :, :, :]

        # AnoGANの異常度、生成した画像の算出
        anomaly_score, generated_img = anogan.compute_anomaly_score(test_img)
        generated_img = denormalize(generated_img)

        # 入力した画像と生成した画像を結合する
        imgs = np.concatenate((denormalize(test_img[0]), generated_img[0]), axis=1)
        
        # スコアと結合画像の保存
        with open(save_score_path, "a") as f:
            f.write(filename + "," + str(anomaly_score) + "\n")
        cv2.imwrite(settings["save_root"] + "/" + filename + "_"+  str(int(anomaly_score)) + '.png', imgs)

        eta_of_loop = time.time() - loop_start
        print("１処理にかかった時間は {}".format(eta_of_loop) + "[sec]でした。")
        print()
        print("--------------------------------------------------------")
        time.sleep(1)


    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

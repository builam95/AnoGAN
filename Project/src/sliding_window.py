import glob
import cv2
import numpy as np
import os
from tqdm import tqdm


def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    ext = ".JPG"
    img_path_list = []
    for curDir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".JPG"):
                img_path = os.path.join(curDir, file)
                img_path_list.append(img_path)

    print("done")
    return img_path_list


def create_save_dir(img_path, save_root, ext):
    filename = os.path.basename(img_path).replace(ext, "")
    dir = os.path.basename(os.path.dirname(img_path))
    save_dir = os.path.join(save_root, dir, filename)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def sliding_window(img_path, save_dir, winW, winH, stepsize):
    BLACK = 0
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    for y in range(0, h, stepsize):
        for x in range(0, int((2 * w) / 3), stepsize):
            trim_img = img[y: y + winH, x: x + winW]
            filename = "x_" + str(x) + "y_" + str(y) + ".jpg"
            trimh, trimw, _ = trim_img.shape
            # 全面黒またはトリミングサイズが指定ウィンドウサイズでなかったらはじく
            if trim_img.all() == BLACK or (trimh, trimw) != (winH, winW):
                # except_save_dir = os.path.join(save_dir, "except")
                # os.makedirs(except_save_dir, exist_ok=True)
                # save_except_path = os.path.join(except_save_dir, filename)
                # cv2.imwrite(save_except_path, trim_img)
                continue
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, trim_img)
            print("{}　保存完了".format(save_path))
            print()

#
# def sp_sliding_window(img, winW, winH, stepsize, save_dir):
#     """
#     鍵画像のうち検査対象とならない部分を伐り出さないでスライディングウィンドウ
#     :param img_path:
#     :param save_dir:
#     :param winW:
#     :param winH:
#     :param stepsize:
#     :return:
#     """
#     count = 0
#     BLACK = 0
#     X = []
#     Y = []
#     X_test = []
#     h, w, _ = img.shape
#     for y in range(0, h, stepsize):
#         for x in range(0, int((2 * w) / 3), stepsize):
#             trim_img = img[y: y + winH, x: x + winW]
#             filename = str(count) + ".jpg"
#             trimh, trimw, _ = trim_img.shape
#             count += 1
#             # 全面黒またはトリミングサイズが指定ウィンドウサイズでなかったらはじく
#             if trim_img.all() == BLACK or (trimh, trimw) != (winH, winW):
#                 save_ex_dir = os.path.join(save_dir, "except")
#                 os.makedirs(save_ex_dir, exist_ok=True)
#                 save_ex_path = os.path.join(save_ex_dir, filename)
#                 cv2.imwrite(save_ex_path, trim_img)
#                 X.append(trim_img)
#                 Y.append(0)
#                 continue
#             save_test_dir = os.path.join(save_dir, "numpy_test.py")
#             os.makedirs(save_test_dir, exist_ok=True)
#             save_test_path = os.path.join(save_test_dir, filename)
#             cv2.imwrite(save_test_path, trim_img)
#             X_test.append(trim_img)
#     return save_test_dir


def label_founder():
    """OriginにあってMarkedにないもの(= 傷部分の画像)を探して、それぞれのラベルをつける。
    originとmarkedの差分をとって、originにあってmarkedにないものをoriginからtestのdent, scratchに分ける。
    """
    # Origin画像の読み込み


if __name__ == "__main__":

    mask_imgs = "/media/kiyo/add_vol/Anomaly_kets/Dataset/NORMAL/masked"
    impath_list = make_img_list(mask_imgs)
    for img_path in impath_list:
        filename = os.path.basename(img_path)
        save_root = os.path.dirname(img_path).replace("masked", "slide_imgs")
        print(filename)

        # 画像のファイル名をディレクトリ名にし、その中にスライドされた画像を保存する。
        save_dir = os.path.join(save_root, filename)
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)

        # start sliding window
        sliding_window(img_path=img_path, save_dir=save_dir, winW=64, winH=64, stepsize=32)

import glob
import os
import numpy as np
import cv2
from tqdm import tqdm


def img_show(img):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_ideal_contour():
    """理想的な形状を読み込む"""

    ideal_img = cv2.imread('./ideals/ideal.jpg', cv2.IMREAD_GRAYSCALE)
    ideal_contours, _ = cv2.findContours(ideal_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    epsilon = 0.003 * cv2.arcLength(ideal_contours[-1], True)
    approx = cv2.approxPolyDP(ideal_contours[-1], epsilon, True)
    return approx


def draw_outer_edge(img):
    """外縁部を白で塗りつぶし"""

    height = img.shape[0]
    width = img.shape[1]
    cv2.rectangle(img, (0, 0), (width, height), (255, 255, 255), 50)

    return img


def get_key_contour_from_contours(contours, ideal_contour):
    """最も鍵に近い輪郭を返す"""
    ret_list = []
    cnt_dict = {}

    for cnt in contours:
        ret = cv2.matchShapes(ideal_contour, cnt, 1, 0.0)
        cnt_dict[ret] = cnt
        ret_list.append(ret)
        print(ret)

    return cnt_dict[min(ret_list)]


def get_contours(src):
    """鍵の輪郭を取得"""

    # ぼかし
    blur = cv2.bilateralFilter(src, 9, 75, 75)
    # hsvに変換
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV_FULL)
    h_img, s_img, v_img = cv2.split(hsv_img)

    # 2値化
    _, thresh_img = cv2.threshold(h_img, 0, 255, cv2.THRESH_OTSU)

    # ゴミとり
    kernel = np.ones((5, 5), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)

    # 領域抽出
    contours, _ = cv2.findContours(thresh_img - 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours
    contours_list = []
    for i, contour in enumerate(contours):
        # 小さな領域の場合は間引く
        area = cv2.contourArea(contour)
        if area < 5000:
            continue
        contours_list.append(contour)
    return contours_list


def mask_extra(src, key_contour):
    """余分な部分をマスク"""

    # マスク作成
    plain_img = np.zeros((src.shape[0], src.shape[1], 1), dtype=np.uint8)
    mask_img = cv2.drawContours(plain_img, [key_contour], -1, 255, -1)

    masked = cv2.bitwise_and(src, src, mask=mask_img)

    return masked


def cut_around(src, key_contour):
    ret = cv2.boundingRect(key_contour)

    x1 = ret[0]
    y1 = ret[1]
    x2 = ret[0] + ret[2]
    y2 = ret[1] + ret[3]

    crop_img = src[y1:y2, x1:x2]

    return crop_img


def mask_background(src_path, save_path):
    """背景をマスクします"""

    # 画像読み込み
    src = cv2.imread(src_path, cv2.IMREAD_COLOR)

    src = draw_outer_edge(img=src)

    ideal_cnt = load_ideal_contour()

    # 輪郭を取得
    contours = get_contours(src=src)
    # マスク処理
    key_contour = get_key_contour_from_contours(contours, ideal_contour=ideal_cnt)
    masked = mask_extra(src=src, key_contour=key_contour)
    masked = cut_around(src=masked, key_contour=key_contour)

    # 保存
    cv2.imwrite(save_path, masked)

# def mask_background(src_path):
#     """背景をマスクします"""
#
#     # 画像読み込み
#     src = cv2.imread(src_path, cv2.IMREAD_COLOR)
#
#     src = draw_outer_edge(img=src)
#
#     ideal_cnt = load_ideal_contour()
#
#     # 輪郭を取得
#     contours = get_contours(src=src)
#     # マスク処理
#     key_contour = get_key_contour_from_contours(contours, ideal_contour=ideal_cnt)
#     masked = mask_extra(src=src, key_contour=key_contour)
#     masked = cut_around(src=masked, key_contour=key_contour)
#
#     return masked


def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    ext = ".jpg"
    img_path_list = []
    for curDir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".JPG"):
                img_path = os.path.join(curDir, file)
                img_path_list.append(img_path)

    print("done")
    return img_path_list


if __name__ == '__main__':
    ori_dir = "/home/kiyo/PycharmProjects/Anomaly_kets/Dataset/NORMAL/kagizentai"
    img_list = make_img_list(ori_dir)
    for img_path in img_list:
        save_path = img_path.replace("kagizentai", "masked")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mask_background(src_path=img_path, save_path=save_path)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def loss_plotter(loss_csv_path, EPOCHS):
    """
    まずはじめにloss.csvファイルの一行目にg_loss,d_lossと手動で挿入する
    :param loss_csv:
    :return:
    """
    df = pd.read_csv(loss_csv_path)
    g_loss = df['g_loss']
    d_loss = df['d_loss']

    epochs = range(len(g_loss))

    # 損失値をプロット
    plt.xlim(0, EPOCHS-1)
    plt.ylim(0, 20)
    plt.plot(epochs, g_loss, linestyle="-", color="green", label="G loss")
    plt.plot(epochs, d_loss, linestyle="-", color="blue", label="D loss")
    plt.title("Generator and Descriminator loss", fontsize=16)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.legend()
    # グラフの保存
    # 読みこんだcsvと同じディレクトリに保存する。
    save_path = loss_csv_path.replace("loss.csv", "loss_figure.png")
    plt.savefig(save_path)


def plot_3img(img1_path, img2_path, img3_path):
    """
    画像を3つ読み込んで、並べて返す。
    """

    img_1 = plt.imread(img1_path)
    img_2 = plt.imread(img2_path)
    img_3 = plt.imread(img3_path)

    plt.subplot()


if __name__ == '__main__':
    loss_csv_path = "/home/mizuno/PycharmProjects/AnoGAN_keys/AnoGAN/learn/N001_N010/N001_N010_kido128_seed100/loss/loss.csv"
    loss_plotter(loss_csv_path)

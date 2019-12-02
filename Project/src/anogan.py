import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K


def sum_of_residual(y_true, y_pred):
    # print(y_true)
    # print("----------------------------------")
    # print(y_pred)
    print("----------------------------------")
    print(K.abs(y_true - y_pred))
    print("----------------------------------")
    # print(y_pred.shape)
    return K.sum(K.abs(y_true - y_pred))


class ANOGAN(object):
    """
    AnoGANに入力した画像に近い画像をGeneratorが出力するための潜在変数を探索する。
    以下のアルゴリズムでzを探索する。
    1. 適当な潜在変数z_tをGeneratorに入力、出力した画像と異常検知したい画像の輝度を比較。
    2. 勾配降下法で未知の画像に近い画像を生成する潜在空間のz_t+1を求める．
    3. 生成した画像G(z_t+1)と未知の画像とのロスを求める．
    4. 1~3を繰り返すことにより，未知の画像と最も近い画像を生成できる潜在変数z_γを求める．
    """
    def __init__(self, input_dim, g):
        self.input_dim = input_dim
        self.g = g
        g.trainable = False

        # Input layer cann't be trained. Add new layer as same size & same distribution
        anogan_in = Input(shape=(input_dim,))
        g_in = Dense((input_dim), activation='tanh', trainable=True)(anogan_in)
        g_out = g(g_in)
        self.model = Model(inputs=anogan_in, outputs=g_out)
        self.model_weight = None

    def compile(self, optim):
        self.model.compile(loss=sum_of_residual, optimizer=optim)
        K.set_learning_phase(0)

    def compute_anomaly_score(self, x, iterations=100):
        # 適当な
        z = np.random.uniform(-1, 1, size=(1, self.input_dim))
        # print(z)

        # learning for changing latent
        # self.model.summary()
        loss = self.model.fit(z, x, batch_size=1, epochs=iterations, verbose=0)

        loss = loss.history['loss'][-1]

        similar_data = self.model.predict_on_batch(z)

        return loss, similar_data

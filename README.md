# AnoGAN

## 学習

`AnoGAN/Project/src`にあるtrain.pyを実行する。

`if __name__ == "__main__"`以下のsettings部分を書き換える。

```python
settings = {"input_dim": 64, "batch_size": 16, "epochs": 100, 
                "train_path": "../../../Dataset/N001_N010/train",
                "save_folder": "epochs_10",
                "g_optim": Adam(lr=0.001, beta_1=0.5, beta_2=0.9),
                "d_optim": Adam(lr=0.001, beta_1=0.5, beta_2=0.9)
           }
```

実験する際に変更が必要な箇所のみ記述。
- epochs: エポック
- train_path: 訓練データが格納されたフォルダまでのパス。Docker内のパスであることに注意
- save_folder:　結果を保存するフォルダ名。train_{画像枚数}__epoch_{エポック数}のような命名則がよいかも。


## 異常度算出
`AnoGAN/Project/src`内のtest.pyを実行する。
`if __name__ == "__main__"`以下のsettings部分を書き換える。

```python
settings = {"input_dim": 64, # 入力画像の一辺のpx。特にいじる必要なし
            "g_weight_path": "../../../Dataset/N001_N010/train", # Gの学習weight
            "d_weight_path": "../", # Dの学習weight
            "test_dir": "../../../Dataset/2019_11_28/DSC05201", # テストデータのフォルダ
            "save_root": "../../../Dataset/2019_11_28/DSC05201", # 結果を保存するフォルダ
            "csv_name": "Normal100_epoch4000" # スコアを記録するcsvファイルの
           }
```

実験する際に変更が必要な箇所のみを記述。

- g_weight_path: Generatorの重み。
- d_weight_path: Discriminatorの重み。
- test_dir: テスト画像が格納されたフォルダ。save_rootと同じにする。
- save_root: 結果を保存するためのフォルダ。test_dirと同じにする。
- csv_name: スコア記録するcsvファイルのファイル名。train_{画像枚数}__epoch_{エポック数}のような命名則がよいかも。

## データ・セットに関して
2019年7月のMIRUの予稿に使用したデータを保存したフォルダが
`Dataset/2019_June/`にあります。

### ディレクトリ構成

```bash
.
├── All_IMGS
├── N001_N010
├── N011_N020
├── N021_N030
├── dent9
├── dent_imgs
├── masked_normal_keys
```

- dent9: MIRUの予稿に使用した9種類のデント。小中大各3つ合計9枚の画像です。
- dent_imgs: 
- slide_imgs_normal：Normalサンプル100本を64*64pxをoverlapなしで切り出した画像が入ってます。この中から1000枚の訓練画像を選ぶことになります。
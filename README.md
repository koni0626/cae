# 使い方

## 訓練するとき
```
python ConvAutoEncoder.py --mode train --train_img_dir="data\cut\0000" --weight_dir="weights" --encode_img_dir="cae_output"
```

## 予測するとき
```
python ConvAutoEncoder.py --mode test --test_img_dir=data\cut\0000test --weight_file=weights_cae\cae_1499.pth --batch_size=2
```

# コマンドのオプション
**--mode**
学習するときはtrain、予測するときはtestを指定する。

**--epochs**
エポック数

**--batch_size**
バッチサイズ。省略時は32を仮定する。

**--train_img_dir**
学習に使用する画像ディレクトリを指定する。

**--weight_dir**
重みファイルの保存先。

**--encode_img_dir**
学習ごとにエンコードした画像を出力するディレクトリ。

**--test_img_dir**
予測に使用する画像ディレクトリ

**--weight_file**
予測に使用する重みファイル

# 出力について
予測を行うと、テスト画像と、AIが出力した画像の差分の値を出力する。
具体的には、各画素の差をとり、差の絶対値の合計値が出力される。

学習済みデータやそれに近いデータをエンコードした場合は差が小さく、異常があったり、全く未知の画像だった場合は差分が大きくなる。

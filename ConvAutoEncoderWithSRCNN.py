# coding:UTF-8
import random
import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms
from torchvision.utils import save_image


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        #Encoder Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512,
                               kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256,
                               kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128,
                               kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=3, padding=1)

        #self.conv5 = nn.Conv2d(in_channels=64, out_channels=32,
        #                       kernel_size=3, padding=1)
        #Decoder Layers
        #self.t_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=64,
        #                                  kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128,
                                          kernel_size=2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=256,
                                          kernel_size=2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(in_channels=256, out_channels=512,
                                          kernel_size=2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(in_channels=512, out_channels=3,
                                          kernel_size=2, stride=2)

        self.srcnn_conv1 = nn.Conv2d(in_channels=3, out_channels=128,
                               kernel_size=9, padding=[9//2, 9// 2], bias=False, padding_mode='replicate')

        self.srcnn_conv2 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=3, padding=[3//2, 3 // 2], bias=False, padding_mode='replicate')

        self.srcnn_conv3 = nn.Conv2d(in_channels=64, out_channels=32,
                               kernel_size=1, padding=[1//2, 1 // 2], bias=False, padding_mode='replicate')

        self.srcnn_conv4 = nn.Conv2d(in_channels=32, out_channels=3,
                               kernel_size=5, padding=[5//2, 5 // 2], bias=False, padding_mode='replicate')

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #encode#                          #in  [i, 1, 118, 118]
        x = self.relu(self.conv1(x))      #out [i, 16, 118, 118]
        x = self.pool(x)                  #out [i, 16, 56, 56]
        x = self.relu(self.conv2(x))      #out [i, 4, 56, 56]
        x = self.pool(x)                  #out [i ,4, 56, 56]
        x = self.relu(self.conv3(x))      #out [i, 4, 28, 28]
        x = self.pool(x)                  #out [i ,4, 28, 28]
        x = self.relu(self.conv4(x))      #out [i, 4, 14, 14]
        x = self.pool(x)                  #out [i ,4, 7, 7]
        #x = self.relu(self.conv5(x))      #out [i, 4, 14, 14]
        #x = self.pool(x)                  #out [i ,4, 7, 7]

        #decode#                          #in  [i, 4,  7,   7]
       # x = self.relu(self.t_conv1(x))    #out [i, 16, 14, 14]
        x = self.relu(self.t_conv2(x))    #out [i, 16, 28, 28]
        x = self.relu(self.t_conv3(x))    #out [i, 16, 56, 56]
        x = self.relu(self.t_conv4(x))    #out [i, 16, 108, 108]
        x = self.sigmoid(self.t_conv5(x)) #out [i, 3, 118, 118]

        # ここからsrcnn
        x = self.relu(self.srcnn_conv1(x))
        x = self.relu(self.srcnn_conv2(x))
        x = self.relu(self.srcnn_conv3(x))
        x = self.relu(self.srcnn_conv4(x))


        return x


def train_net(device,
              n_epochs,
              train_loader,
              net,
              weight_dir,
              encode_img_dir,
              optimizer_cls = optim.Adam,
              loss_fn = nn.MSELoss()):
    """
    n_epochs…訓練の実施回数
    net …ネットワーク
    device …　"cpu" or "cuda:0"
    """
    losses = []         #loss_functionの遷移を記録
    optimizer = optimizer_cls(net.parameters(), lr = 1e-4)
    net.to(device)
    for epoch in range(n_epochs):
        running_loss = 0.0
        net.train()

        for i, XX in enumerate(train_loader):
            XX = XX.to(device)
            optimizer.zero_grad()

            XX_pred = net(XX)             #ネットワークで予測

            try:
                save_image(XX_pred, os.path.join(encode_img_dir, "{:03d}.jpg".format(epoch)))
            except:
                pass

            loss = loss_fn(XX, XX_pred)   #予測データと元のデータの予測
            loss.backward()
            optimizer.step()              #勾配の更新
            running_loss += loss.item()

        losses.append(running_loss / i)
        print("epoch", epoch, ": ", running_loss / i)
        gen_file_name = os.path.join(weight_dir, "cae_{}.pth".format(epoch))
        torch.save(net.state_dict(), gen_file_name)

    return losses


class CAEDataSet(data.Dataset):
    """
    ジェネレータークラス
    """

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)

    def __getitem__(self, index):
        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)
       # img = img.convert("RGB")  # 3チャンネルに変換

        img_transformed = self.transform(img)

        return img_transformed


def SSIM(x, y,window_size=3):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    clip_size = (window_size -1)//2

    mu_x = nn.functional.avg_pool2d(x, window_size, 1, padding=0)
    mu_y = nn.functional.avg_pool2d(y, window_size, 1, padding=0)

    x = x[:,:,clip_size:-clip_size,clip_size:-clip_size]
    y = y[:,:,clip_size:-clip_size,clip_size:-clip_size]

    sigma_x = nn.functional.avg_pool2d((x  - mu_x)**2, window_size, 1, padding=0)
    sigma_y = nn.functional.avg_pool2d((y - mu_y)** 2, window_size, 1, padding=0)

    sigma_xy = (
        nn.functional.avg_pool2d((x- mu_x) * (y-mu_y), window_size, 1, padding=0)
    )

    mu_x = mu_x[:,:,clip_size:-clip_size,clip_size:-clip_size]
    mu_y = mu_y[:,:,clip_size:-clip_size,clip_size:-clip_size]

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    loss = torch.clamp((1 - SSIM) , 0, 2)
    #save_image(loss, 'SSIM_GRAY.png')

    return  loss, torch.mean(loss)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resize = (112, 112)
    parser = argparse.ArgumentParser(description="Convolution Auto Encoder")
    parser.add_argument("--mode", help="学習する場合train、予測する場合はtestを指定する", required=False)
    parser.add_argument("--epochs", help="エポック数", type=int, default=1500)
    parser.add_argument("--batch_size", help="学習時のミニバッチのサイズ", type=int, default=32)
    parser.add_argument("--train_img_dir", help="学習に使用する画像ディレクトリ")
    parser.add_argument("--weight_dir", help="重みファイルの保存先")
    parser.add_argument("--encode_img_dir", help="学習ごとにエンコードした画像を出力するディレクトリ")

    parser.add_argument("--test_img_dir", help="予測に使用する画像ディレクトリ")
    parser.add_argument("--weight_file", help="予測に使用する重みファイル")

    args = parser.parse_args()

    if args.mode == "train":
        batch_size = args.batch_size
        train_img_dir = args.train_img_dir
        epochs = args.epochs
        weight_dir = args.weight_dir
        encode_img_dir = args.encode_img_dir

        if not os.path.exists(train_img_dir):
            print("{}がありません".format(train_img_dir))
            sys.exit(1)
        else:
            train_img_list = glob.glob(os.path.join(train_img_dir, "*"))

        if not os.path.exists(weight_dir):
            os.mkdir(weight_dir)

        train_dataset = CAEDataSet(file_list=train_img_list, transform=transforms.Compose([
                                    transforms.Resize(resize),
                                    transforms.ToTensor()]))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = ConvAutoencoder()
        model.train()

        losses = train_net(device,
                           n_epochs=epochs,
                           train_loader=train_dataloader,
                           net=model,
                           weight_dir=weight_dir,
                           encode_img_dir=encode_img_dir)
    else:
        # testの場合
        test_img_dir = args.test_img_dir
        weight_file = args.weight_file
        print(weight_file)
        net = ConvAutoencoder()
        weight = torch.load(weight_file)
        net.load_state_dict(weight)
        net.eval()
        net = net.to(device)

        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()])

        test_img_dir = args.test_img_dir
        test_img_file_list = glob.glob(os.path.join(test_img_dir, "*"))

        total_loss = 0.
        for i, file_name in enumerate(test_img_file_list):
            img = Image.open(file_name)
            img = transform(img)

            img = torch.unsqueeze(img, 0) #次元を一つ追加
            with torch.no_grad():
                x = img.to(device)

                out = net(x)
                if not os.path.exists("test_encode_img_dir"):
                    os.makedirs("test_encode_img_dir")



                diff = x - out
                #loss = abs(diff[0])*10.0
                loss, loss_mean = SSIM(x, out)
                #print(loss)
                total_loss += loss_mean
                print("{}:loss:{}".format(i,loss_mean))
                loss_img = loss#torch.unsqueeze(loss, 0)
                #sample = torch.cat([x, out, loss_img], dim=0)
                save_image(loss, os.path.join("test_encode_img_dir", "{:03d}.jpg".format(i)))

                if loss_mean >= 0.30:
                    print(file_name)
                    if not os.path.exists("test"):
                        os.mkdir("test")
                    #save_image(sample, os.path.join("test", "{:03d}.jpg".format(i)))
                    shutil.copy(file_name, os.path.join("test", file_name.split(os.sep)[-1]))

        mean = total_loss / len(test_img_file_list)
        print("mean:{}".format(mean))






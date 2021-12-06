import os
from tqdm import tqdm, trange

import numpy as np
import openpyxl
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = '/Users/youngjaepark/Desktop/pythonProject/CAPTCHA/imgs'
ANNOTATION_PATH = '/Users/youngjaepark/Desktop/pythonProject/CAPTCHA/captcha_dataset.xlsx'
SAVE_PATH = './saved_model'

if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

MAX_DATATSET_NUM = 500

RESIZE_WIDTH = 80
RESIZE_HEIGHT = 40

str_to_ans = {}
case = ['2','3','4','5','6',
        '7','8','a','b','c',
        'd','e','f','g','h',
        'k','l','m','n','p',
        'r','w','x','y']
for i, c in enumerate(case):
    str_to_ans[c] = i



class Captcha_dataset(Dataset):
    def __init__(self, to=1, untill=500):
        dataname = [f'data{i}.png' for i in range(to, untill+1)]
        self.imgs = [Image.open(os.path.join(IMAGE_PATH,path)).convert('L')
                     for path in dataname]

        wb = openpyxl.load_workbook(ANNOTATION_PATH)
        sheet = wb.active
        self.annotation = []
        for row in sheet.iter_rows(min_row=to, max_row=untill, min_col=2, max_col=2):
            for cell in row:
                self.annotation.append(cell.value)
        self.transform = Compose([ToTensor(),
                                  Resize((RESIZE_HEIGHT, RESIZE_WIDTH))])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx, show=False):
        img = self.imgs[idx]
        annotation = str(self.annotation[idx])
        temp_img = np.asarray(img, dtype=np.uint8)
        temp_img = temp_img.squeeze()
        temp_img = np.where(temp_img > 0, 0, 255).astype(np.uint8)

        # 세로 선 2줄만 있는 경우 삭제
        count_temp_img = np.sum(temp_img, axis=0) // 255
        count_temp_img = np.where(count_temp_img <= 4, True, False)
        temp_img[:, count_temp_img] = 0

        # 의미있는 영역만 추출
        count_temp_img = np.sum(temp_img, axis=0) // 255
        count_temp_img = np.where(count_temp_img!=0, True, False)
        temp_img = temp_img[:,count_temp_img]
        temp_shape = temp_img.shape

        temp_img = cv2.blur(temp_img, (3,3))
        ret, temp_img = cv2.threshold(temp_img, 170, 255, cv2.THRESH_BINARY)
        temp_img = cv2.dilate(temp_img, np.ones((3, 1), np.uint8))
        temp_img = cv2.erode(temp_img, np.ones((2, 2), np.uint8))

        temp_img = Image.fromarray(temp_img)

        if show:
            plt.suptitle(f'{annotation} {temp_shape}')
            plt.imshow(temp_img)
            plt.show()

        img = self.transform(temp_img)

        if show:
            plt.suptitle(f'{annotation} {(RESIZE_HEIGHT, RESIZE_WIDTH)}')
            temp_img = np.asarray(img)
            temp_img = np.squeeze(temp_img)
            plt.imshow(temp_img)
            plt.show()

        return img, annotation

class Breaker_model(torch.nn.Module):
    def __init__(self):
        super(Breaker_model, self).__init__()
        self.first_conv = torch.nn.Conv2d(1, 20, 3, stride=1, padding=1)
        self.second_conv = torch.nn.Conv2d(20, 50, 3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
        self.hidden_dense = torch.nn.Linear(10000, 1000)
        self.character_dense = [torch.nn.Linear(1000, 24) for _ in range(4)]

        self.softmax = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(2, stride=2)
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, data, drop_out=False):
        # Conv1
        data = self.first_conv(data)
        data = self.relu(data)
        data = self.max_pool(data)

        # Conv2
        data = self.second_conv(data)
        data = self.relu(data)
        data = self.max_pool(data)

        # Hidden Dense
        data = self.flatten(data)
        data = self.relu(data)
        if drop_out:
            data = self.dropout(data)
        data = self.hidden_dense(data)
        if drop_out:
            data = self.dropout(data)

        # Character Dense
        character_list = []
        for dense in self.character_dense:
            temp = dense(data)
            temp = self.softmax(temp)
            temp = torch.unsqueeze(temp, 1)
            character_list.append(temp)


        return torch.cat(character_list, 1)

def split_ans(batch):
    batch = list(batch)
    ans_list = []
    for string in batch:
        temp = []
        for c in str(string):
            temp.append(str_to_ans[c])
        ans_list.append(temp)
    return ans_list

def decoding(arr):
    temp = ''
    for i in range(4):
        temp += case[arr[i]]
    return temp

def gen_plot(imgs, annotations, predictions, batch_size=10):
    fig = plt.figure(figsize=(5., 5.), dpi=200)
    sub_plots = [fig.add_subplot(2, 5, i+1) for i in range(batch_size)]
    imgs = imgs.permute(0, 2, 3, 1)
    fig.tight_layout()
    for idx, sub_plot in enumerate(sub_plots):
        sub_plot.set_title(f'GT:{annotations[idx]} // PRED:{decoding(predictions[idx])}',
                           fontsize=6)
        sub_plot.imshow(imgs[idx])
        ans = annotations[idx] == predictions[idx]
        sub_plot.set_xlabel(f'{ans}')
        sub_plot.axis('off')

    return fig

if __name__ == '__main__':
    # CD = Captcha_dataset(to=1, untill=650)
    # # CD = Captcha_dataset(to=651, untill=700)
    # print(len(CD.imgs), len(CD.annotation))
    # for i in range(10):
    #     img, annotation = CD.__getitem__(i, show=True)
    #
    # exit()

    EPOCH = 1000
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 20

    CD_train = Captcha_dataset(to=1, untill=650)
    CD_test = Captcha_dataset(to=651, untill=700)

    train_loader = DataLoader(CD_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CD_test, batch_size=BATCH_SIZE//2, shuffle=False)

    model = Breaker_model()
    CE_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(LEARNING_RATE))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    writer = SummaryWriter(log_dir=os.path.join('runs', f'{EPOCH}_blur_shuffle_resize80_total700_dropout0.3_LRscheduler'))

    p_bar = trange(1, EPOCH+1)
    for epoch in p_bar:
        # Train
        p_bar.set_description('Training')
        model.train()
        ep_loss = []
        for img, annotation in train_loader:
            optimizer.zero_grad()
            prediction = model(img)
            answers = split_ans(annotation)
            answers = np.asarray(answers)
            answers = torch.Tensor(answers).to(torch.long)

            total_loss = 0.
            for i in range(4):
                PRED = torch.squeeze(prediction[:, i, :])
                GT = answers[:, i]
                total_loss += CE_loss(PRED, GT)

            ep_loss.append(total_loss)
            total_loss.backward()
            optimizer.step()
        p_bar.set_postfix_str(f'{epoch}/{EPOCH} Total Loss: {sum(ep_loss)/len(ep_loss)}')
        # print(f'{epoch}/{EPOCH} Total Loss: {sum(ep_loss)/len(ep_loss)}')
        writer.add_scalar('train/loss', sum(ep_loss)/len(ep_loss), epoch)
        writer.add_scalar('train/LEARNING_RATE', optimizer.param_groups[0]['lr'], epoch)

        if epoch == EPOCH:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, SAVE_PATH+f'/{EPOCH}_LRSCH.tar')

        # Validation
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                valid_total_loss = []
                for cnt, batch in enumerate(test_loader):
                    img, annotation = batch
                    outputs = model(img)
                    decoded_outputs = torch.argmax(outputs, dim=-1)

                    pred_img = gen_plot(img, annotation, decoded_outputs)
                    writer.add_figure(f'test/pred_{epoch}', pred_img, cnt)

                    answers = split_ans(annotation)
                    answers = np.asarray(answers)
                    answers = torch.Tensor(answers).to(torch.long)

                    total_loss = 0.
                    for i in range(4):
                        PRED = torch.squeeze(outputs[:, i, :])
                        GT = answers[:, i]
                        total_loss += CE_loss(PRED, GT)
                    valid_total_loss.append(total_loss)
                valid_total_loss = sum(valid_total_loss)
                writer.add_scalar(f'test/loss', valid_total_loss, epoch)
                scheduler.step(valid_total_loss)




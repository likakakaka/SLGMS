from models.HybridGNet2IGSC import Hybrid

import os
import numpy as np
from torchvision import transforms
import torch

from utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
import scipy.sparse as sp

import cv2
import pathlib
import re


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def getDenseMask(RL, LL, H):
    img = np.zeros([1024, 1024], dtype='uint8')

    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    img_copy1 = img.copy()
    img2 = cv2.drawContours(img_copy1, [H], -1, 1, -1)
    ###img is the segmentation result without heart, img2 is the segmentation result with heart
    return img, img2


def get_original_coordinates(resized_coordinates, resized_size, original_size):
    resized_width, resized_height = resized_size
    original_width, original_height = original_size

    x_resized, y_resized = resized_coordinates

    x_original = (x_resized / resized_width) * original_width
    y_original = (y_resized / resized_height) * original_height

    return x_original, y_original


if __name__ == "__main__":
    from PIL import Image
    import math

    device = "cuda:0"

    A, AD, D, U = genMatrixesLungsHeart()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config = {}

    config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to('cuda:0') for x in X] for X in (A_, D_, U_))

    config['latents'] = 64
    config['inputsize'] = 1024

    f = 32
    config['filters'] = [2, f, f, f, f // 2, f // 2, f // 2]
    config['skip_features'] = f

    hybrid = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
    msg = hybrid.load_state_dict(torch.load("/disk3/wjr/workspace/chexmask_bestMSE.pt"))
    print(msg)
    hybrid.eval()
    print('Model loaded')

    folder = '/disk1/wjr/dataset/NIHorg_dataset/images'
    data_root = pathlib.Path(folder)
    all_files = list(data_root.glob('*.png'))
    all_files = [str(path) for path in all_files]

    all_files.sort(key=natural_key)

    output_dir = '/disk3/wjr/dataset/NIHorg_dataset/vis_448_process_temp/'
    os.makedirs(output_dir, exist_ok=True)
    path_seg_org = output_dir + 'seg_rec_image'
    os.makedirs(path_seg_org, exist_ok=True)
    path_mask_org = output_dir + 'seg_rec_mask'
    os.makedirs(path_mask_org, exist_ok=True)

    contador = 0
    with torch.no_grad():
        for image in all_files:
            print('\r', contador + 1, 'of', len(all_files), end='')
            image_name = all_files[contador].split('/')[-1]
            image_path_seg_org = os.path.join(path_seg_org, image_name)
            image_mask_path_seg_org = os.path.join(path_mask_org, image_name)
            img_cv2 = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img_cv2 is None:
                print("Failed to read the image with OpenCV.")
                aa = Image.open(image).convert('L')  # 转换为灰度图像
                img_cv2 = np.array(aa)
            original_h, original_w = img_cv2.shape[:2]
            img = cv2.resize(img_cv2, (1024, 1024)) / 255.0
            resize_h, resize_w = img.shape[:2]
            data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).float()
            output = hybrid(data)
            if len(output) > 1:
                output = output[0]
            output = output.cpu().numpy().reshape(-1, 2) * 1024
            output = output.round().astype('int')
            contador += 1

            RL = output[:44]
            LL = output[44:94]
            H = output[94:]
            ####RL表示右肺区，LL表示左肺区，H表示心脏，如果只需要特定部位，可以在对应getDenseMask将其余部位注释掉，此处我们仅需要肺区，因此将心脏注释掉
            outseg, outseg_wheart = getDenseMask(RL, LL, H)

            imgseg = outseg_wheart * img
            # 查找二值图像中的轮廓
            contours, _ = cv2.findContours(outseg_wheart, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 初始化最大外接矩形的变量
            max_area = 0
            max_rect = None
            xmin = 1024
            ymin = 1024
            xmax = 0
            ymax = 0
            # 遍历所有轮廓
            for contour in contours:
                # 找到轮廓的外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                # result2[y:y+h, x:x+w] = img[y:y+h, x:x+w]
                x2 = x + w
                y2 = y + h
                # 计算外接矩形的面积
                if x < xmin:
                    xmin = x
                if y < ymin:
                    ymin = y
                if x2 > xmax:
                    xmax = x2
                if y2 > ymax:
                    ymax = y2
            # 打印最大外接矩形的信息
            result = np.zeros((1024, 1024), dtype=img.dtype)
            result[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]
            x_ori_min, y_ori_min = get_original_coordinates((xmin, ymin),
                                                            (resize_w, resize_h), (original_w, original_h))
            x_ori_max, y_ori_max = get_original_coordinates((xmax, ymax),
                                                            (resize_w, resize_h), (original_w, original_h))
            org_seg_img = img_cv2[int(math.floor(y_ori_min)):int(math.ceil(y_ori_max)),
                          int(math.floor(x_ori_min)):int(math.ceil(x_ori_max))]
            org_seg_img = cv2.resize(org_seg_img,
                                   ((448, 448)))
            cv2.imwrite(image_path_seg_org, org_seg_img)
            outsegorg = cv2.resize(np.uint8(outseg[ymin:ymax, xmin:xmax] * 255.0),
                                   ((448, 448)))
            cv2.imwrite(image_mask_path_seg_org, outsegorg)



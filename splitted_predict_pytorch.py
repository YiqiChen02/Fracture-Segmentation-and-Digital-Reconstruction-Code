import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
from nets import DSCNet_pro as DSCNet, GRC_DSCNet

from CrackImageClass import *
from BigCrackImageClass import *
from SplittedCrackImageClass import *


# 自定义Dataset类
class CrackDataset(Dataset):
    def __init__(self, image_folder):
        self.image_paths = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.endswith((".jpg", ".png", ".jpeg", ".webp")) 
            and not f.endswith(('predict.png','predict.jpg','predict.jpeg','predict.webp'))
        ]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        img = self.transform(img)
        return img, self.image_paths[idx]

def main():

    target_folder = r"E:\cyq\Unet\Unet_ob_exe2\data\对比\裂隙合集_pure3"
    save_folder = r"E:\cyq\Unet\Unet_ob_exe2\data\对比\裂隙合集_pure3"
    # target_color = "E05455"
    # rotation_angle = 0  # 以度数°为单位
    # origin = [-2.5, 0]
    BigCrackImage.target_folder = target_folder
    BigCrackImage.save_folder = save_folder

    # 加载PyTorch模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DSCNet(n_channels=1, n_classes=1, kernel_size=9, 
    #               extend_scope=1.0, if_offset=True, device=device, 
    #               number=16, dim=1).to(device)
    model = GRC_DSCNet(n_channels=1, n_classes=1, kernel_size=9, 
                  extend_scope=1.0, if_offset=True, device=device, 
                  number=16, dim=1).to(device)
    model.load_state_dict(torch.load("G-DSCNet_Best_pure.pth"))  # PyTorch模型文件
    model.eval()

    # 创建数据加载器
    dataset = CrackDataset(target_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (img_tensor, img_path) in enumerate(dataloader):
            img_tensor = img_tensor.to(device)
            
            # 处理大图
            f = os.path.basename(img_path[0])
            input_origin = "-5 , 6.640625"
            x, y = input_origin.split(",")
            origin = np.array([x, y], dtype=float)
            rotation_angle = float(0)
            real_distance = float(5)
            target_color = "D62132"
            
            big_img = BigCrackImage(f, origin, rotation_angle, real_distance, target_color)

            big_img.split_big_image()
            BigCrackImage.img_object_recorder.append(big_img)

            # 处理分割后的小图
            splitted_dataset = CrackDataset(big_img.splitted_images_folder)
            splitted_loader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)

            big_img_per_distance = big_img.per_pixel_real_distance
            SplittedCrackImage.target_folder = big_img.splitted_images_folder
            SplittedCrackImage.save_folder  = big_img.splitted_images_folder
            
            for j, (splitted_img_tensor, splitted_path) in enumerate(splitted_loader):
                splitted_img_tensor = splitted_img_tensor.to(device)
                splitted_img_pure_path = os.path.basename(splitted_path[0])
                
                # 预测
                output = model(splitted_img_tensor)
                result_for_splitted = output.squeeze().cpu().numpy()
                
                # 保存结果
                splitted_img = SplittedCrackImage(
                    splitted_img_pure_path, 
                    big_img.origin, 
                    big_img.rotation_angle, 
                    big_img.per_pixel_real_distance
                )
                splitted_img.predict_img_np = result_for_splitted
                splitted_img.save_predict_pic_result_for_image(result_for_splitted)
            
            # 拼接结果
            big_img.stitch_image()
            big_img.save_position_result()
            big_img.save_lines_result()

if __name__ == "__main__":
    main()
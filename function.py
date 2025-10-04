from __future__ import print_function
import math
import numpy as np 
from PIL import Image
import os
import matplotlib.pyplot as plt

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
not_crack = [255,255,255]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

COLOR_DICT_CRACK = np.array([Unlabelled,not_crack])

def color_distance(e1, e2):
    r_mean = (e1[0] + e2[0]) / 2
    r = e1[0] - e2[0]
    g = e1[1] - e2[1]
    b = e1[2] - e2[2]
    return math.sqrt((((512 + r_mean) * r * r) // 8) + 4 * g * g + (((767 - r_mean) * b * b) // 8))

def find_center_of_mass(points):
    # 计算一组点的质心坐标
    center = np.mean(points, axis=0)
    return center


def labelVisualize(num_class , color_dict ,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    # print(img)
    img[img > 0.5] = 1
    img[img <= 0.5] = 0
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]

    return img_out/255

def get_red_point_angel(point1,point2):
    
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    # 使用反正切函数计算夹角（注意要转换为度数）
    angle_rad = math.atan2(dy, dx)
    
    return angle_rad

def is_grayscale(image):
  
    return image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

def rgb_to_gray(rgb_img):
    # 使用加权平均法将 RGB 图像转换为灰度图像
    # 加权平均公式：gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_img = np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140])
    return gray_img

def hilditch(img):
    # get shape
    if len(img.shape) == 3:
        H, W, C = img.shape
    elif len(img.shape) == 2:
        H,W = img.shape

    # prepare out image
    out = np.zeros((H, W))
    out[img[..., 0] > 0] = 1
    plt.imsave(r"E:\cyq\Unet\Unet_ob_exe2\data\lines_test\test2\temp.png",out,cmap='gray')

    # inverse pixel value
    tmp = out.copy()
    _tmp = 1 - tmp

    count = 1
    while count > 0:
        count = 0
        tmp = out.copy()
        _tmp = 1 - tmp

        tmp2 = out.copy()
        _tmp2 = 1 - tmp2
        
        # each pixel
        for y in range(H):
            for x in range(W):
                # skip black pixel
                if out[y, x] < 1:
                    continue
                print(x, y)
                judge = 0
                
                ## condition 1
                if (tmp[y, min(x+1, W-1)] * tmp[max(y-1,0 ), x] * tmp[y, max(x-1, 0)] * tmp[min(y+1, H-1), x]) == 0:
                    judge += 1
                    
                ## condition 2
                c = 0
                c += (_tmp[y, min(x+1, W-1)] - _tmp[y, min(x+1, W-1)] * _tmp[max(y-1, 0), min(x+1, W-1)] * _tmp[max(y-1, 0), x])
                c += (_tmp[max(y-1, 0), x] - _tmp[max(y-1, 0), x] * _tmp[max(y-1, 0), max(x-1, 0)] * _tmp[y, max(x-1, 0)])
                c += (_tmp[y, max(x-1, 0)] - _tmp[y, max(x-1, 0)] * _tmp[min(y+1, H-1), max(x-1, 0)] * _tmp[min(y+1, H-1), x])
                c += (_tmp[min(y+1, H-1), x] - _tmp[min(y+1, H-1), x] * _tmp[min(y+1, H-1), min(x+1, W-1)] * _tmp[y, min(x+1, W-1)])
                if c == 1:
                    judge += 1
                    
                ## condition 3
                if np.sum(tmp[max(y-1, 0) : min(y+2, H), max(x-1, 0) : min(x+2, W)]) >= 3:
                    judge += 1

                ## condition 4
                if np.sum(out[max(y-1, 0) : min(y+2, H), max(x-1, 0) : min(x+2, W)]) >= 2:
                    judge += 1

                ## condition 5
                _tmp2 = 1 - out

                c = 0
                c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] * _tmp2[max(y-1, 0), min(x+1, W-1)] * _tmp2[max(y-1, 0), x])
                c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] * (1 - tmp[max(y-1, 0), max(x-1, 0)]) * _tmp2[y, max(x-1, 0)])
                c += (_tmp2[y, max(x-1, 0)] - _tmp2[y, max(x-1, 0)] * _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] * _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                if c == 1 or (out[max(y-1, 0), max(x-1,0 )] != tmp[max(y-1, 0), max(x-1, 0)]):
                    judge += 1

                c = 0
                c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] * _tmp2[max(y-1, 0), min(x+1, W-1)] * (1 - tmp[max(y-1, 0), x]))
                c += ((1-tmp[max(y-1, 0), x]) - (1 - tmp[max(y-1, 0), x]) * _tmp2[max(y-1, 0), max(x-1, 0)] * _tmp2[y, max(x-1, 0)])
                c += (_tmp2[y, max(x-1,0 )] - _tmp2[y, max(x-1,0 )] * _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] * _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                if c == 1 or (out[max(y-1, 0), x] != tmp[max(y-1, 0), x]):
                    judge += 1

                c = 0
                c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] * (1 - tmp[max(y-1, 0), min(x+1, W-1)]) * _tmp2[max(y-1, 0), x])
                c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] * _tmp2[max(y-1, 0), max(x-1, 0)] * _tmp2[y, max(x-1, 0)])
                c += (_tmp2[y, max(x-1, 0)] - _tmp2[y, max(x-1, 0)] * _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] * _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                if c == 1 or (out[max(y-1, 0), min(x+1, W-1)] != tmp[max(y-1, 0), min(x+1, W-1)]):
                    judge += 1

                c = 0
                c += (_tmp2[y, min(x+1, W-1)] - _tmp2[y, min(x+1, W-1)] * _tmp2[max(y-1, 0), min(x+1, W-1)] * _tmp2[max(y-1, 0), x])
                c += (_tmp2[max(y-1, 0), x] - _tmp2[max(y-1, 0), x] * _tmp2[max(y-1, 0), max(x-1, 0)] * (1 - tmp[y, max(x-1, 0)]))
                c += ((1 - tmp[y, max(x-1, 0)]) - (1 - tmp[y, max(x-1, 0)]) * _tmp2[min(y+1, H-1), max(x-1, 0)] * _tmp2[min(y+1, H-1), x])
                c += (_tmp2[min(y+1, H-1), x] - _tmp2[min(y+1, H-1), x] * _tmp2[min(y+1, H-1), min(x+1, W-1)] * _tmp2[y, min(x+1, W-1)])
                if c == 1 or (out[y, max(x-1, 0)] != tmp[y, max(x-1, 0)]):
                    judge += 1
                
                if judge >= 8:
                    out[y, x] = 0
                    count += 1
                    
    out = out.astype(np.uint8)*255

    return out

def adjacent_list(points):
    adjacent_dict = {}
    for point in points:
        x, y = point
        for dx,dy in [[0,1],[0,-1],[1,0],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]:
            adjacent_point = ( x + dx , y + dy)
            if adjacent_point in points:
                if point not in adjacent_dict:
                    adjacent_dict[point] = [adjacent_point]
                else:
                    adjacent_dict[point].append(adjacent_point)

    return adjacent_dict


def get_points_for_comp_white(img):
    white_positions = []
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if pixel >= 0.5:
                white_positions.append((j, 257 - i))  # Save white pixel position
    return white_positions


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def split_image(image, block_size=(256, 256)):
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[..., :3]
    elif len(image.shape) == 2:
        image = np.reshape(image,image.shape+(1,))

    height, width = image.shape[0],image.shape[1]
    blocks = []
    
    for y in range(0, height, block_size[1]):
        for x in range(0, width, block_size[0]):
            # 裁剪图像
            block = image[y:y + block_size[1], x:x + block_size[0]]
            # 如果块的尺寸小于256x256，则填充
            if block.shape[0] < block_size[1] or block.shape[1] < block_size[0]:
                if not is_grayscale(image):
                    new_block = np.full((256, 256, 3), 255)  # 使用白色填充
                else:
                    new_block = np.full((256, 256, 1), 255) 
                new_block[0:block.shape[0], 0:block.shape[1]] = block
                block = new_block
            
            blocks.append(block)
    
    return blocks

def convert_images_to_png(folder_path):
    # 确保文件夹路径存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在。")
        return
    
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # 如果是图片文件，进行转换
        if os.path.isfile(file_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            # 打开图像文件
            try:
                image = Image.open(file_path)
            except Exception as e:
                print(f"无法打开文件 '{file_name}': {e}")
                continue
            
            # 提取图像文件名和扩展名
            image_name, image_ext = os.path.splitext(file_name)
            
            # 转换为PNG格式并保存
            png_file_path = os.path.join(folder_path, f"{image_name}.png")
            try:
                image.save(png_file_path, "PNG")
                print(f"已将文件 '{file_name}' 转换为PNG格式,并保存为 '{png_file_path}'")
            except Exception as e:
                print(f"无法保存文件 '{file_name}': {e}")
                continue


if __name__ == "__main__":
    print("全部导入成功")
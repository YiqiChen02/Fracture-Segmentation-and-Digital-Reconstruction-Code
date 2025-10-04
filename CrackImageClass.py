from __future__ import print_function
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import math
from function import *


class CrackImage:

    image_num = 0
    target_folder = ""
    save_folder = ""
    img_object_recorder = []
    img_np_recorder = []
    predict_pure_path_recorder = []
    predict_path_recorder = []
    predict_pure_path_recorder_png = []
    predict_path_recorder_png = []

    def __init__(self,pure_path,origin,rotation_angle,real_distance,target_color) -> None:
        CrackImage.image_num += 1
        self.pure_path = pure_path
        self.origin = origin
        self.rotation_angle = rotation_angle
        self.real_distance = real_distance
        self.target_color = target_color
        self.img_np = self.get_img_np_info()[0]
        self.img_shape = self.get_img_np_info()[1]
        self.initial_img = self.get_img_np_info()[2]
        self.target_path = os.path.join(self.target_folder,self.pure_path)
        self.predict_pure_path = self.get_predict_pure_path()
        self.predict_path = os.path.join(self.save_folder,self.predict_pure_path)
        self.comp_distance = self.get_comp_distance()
        self.predict_white_SS_pure_path = self.get_predict_white_SS_pure_path()
        self.per_pixel_real_distance = self.get_per_pixel_real_distance()
        self.predict_img_np = None
        self.predict_CutL2_pure_path = self.get_predict_CutL2_pure_path()
        # self.points_for_comp_white = self.get_points_for_comp_white()


    def get_img_np_info(self,as_gray = False):
        target_path = os.path.join(self.target_folder,self.pure_path)
        img_np = io.imread(target_path,as_gray = as_gray)
        initial_img = img_np
        # test_path = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.png')]
        if  is_grayscale(img_np):
            img_np = io.imread(target_path,as_gray = not as_gray)
        else:
            img_np = rgb_to_gray(img_np)
        return img_np/255,img_np.shape,initial_img

    @classmethod
    def testGenerator(cls,target_size = (256,256),flag_multi_class = False):
        if cls.image_num:
            for img_np in cls.img_np_recorder:
                img_np = trans.resize(img_np,target_size)
                img_np = np.reshape(img_np,img_np.shape+(1,)) if (not flag_multi_class) else img_np
                img_np = np.reshape(img_np,(1,)+img_np.shape)
                yield img_np
        else:
            raise ValueError("该文件夹中无指定图片文件噢~")

    def testGenerator_for_image(self, target_size=(256,256), flag_multi_class = False):
        img_np = self.img_np
        img_np = trans.resize(img_np,target_size)
        img_np = np.reshape(img_np,img_np.shape+(1,)) if (not flag_multi_class) else img_np
        img_np = np.reshape(img_np,(1,)+img_np.shape)
        return img_np


    def find_color_blocks(self, threshold= 30000):
        img = Image.open(self.target_path)
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        # 将目标颜色转换为RGB模式
        target_color = tuple(int(self.target_color[i:i+2], 16) for i in (0, 2, 4))
        color_pixels = []
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                pixel_color = img_array[i, j]
                # 计算与目标颜色的欧氏距离
                distance =  color_distance(pixel_color, target_color)
                # 如果距离小于阈值，则将该像素坐标加入列表
                if distance < threshold:
                    color_pixels.append((i, j))
        
        # 将像素坐标按列进行排序，以便区分左右两个色块
        sorted_indices = np.argsort([j for _, j in color_pixels])

        # 将像素坐标分为左右两个色块
        sorted_pixels = [color_pixels[idx] for idx in sorted_indices]
        left_block = np.array(sorted_pixels[:len(sorted_pixels)//2])
        right_block = np.array(sorted_pixels[len(sorted_pixels)//2:])
        
        return left_block, right_block

    def get_comp_distance(self):
        left_block, right_block = self.find_color_blocks()
        # print("left_block:",left_block)
        # print("right_block:",right_block)
        # 计算左右两个色块的中心点
        left_center = find_center_of_mass(left_block)
        right_center = find_center_of_mass(right_block)
        # print("left_center:",left_center)
        # print("right_center:",right_center)

        comp_distance = self.calculate_distance_between_points(left_center,right_center)
        return comp_distance


    def calculate_distance_between_points(self,point1, point2):
        angel = get_red_point_angel(point1, point2)
        shape = self.img_shape
        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        if type(self) is CrackImage:
            distance = np.sqrt((distance*math.sin(angel)*256/shape[1])**2+(distance*math.cos(angel)*256/shape[0])**2)
        return distance

    def get_predict_pure_path(self):
        splitted_path = self.pure_path.split(".")
        splitted_path.insert(1,"_predict.")
        predict_pure_path = "".join(splitted_path)
        CrackImage.predict_pure_path_recorder.append(predict_pure_path)
        predict_path = os.path.join(self.save_folder,predict_pure_path)
        CrackImage.predict_path_recorder.append(predict_path)
        return predict_pure_path

    def get_predict_pure_path_png(self):
        splitted_path = self.pure_path.split(".")
        del splitted_path[1]
        splitted_path.insert(1,"_predict.png")
        predict_pure_path_png = "".join(splitted_path)
        CrackImage.predict_pure_path_recorder_png.append(predict_pure_path_png)
        predict_path_png = os.path.join(self.save_folder,predict_pure_path_png)
        CrackImage.predict_path_recorder_png.append(predict_path_png)
        return predict_pure_path_png
    
    def get_predict_white_SS_pure_path(self):
        splitted_path = self.pure_path.split(".")
        del splitted_path[1]
        splitted_path.insert(1,"_predict_white_.SS")
        predict_white_SS_path = "".join(splitted_path)
        return predict_white_SS_path
    
    def get_predict_CutL2_pure_path(self):
        split_string = self.pure_path.split(".")
        del split_string[1]
        split_string.insert(1,"_predict_lines.CutL2")
        predict_CutL2_pure_path = "".join(split_string)
        return predict_CutL2_pure_path


    @classmethod
    def save_predict_pic_results(cls, results, num_classes = 2):
        for i,item in enumerate(results):
            img = labelVisualize(num_classes,COLOR_DICT_CRACK,item)
            plt.imsave(cls.predict_path_recorder[i],img)

    def save_predict_pic_result_for_image(self,result,num_classes = 2):
        img = labelVisualize(num_classes,COLOR_DICT_CRACK,result)
        plt.imsave(self.predict_path,img)


    def get_per_pixel_real_distance(self):
        return self.real_distance/self.comp_distance

    def save_position_result(self):
        predict_white_SS_pure_path = self.predict_white_SS_pure_path
        result = self.predict_img_np
        black_positions = []
        white_positions = []
        origin_x = self.origin[0]
        origin_y = self.origin[1]
        print(origin_x,origin_y ,"------")
        print(self.img_shape[0],self.img_shape[1])
        real_pos = [0, 0]
        per_pixel_real_distance = self.per_pixel_real_distance
        theta = np.radians(self.rotation_angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        for i, row in enumerate(result):
            for j, pixel in enumerate(row):
                if pixel <= 0.5:
                    
                    black_positions.append([ j, 256 +1 - i])  # Save black pixel position
                else:
                    white_positions.append([ j, 256 +1 - i])  # Save white pixel position
        
        # Write black pixel positions to a file
        # with open(os.path.join(CrackImage.save_folder,predict_white_SS_pure_path), 'w') as f:
        #     for pos in black_positions:
        #         real_pos = np.array([
        #             pos[0] * per_pixel_real_distance + origin_x,
        #             pos[1] * per_pixel_real_distance + origin_y
        #         ])
        #         rotated_real_pos = np.dot(rotation_matrix,real_pos)
        #         f.write(f"({pos[0]}, {pos[1]}),({rotated_real_pos[0]},{rotated_real_pos[1]}),({per_pixel_real_distance})\n")
        
        # Write white pixel positions to a file
        with open(os.path.join(self.save_folder,predict_white_SS_pure_path), 'w') as f:
            for pos in white_positions:
                real_pos = np.array([
                    pos[0] * per_pixel_real_distance + origin_x,
                    pos[1] * per_pixel_real_distance + origin_y
                ])
                rotated_real_pos = np.dot(rotation_matrix,real_pos)
                # f.write(f"({pos[0]}, {pos[1]}),({rotated_real_pos[0]},{rotated_real_pos[1]}),({per_pixel_real_distance})\n")
                f.write(f"({rotated_real_pos[0]:.15f},{rotated_real_pos[1]:.15f},{per_pixel_real_distance:.15f})\n")

    def save_position_result_256(self):
            predict_white_SS_pure_path = self.predict_white_SS_pure_path
            result = self.predict_img_np
            black_positions = []
            white_positions = []
            origin_x = self.origin[0]
            origin_y = self.origin[1]
            print(origin_x,origin_y ,"------")
            real_pos = [0, 0]
            per_pixel_real_distance = self.per_pixel_real_distance
            theta = np.radians(self.rotation_angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            for i, row in enumerate(result):
                for j, pixel in enumerate(row):
                    if pixel <= 0.5:
                        black_positions.append([ j, 256+1 - i])  # Save black pixel position
                    else:
                        white_positions.append([ j, 256+1 - i])  # Save white pixel position

            with open(os.path.join(self.save_folder,predict_white_SS_pure_path), 'w') as f:
                for pos in white_positions:
                    real_pos = np.array([
                        pos[0] * per_pixel_real_distance + origin_x,
                        pos[1] * per_pixel_real_distance + origin_y
                    ])
                    rotated_real_pos = np.dot(rotation_matrix,real_pos)
                    # f.write(f"({pos[0]}, {pos[1]}),({rotated_real_pos[0]},{rotated_real_pos[1]}),({per_pixel_real_distance})\n")
                    f.write(f"({rotated_real_pos[0]:.15f},{rotated_real_pos[1]:.15f},{per_pixel_real_distance:.15f})\n")


    def save_lines_result(self):
        predict_img_np = self.predict_img_np
        predict_img_np = labelVisualize(2 , COLOR_DICT_CRACK, predict_img_np)
        predict_img_np_2n = predict_img_np[:,:,0]>0.5
        # thinned_image = hilditch(predict_img_np)
        thinned_image = skeletonize(predict_img_np_2n)
        thinned_image = thinned_image.astype(np.float64)
        thinned_points = get_points_for_comp_white(thinned_image)
        adjacent_dict = adjacent_list(thinned_points)
        self.thinned_points_to_lines(adjacent_dict)

    def thinned_points_to_lines(self,adjacent_dict):
        origin_x = self.origin[0]
        origin_y = self.origin[1]
        per_pixel_real_distance = self.per_pixel_real_distance
        theta = np.radians(self.rotation_angle)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        with open(os.path.join(self.save_folder,self.predict_CutL2_pure_path),'w') as f:
            for key, values in adjacent_dict.items():
                real_key = np.array([
                    key[0] * per_pixel_real_distance + origin_x,
                    key[1] * per_pixel_real_distance + origin_y
                ])
                rotated_key = np.dot(rotation_matrix,real_key)
                for value in values:
                    real_value = np.array([
                        value[0] * per_pixel_real_distance + origin_x,
                        value[1] * per_pixel_real_distance + origin_y
                    ])
                    rotated_value = np.dot(rotation_matrix,real_value)
                    f.write(f"({rotated_key[0]:.15f},{rotated_key[1]:.15f}),({rotated_value[0]:.15f},{rotated_value[1]:.15f})\n")
                    f.write(f";\n")


# if __name__ == "__main__":
#     target_folder = "data/daily_test/test_0516_10"
#     save_folder = "data/daily_test/test_0516_10"
#     target_color = "FF2401"
#     rotation_angle = 0 #以度数°为单位
#     origin = [0,0]

#     CrackImage.target_folder = target_folder
#     CrackImage.save_folder = save_folder

#     for i,f in enumerate(os.listdir(target_folder)):
#         if f.endswith((".jpg", ".png", ".jpeg", ".webp")) and not f.endswith(('predict.png','predict.jpg','predict.jpeg','predict.webp')):
#             img_path = os.path.join(target_folder,f)
#             with open(img_path,"r") as file:

#                 # input_origin = input("请输入第%d张图片的原点坐标x,y:" % (i+1))
#                 input_origin = "0,0"
#                 x,y = input_origin.split(",")
#                 origin = np.array([x, y], dtype=float)

#                 rotation_angle = "0"
#                 # rotation_angle = input("请输入第%d张图片的旋转角度(以度数°为单位):" % (i+1))
#                 rotation_angle = float(rotation_angle)

#                 real_distance = "1000"
#                 # real_distance = input("请输入第%d张图片中标记点之间的真实距离:" % (i+1))
#                 real_distance = float(real_distance)

#                 target_color = "FF1100"
#                 # target_color = input("请输入第%d张图片中标记颜色:" % (i+1))

#                 img = CrackImage(f,origin,rotation_angle,real_distance,target_color)
#                 img_np = img.img_np
#                 CrackImage.img_object_recorder.append(img)
#                 CrackImage.img_np_recorder.append(img_np)
                
#     model = unet()
#     model.load_weights("Unet_Positive_Negative_membrane.keras")
#     testGene = CrackImage.testGenerator()
#     results = model.predict(testGene,CrackImage.image_num,verbose = 1)
#     CrackImage.save_predict_pic_results(results)
    
    
#     for item,result in zip(CrackImage.img_object_recorder,results):
#         print(result.shape)
#         item.predict_img_npc1 = result
#         result = result[:,:,0] if len(result.shape)==3 else result
#         item.predict_img_np = result
#         item.save_position_result()
#         item.save_lines_result()
    
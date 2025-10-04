from CrackImageClass import *
# from model import *
from function import *

class BigCrackImage(CrackImage):
    image_num = 0
    target_folder = ""
    save_folder = ""
    img_object_recorder = []
    img_np_recorder = []
    predict_pure_path_recorder = []
    predict_path_recorder = []
    # splitted_images_ob_recorder = []
    # splitted_images_np_recorder = []
    
    def __init__(self, pure_path, origin, rotation_angle, real_distance, target_color) -> None:
        BigCrackImage.image_num += 1 
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
        self.predict_img_np = None  #对于大尺寸图像，该属性在粘结小预测图的过程中得到,已经过灰度化处理
        self.predict_CutL2_pure_path = self.get_predict_CutL2_pure_path()
        

        self.splitted_images_np_recorder = None
        self.splitted_images_ob_recorder = None
        self.splitted_images_pure_folder = self.get_splitted_images_pure_folder()
        self.splitted_images_folder = os.path.join(self.save_folder,self.splitted_images_pure_folder)
        

    def get_splitted_images_pure_folder(self):
        splitted_path = self.pure_path.split(".")
        del splitted_path[1]
        splitted_path.insert(1,"_splitted_images")
        splitted_images_pure_folder = "".join(splitted_path)
        return splitted_images_pure_folder

    # 该函数用于分割大尺寸图像，并保存分割后的图像
    def split_big_image(self):
        if not os.path.exists(self.splitted_images_folder):
            os.makedirs(self.splitted_images_folder)
        initial_img = self.initial_img
        splitted_images = split_image(initial_img)
        for i,image in enumerate(splitted_images):
            plt.imsave(os.path.join(self.splitted_images_folder,f"split_image{i}.png"), image/255)       
    
    def get_info_from_big_for_splitted(self,target_size= (256,256)):
        img_shape = self.img_shape
        heigh, width = img_shape[:2]
        num_heigh = math.ceil(heigh / target_size[1])
        num_width = math.ceil(width / target_size[0])
        return num_heigh,num_width

    def get_predict_pure_path(self):
        splitted_path = self.pure_path.split(".")
        splitted_path.insert(1,"_predict.")
        predict_pure_path = "".join(splitted_path)
        BigCrackImage.predict_pure_path_recorder.append(predict_pure_path)
        predict_path = os.path.join(self.save_folder,predict_pure_path)
        BigCrackImage.predict_path_recorder.append(predict_path)
        return predict_pure_path
    
    def stitch_image(self):
        num_split_image = self.get_info_from_big_for_splitted()
        origin_image_shape = self.img_shape
        combined_image = np.zeros((num_split_image[0] * 256, num_split_image[1] * 256, 3), dtype=np.uint8)
        for i in range(num_split_image[0]):  # 遍历行
            for j in range(num_split_image[1]):  # 遍历列
                img_path = os.path.join(self.splitted_images_folder, f"split_image{i * num_split_image[1] + j}_predict.png")
                if os.path.exists(img_path):
                    img = io.imread(img_path)
                    start_row = i * 256
                    start_col = j * 256
                    if img.shape[2] == 4:
                        img = img[..., :3] 
                    combined_image[start_row:start_row + 256, start_col:start_col + 256, :] = img  # 放入正确位置
        combined_image = combined_image[:origin_image_shape[0],:origin_image_shape[1]]
        self.predict_img_np = rgb_to_gray(combined_image)
        # 保存组合后的图像
        io.imsave(self.predict_path, combined_image)
    
    

# if __name__ == "__main__":
#     target_folder = "data/membrane/test_0512_split"
#     save_folder = "data/membrane/test_0512_split"
#     target_color = "FF2401"
#     rotation_angle = 0 #以度数°为单位
#     origin = [0,0]

#     BigCrackImage.target_folder = target_folder
#     BigCrackImage.save_folder = save_folder

#     model = unet()
#     model.load_weights("Unet_Positive_Negative_membrane.keras")

#     for i,f in enumerate(os.listdir(target_folder)):
#         if f.endswith((".jpg", ".png", ".jpeg", ".webp")) and not f.endswith(('predict.png','predict.jpg','predict.jpeg','predict.webp')):
#             img_path = os.path.join(target_folder,f)
#             # input_origin = input("请输入第%d张图片的原点坐标x,y:" % (i+1))
#             input_origin = "-2.500,0"
#             x,y = input_origin.split(",")
#             origin = np.array([x, y], dtype=float)

#             rotation_angle = "0"
#             # rotation_angle = input("请输入第%d张图片的旋转角度(以度数°为单位):" % (i+1))
#             rotation_angle = float(rotation_angle)

#             real_distance = "2.500"
#             # real_distance = input("请输入第%d张图片中标记点之间的真实距离:" % (i+1))
#             real_distance = float(real_distance)

#             target_color = "FF2401"
#             # target_color = input("请输入第%d张图片中标记颜色:" % (i+1))
            
#             big_img = BigCrackImage(f,origin,rotation_angle,real_distance,target_color)
#             big_img.split_big_image()
#             BigCrackImage.img_object_recorder.append(big_img)
#             big_img_per_distance = big_img.per_pixel_real_distance
#             for i,splitted_img_pure_path in enumerate(os.listdir(big_img.splitted_images_folder)):
#                 if splitted_img_pure_path.endswith((".jpg", ".png", ".jpeg", ".webp")) and not f.endswith(('predict.png','predict.jpg','predict.jpeg','predict.webp')):
#                     splitted_img_path = os.path.join(big_img.splitted_images_folder,splitted_img_pure_path)
#                     splitted_img = SplittedCrackImage(splitted_img_pure_path,big_img.origin,big_img.rotation_angle,big_img_per_distance)
                    
#                     big_img.splitted_images_ob_recorder.append(splitted_img)

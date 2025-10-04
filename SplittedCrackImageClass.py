from BigCrackImageClass import *


class SplittedCrackImage(BigCrackImage):
    image_num = 0
    target_folder = ""
    
    def __init__(self, pure_path, origin, rotation_angle,big_img_per_distance) -> None:
        SplittedCrackImage.image_num += 1
        self.pure_path = pure_path
        self.origin = origin
        self.rotation_angle = rotation_angle
        # self.real_distance = real_distance
        # self.target_color = target_color
        self.img_np = self.get_img_np_info()[0]
        self.img_shape = self.get_img_np_info()[1]
        self.initial_img = self.get_img_np_info()[2]
        self.target_path = os.path.join(self.target_folder,self.pure_path)
        
        self.predict_pure_path = self.get_predict_pure_path()
        self.predict_path = os.path.join(self.save_folder,self.predict_pure_path)
        # self.comp_distance = self.get_comp_distance()
        self.predict_white_SS_pure_path = self.get_predict_white_SS_pure_path()
        self.per_pixel_real_distance = big_img_per_distance
        self.predict_img_np = None
        self.predict_CutL2_pure_path = self.get_predict_CutL2_pure_path()
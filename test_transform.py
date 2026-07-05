import cv2
import numpy as np
import random
import rasterio
from rasterio.windows import Window

class RegistrationSampleGenerator:
    def __init__(self, image_size=256, displacement_range=5):
        """
        初始化生成器
        
        Args:
            image_size: 输入图像的尺寸 (论文中使用的是 256x256)
            displacement_range: 论文中提到的 "specific range"，即角点可以移动的最大像素数
        """
        self.image_size = image_size
        self.displacement_range = displacement_range
        # 论文中提到的四个固定位置的点 (F1, F2, F3, F4)
        # 坐标对应关系: (20,20), (236,20), (236,236), (20,236) for a 256x256 image
        self.src_points = np.float32([[20, 20], 
                                     [image_size-20, 20], 
                                     [image_size-20, image_size-20], 
                                     [20, image_size-20]])
        
        # 定义最大和最小缩放比例（模拟传感器的缩放变化）
        self.scale_min, self.scale_max = 0.9, 1.1
        # 定义最大旋转角度（模拟传感器的旋转）
        self.rotation_max = 5 # 度

    def generate_random_transformation(self):
        """
        步骤1: 随机生成变换矩阵 (PTM)
        论文提到: "The PTM is first generated randomly within a specific range"
        
        Returns:
            M: 3x3 透视变换矩阵
            true_displacements: 8维真值向量 (dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4)
        """
        h, w = self.image_size, self.image_size
        
        # --- 模拟真实的几何变换 ---
        # 1. 随机缩放
        scale = random.uniform(self.scale_min, self.scale_max)
        
        # 2. 随机旋转 (围绕图像中心)
        center = (w // 2, h // 2)
        angle = random.uniform(-self.rotation_max, self.rotation_max)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 3. 随机平移
        tx = random.randint(-self.displacement_range, self.displacement_range)
        ty = random.randint(-self.displacement_range, self.displacement_range)
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty
        
        # 4. 先将四个源点做仿射变换，再叠加透视扰动
        # 这样旋转/缩放/平移会真正影响最终的透视变换矩阵
        affine_src_points = cv2.transform(self.src_points.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)

        # 生成目标点 (dst_points)
        dst_points = affine_src_points.copy()
        for i in range(4):
            # 在仿射变换后的点基础上增加随机位移，形成透视畸变
            dx = random.randint(-self.displacement_range, self.displacement_range)
            dy = random.randint(-self.displacement_range, self.displacement_range)
            dst_points[i] = affine_src_points[i] + [dx, dy]
        
        # --- 计算变换矩阵 ---
        # 论文公式 (12) - (17) 对应的 OpenCV 实现
        M = cv2.getPerspectiveTransform(self.src_points, dst_points)
        
        # --- 计算真值位移 ---
        # 论文: "The true regression sample value (Δx, Δy) is used to determine..."
        true_displacements = []
        for src, dst in zip(self.src_points, dst_points):
            true_displacements.extend([dst[0] - src[0], dst[1] - src[1]])
        
        return M, np.array(true_displacements)

    def create_sample_pair(self, input_image):
        """
        步骤2 & 3: 创建样本对
        
        Args:
            input_image: 原始图像 (numpy array, HxWxC)
            
        Returns:
            ref_img: 参考图像 (经过预处理的 input_image)
            sensed_img: 感知图像 (经过变换后的图像)
            gt_displacement: 真值位移向量
        """
        h, w = self.image_size, self.image_size
        
        # 1. 生成变换矩阵和真值
        M, gt_displacement = self.generate_random_transformation()
        
        # 2. 读取并预处理图像 (模拟论文中的 P1 和 P2)
        # 假设 input_image 已经是 1m 分辨率且对齐的
        # 这里简单进行 resize 和归一化
        img = cv2.resize(input_image, (w, h))
        
        # 3. 生成 "感知图像" (Sensed Image)
        # 论文: "distort one of the pixel-aligned images P1 and P2"
        sensed_img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # 4. 裁剪参考图像 (模拟论文中的 "cropped between the reference and sensed image")
        # 为了简单起见，这里直接使用原图作为参考，实际训练中可能需要根据 M 矩阵裁剪对应的区域
        ref_img = img.copy()
        
        return ref_img, sensed_img, gt_displacement

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 初始化生成器
    # 论文中图像尺寸为 256x256，位移范围根据 RMSE 指标估计在 50像素以内
    generator = RegistrationSampleGenerator(image_size=256, displacement_range=5)
    
    # 2. 使用 rasterio 从指定遥感图像中随机读取一个 256x256 patch
    image_path = r"E:\数据集\Houston2013\2013_IEEE_GRSS_DF_Contest_LiDAR.tif"
    crop_size = 256
    with rasterio.open(image_path) as dataset:
        if dataset.width < crop_size or dataset.height < crop_size:
            raise ValueError(f"源图像尺寸过小: {(dataset.height, dataset.width)}, 需要至少 {crop_size}x{crop_size}")

        max_y = dataset.height - crop_size
        max_x = dataset.width - crop_size
        top = random.randint(0, max_y)
        left = random.randint(0, max_x)

        window = Window(left, top, crop_size, crop_size)
        source_image = dataset.read(window=window)

    if source_image.shape[0] == 1:
        dummy_image = source_image[0]
    else:
        dummy_image = np.transpose(source_image, (1, 2, 0))
    
    # 3. 生成一个训练样本对
    ref, sensed, true_disp = generator.create_sample_pair(dummy_image)
    
    print("生成的真值位移参数 (8维):", true_disp)
    print("参考图像形状:", ref.shape)
    print("感知图像形状:", sensed.shape)
    
    # 4. (可选) 可视化结果
    # 将两张图水平拼接，可以看到变换效果
    comparison = np.hstack((ref, sensed))
    cv2.imshow("Left: Reference, Right: Sensed (Warped)", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
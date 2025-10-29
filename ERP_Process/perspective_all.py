import os
import cv2
import Equirec2Perspec as E2P

def process_panorama(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否是PNG或JPG格式
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(input_folder, filename)
            # 加载等距圆柱投影图像
            equ = E2P.Equirectangular(input_path)

            # 文件名（无扩展名）用于创建子文件夹
            base_filename = os.path.splitext(filename)[0]
            # 生成子文件夹路径，这里假设文件夹已经是以数字序号命名
            output_subfolder = os.path.join(output_folder, base_filename)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            # 定义输出图像的尺寸
            output_height = 512
            output_width = 512

            # 定义六个视角的参数：FOV, theta, phi
            views = {
                '1': (90, 0, 0),
                '2': (90, 45, 0),
                '3': (90, 90, 0),
                '4': (90, 135, 0),
                '5': (90, 180, 0),
                '6': (90, 225, 0),
                '7': (90, 270, 0),
                '8': (90, 315, 0)
            }

            # 生成并保存每个视角的图像
            for view_name, (fov, theta, phi) in views.items():
                img = equ.GetPerspective(fov, theta, phi, output_height, output_width)
                output_path = os.path.join(output_subfolder, f'{view_name}.png')
                cv2.imwrite(output_path, img)
            print(f"Processed {filename} and saved views to {output_subfolder}")

if __name__ == '__main__':
    input_folder = '/DATA/DATA1/yangliu/data/feedback_all_ts50/sal'  # 替换为实际的输入文件夹路径
    output_folder = '/DATA/DATA1/yangliu/data/feedback_all_ts50/sal_split'  # 替换为实际的输出文件夹路径
    process_panorama(input_folder, output_folder)
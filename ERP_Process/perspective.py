import os
import cv2
import Equirec2Perspec as E2P



class get_perspective():
    def forward(self, img_360_path):
        equ = E2P.Equirectangular(img_360_path)
        output_height = 512
        output_width = 512

        views = {
            'front': (110, 0, 0),
            'left': (110, 90, 0),
            'right': (110, -90, 0),
            'back': (110, 180, 0),
            'up': (110, 0, 90),
            'down': (110, 0, -90)
        }
        img = []
        for view_name, (fov, theta, phi) in views.items():
                img = equ.GetPerspective(fov, theta, phi, output_height, output_width)
                output_path = os.path.join('/DATA/DATA1/yangliu/code/ERP_Process', f'{view_name}.png')
                cv2.imwrite(output_path, img)
        return img

if __name__ == '__main__':
    # 加载等距圆柱投影图像
    '''
    equ = E2P.Equirectangular('/DATA/DATA1/yangliu/CNNIQA/dataset1/indoor_5_inpaint1.bmp')

    # 定义输出图像的尺寸
    output_height = 1024
    output_width = 1024

    # 定义六个视角的参数：FOV, theta, phi
    views = {
        'front': (90, 0, 0),
        'left': (90, 90, 0),
        'right': (90, -90, 0),
        'back': (90, 180, 0),
        'up': (90, 0, 90),
        'down': (90, 0, -90)
    }

    # 生成并保存每个视角的图像
    for view_name, (fov, theta, phi) in views.items():
        img = equ.GetPerspective(fov, theta, phi, output_height, output_width)
        cv2.imwrite(f'/DATA/DATA1/yangliu/code/Equirec2Perspec-master/{view_name}.png', img)
    '''
    perspective = get_perspective()
    perspective.forward('/DATA/DATA1/yangliu/data/pano600/outdoornew_4_text2light1.bmp')

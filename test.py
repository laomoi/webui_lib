
import webui_lib

from PIL import Image
import numpy as np

def start():
    webui_lib.initialize()
    print(webui_lib.get_cn_model_list())
    #lora
    #images = webui_lib.txt2img({'prompt': "box <lora:Hgame_ICON:1>"} )
    #grid
    #images = webui_lib.txt2img({'prompt': "1girl"},  "X/Y/Z plot", [4, '10,20', 17, '0.2,0.5', 0, '', True, False, False, False, 0], None)
    # (True, 'canny', 'control_canny [9d312881]', 1, {'image': array(), 'mask': array()}, False, 'Scale to Fit (Inner Fit)', False, False, 512, 100, 200, 0, 1, False,

    #controlnet
    png = Image.open('./1.png')
    mask = Image.new("RGB",(png.width,png.height),(0,0,0,255))
    canny_model_name = webui_lib.get_cn_model_name('control_canny')
    if canny_model_name is None:
        canny_model_name = webui_lib.get_cn_model_name('control_sd15_canny')
    if canny_model_name is None:
        print("cannot find canny model in controlnet")
    else:
        print("find canny control:",canny_model_name)
    images = webui_lib.txt2img({'prompt': "1girl", 'width':512, 'height':512}, None, None, [
        {
            'enabled': True,
            'module': 'canny',
            'model': canny_model_name,
            'weight': 1,
            'image': {'image': np.array(png), 'mask':np.array(mask)},
            'scribble_mode': False,
            'resize_mode': "Scale to Fit (Inner Fit)",
            'rgbbgr_mode': False,
            'lowvram': False,
            'pres': 512,
            'pthr_a': 100,
            'pthr_b': 200,
            'guidance_start': 0,
            'guidance_end': 1,
            'guess_mode': False
        }
    ])

    i = 1
    for image in images:
        webui_lib.save_image(image, "./test_" + str(i)  + ".png" )
        print("saved to " + "./test_" + str(i)  + ".png" )
        i = i + 1
    #grid script
    #xyz_grid args:
    #x_type, x_values, y_type, y_values, z_type, z_values, draw_legend, include_lone_images, include_sub_grids, no_fixed_seeds, margin_size
    # webui_lib.txt2img({'prompt': "1girl"}, "./test", "X/Y/Z plot", [4, '10,20', 17, '0.2,0.5', 0, '', True, False, False, False, 0], None)

if __name__ == "__main__":
    start()

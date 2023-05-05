import sys
import os


sd_root_path = ""



parent_dir_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if sd_root_path == "":
    sd_root_path = os.path.join(parent_dir_path, 'stable-diffusion-webui')

print("sd_root_path", sd_root_path)
sys.path.append(sd_root_path)
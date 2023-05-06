import sys
import os
import configparser
class MyParser(configparser.ConfigParser):
    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d
def get_config(config_path: str):
    config = MyParser()
    config.read(config_path, encoding='utf-8')
    return config.as_dict()

config_file =  os.path.join( os.path.abspath(os.path.dirname(__file__)), "webui_config.ini" )
config_data = get_config(config_file)

sd_root_path = ""
if config_data['path']['sd_root_path'] == "":
    parent_dir_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    sd_root_path = os.path.join(parent_dir_path, 'stable-diffusion-webui')
else:
    sd_root_path = config_data['path']['sd_root_path']


print("sd_root_path", sd_root_path)
sys.path.append(sd_root_path)
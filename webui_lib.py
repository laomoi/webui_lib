
import webui_config
import sys
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
import modules.shared as shared
from PIL import PngImagePlugin
from modules import script_callbacks, extra_networks
from modules import  scripts
from scripts.xyz_grid import AxisOption,axis_options
from modules.scripts import ScriptRunner
import modules.sd_models as sd_models
from modules import extensions

is_old_cn_version_before_4 = False #old controlnet has different params and process args
is_imported_controlnet = False
controlnet_model_list = []

def initialize():
    import webui
    print("load model", shared.opts.sd_model_checkpoint)
    webui.initialize()
    script_callbacks.before_ui_callback()


def get_checkpoint_list():
    return sd_models.checkpoints_list

def get_cn_model_list():
    global is_imported_controlnet
    global controlnet_model_list
    if not is_imported_controlnet:
        for ext in extensions.extensions:
            if ext.name == 'sd-webui-controlnet':
                #import controlnet
                print(" controlnet path", ext.path)
                sys.path.append( ext.path)
                
                from scripts import global_state
                global_state.update_cn_models()
                controlnet_model_list = list(global_state.cn_models_names.values())
                is_imported_controlnet = True
                break
    return controlnet_model_list

def get_cn_model_name(start_str):
    model_list = get_cn_model_list()
    for m in model_list:
        if m.startswith(start_str):
            return m
    return None

def set_default_model(sd_model_name):
    shared.opts.sd_model_checkpoint = sd_model_name

def set_lora_dir(dir):
    shared.cmd_opts.lora_dir = dir

def reload_model(sd_model_name):
    shared.opts.sd_model_checkpoint = sd_model_name
    sd_models.reload_model_weights()

def txt2img(params, outer_script_name=None, outer_script_args=None, controlnets=None):
    args = {
        'enable_hr': False,
        'denoising_strength': 0.0,
        'firstphase_width': 0,
        'firstphase_height': 0,
        'hr_scale': 2.0,
        'hr_upscaler': None,
        'hr_second_pass_steps': 0,
        'hr_resize_x': 0,
        'hr_resize_y': 0,
        'prompt': '1girl,',
        'styles': None,
        'seed': -1,
        'subseed': -1,
        'subseed_strength': 0.0,
        'seed_resize_from_h': -1,
        'seed_resize_from_w': -1,
        'sampler_name': 'Euler a',
        'batch_size': 1,
        'n_iter': 1,
        'steps': 20,
        'cfg_scale': 7.0,
        'width': 512,
        'height': 512,
        'restore_faces': False,
        'tiling': False,
        'negative_prompt': '',
        'eta': 0.0,
        's_churn': 0.0,
        's_tmax': 0.0,
        's_tmin': 0.0,
        's_noise': 1.0,
        'override_settings': {},
        'override_settings_restore_afterwards': True,
        'script_args': [],
        'sampler_index': None,
        'do_not_save_samples': True,
        'do_not_save_grid': True
    }


    if params is not None :
        for k in params:
            if k in args:
                args[k] = params[k]

    scripts_runner = scripts.scripts_txt2img
    p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
    p.script_args = []
    return generate_img(p, scripts_runner, outer_script_name, outer_script_args, controlnets)


def img2img(params, outer_script_name=None, outer_script_args=None, controlnets=None):
    args = {
        'outpath_samples': shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
        'outpath_grids': shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
        'prompt': '',
        'negative_prompt': '',
        'styles': [],
        'seed': -1.0,
        'subseed': -1.0,
        'subseed_strength': 0,
        'seed_resize_from_h': 0,
        'seed_resize_from_w': 0,
        'seed_enable_extras': False,
        'sampler_name': 'Euler a',
        'batch_size': 1,
        'n_iter': 1,
        'steps': 20,
        'cfg_scale': 7,
        'width': 512,
        'height': 512,
        'restore_faces': False,
        'tiling': False,
        'override_settings': {},
        'init_images': None,
        'resize_mode':  0,
        'denoising_strength':  0.75,
        'image_cfg_scale':  1.5,
        'mask':  None,
        'mask_blur':  4,
        'inpainting_fill':  1,
        'inpaint_full_res':  False,
        'inpaint_full_res_padding':  32,
        'inpainting_mask_invert':  0,
        'initial_noise_multiplier':  shared.opts.initial_noise_multiplier
    }


    if params is not None :
        for k in params:
            if k in args:
                args[k] = params[k]

    scripts_runner = scripts.scripts_txt2img #scripts.scripts_img2img
    p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
    p.script_args = []
    return generate_img(p, scripts_runner, outer_script_name, outer_script_args, controlnets)


def generate_img(p, scripts_runner, outer_script_name=None, outer_script_args=None, controlnets=None):

    outer_script, outer_script_idx = get_script(outer_script_name, scripts_runner, scripts_runner.selectable_scripts)
    cn_script,cn_script_idx = get_script("controlnet", scripts_runner, scripts_runner.alwayson_scripts)

    merge_scripts(p, False, outer_script, outer_script_idx, outer_script_args, cn_script, controlnets)

    shared.state.begin()
    if outer_script is not None:
        p.outpath_grids = shared.opts.outdir_txt2img_grids
        p.outpath_samples = shared.opts.outdir_txt2img_samples
        init_script(outer_script, outer_script_name, False)
        processed = scripts_runner.run(p, *p.script_args)
    else:
        processed = process_images(p)
    shared.state.end()
    return processed.images

def merge_scripts(p, is_img2img, outer_script, outer_script_idx, outer_script_args, cn_script, controlnets):
    cn_args = []
    if controlnets is not None and len(controlnets) > 0 and cn_script is not None:
        p.scripts = ScriptRunner()
        p.scripts.alwayson_scripts.append(cn_script)
        if is_old_cn_version_before_4:
            cn_args.append(is_img2img)
            is_ui = False
            cn_args.append(is_ui)
        # set args to controlnet script
        keys = ['enabled', 'module', 'model', 'weight', 'image', 'scribble_mode', 'resize_mode', 'rgbbgr_mode',
                'lowvram', 'pres', 'pthr_a', 'pthr_b', 'guidance_start', 'guidance_end', 'guess_mode']
        for cn in controlnets:
            default_param = {
                'enabled': True,
                'module': 'None',
                'model': 'None',
                'weight': 1,
                'image': None,
                'scribble_mode': False,
                'resize_mode': "Scale to Fit (Inner Fit)",
                'rgbbgr_mode': False,
                'lowvram': False,
                'pres': 64,
                'pthr_a': 64,
                'pthr_b': 64,
                'guidance_start': 0,
                'guidance_end': 1,
                'guess_mode': False
            }
            for k in cn:
                if k in default_param:
                    default_param[k] = cn[k]
            for k in keys:
                cn_args.append(default_param[k])
            # print("input args:", default_param)
    # controlnet args
    # (True, 'canny', 'control_canny [9d312881]', 1, {'image': array(), 'mask': array()}, False, 'Scale to Fit (Inner Fit)', False, False, 512, 100, 200, 0, 1, False,
    # enabled, module, model, weight, image, scribble_mode, resize_mode, rgbbgr_mode, lowvram, pres, pthr_a, pthr_b, guidance_start, guidance_end, guess_mode

    # merge script args
    if outer_script is not None:
        p.script_args = [outer_script_idx + 1]

    related_scripts = [[outer_script, outer_script_args], [cn_script, cn_args]]
    for script_info in related_scripts:
        if script_info[0] is not None and script_info[1] is not None and len(script_info[1]) > 0:
            script_info[0].args_from = len(p.script_args)
            p.script_args = p.script_args + script_info[1]
            script_info[0].args_to = len(p.script_args)


def init_script(script, script_name, is_img2img):
    # xyz grid script need to init the axis_options
    if script_name == 'X/Y/Z plot':
        script.current_axis_options = [x for x in axis_options if type(x) == AxisOption or x.is_img2img == is_img2img]


def save_image(image, path):
    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True
    image.save(path, format="PNG", pnginfo=(metadata if use_metadata else None))

def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except:
        return None
def get_script(script_name, script_runner, scripts_set):
    if script_name is None:
        return None, None
    if not script_runner.scripts:
        script_runner.initialize_scripts(False)
    script_idx = script_name_to_index(script_name, scripts_set)
    if script_idx is not None:
        script = scripts_set[script_idx]
        return script, script_idx
    else:
        return None, None



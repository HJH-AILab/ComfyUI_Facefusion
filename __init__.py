from ast import Not
import os
import sys
import torch

import folder_paths

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append( f"{ROOT_DIR}/facefusion")

from .utils.helper import override_defs_for_comfyui
override_defs_for_comfyui()

GLOBAL_CATEGORY = "HJH_Facefusion🪅"

import facefusion
from facefusion import state_manager
from facefusion.filesystem import list_directory, is_video

BOOL_SET = [True, False]
import facefusion.choices as CHOICES_SET
from facefusion.processors import choices as PROCESSOR_CHOICES_SET


HF_URL = CHOICES_SET.download_provider_set["huggingface"]["urls"][0]
HF_M_URL = CHOICES_SET.download_provider_set["huggingface"]["urls"][1]
def init_facefusion_state_manager():
    facefusion.ROOT_DIR = f"{ROOT_DIR}/facefusion"

    state_manager.set_item('execution_providers', ["cpu"])

    available_processors = [ file.get('name') for file in list_directory('facefusion/processors/modules') ]
    state_manager.set_item('processors',available_processors)

    state_manager.set_item('download_providers',CHOICES_SET.download_providers)

    state_manager.set_item('temp_path',f"{ROOT_DIR}/facefusion/.temp")

    state_manager.set_item('temp_frame_format',"png") #[ 'bmp', 'jpg', 'png' ]

    state_manager.set_item('output_audio_volume', 100)



init_facefusion_state_manager()

DEFAULT_SET={
    "face_swpper": "inswapper_128",
    "face_enhancer": "gfpgan_1.4",
    "frame_enhancer":"span_kendata_x4",
    "lip_syncer":"wav2lip_gan_96",
    "face_detector":"yolo_face",
}

# 换脸
class FacefusionFaceSwapperProcessor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        default_model = DEFAULT_SET["face_swpper"]
        """定义输入参数"""
        return {
            "required": {
                "options":("FFPROCESSOROPTIONS", ),
                "model": (PROCESSOR_CHOICES_SET.face_swapper_models, {"default":default_model}),
                "pixel_boost":(PROCESSOR_CHOICES_SET.face_swapper_set[default_model], {"default":PROCESSOR_CHOICES_SET.face_swapper_set[default_model][0]}),
            },
        }
    
    RETURN_TYPES = ("FFFACESWAPPERPROCESSOR",)
    RETURN_NAMES = ("face_swapper_processor",)
    FUNCTION = "run"
    CATEGORY = GLOBAL_CATEGORY

    def run(self, options, model, pixel_boost):
        for key, value in options.items():
            if value == 'none':
                options[key] = None
            elif(key == 'execution_providers'):
                state_manager.set_item('execution_providers', [options.get('execution_providers')])
            else:
                state_manager.set_item(key,options[key])
        
        import facefusion.processors.modules.face_swapper as face_swapper_processor
        if face_swapper_processor.get_model_name() is not None:
            face_swapper_processor.clear_inference_pool()

        state_manager.set_item('face_swapper_model', model)
        state_manager.set_item('face_swapper_pixel_boost', pixel_boost)

        face_swapper_processor.pre_check()

        return model,

# 帧增强
class FacefusionFrameEnhancerProcessor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        default_model = DEFAULT_SET["frame_enhancer"]
        """定义输入参数"""
        return {
            "required": {
                "options":("FFPROCESSOROPTIONS", ),
                "model": (PROCESSOR_CHOICES_SET.frame_enhancer_models, {"default":default_model}),
                "frame_enhancer_blend":("INT", {"default":80,
                                                  "min": PROCESSOR_CHOICES_SET.frame_enhancer_blend_range[0],
                                                  "max": PROCESSOR_CHOICES_SET.frame_enhancer_blend_range[-1],
                                                  "step": PROCESSOR_CHOICES_SET.frame_enhancer_blend_range[1]-PROCESSOR_CHOICES_SET.frame_enhancer_blend_range[0],
                                                  "display":"slider"}),
            },
        }
    
    RETURN_TYPES = ("FFFRAMEENHANCERPROCESSOR",)
    RETURN_NAMES = ("frame_enhancer_processor",)
    FUNCTION = "run"
    CATEGORY = GLOBAL_CATEGORY

    def run(self, options, model, frame_enhancer_blend):
        for key, value in options.items():
            if value == 'none':
                options[key] = None
            elif(key == 'execution_providers'):
                state_manager.set_item('execution_providers', [options.get('execution_providers')])
            else:
                state_manager.set_item(key,options[key])

        import facefusion.processors.modules.frame_enhancer as frame_enhancer_processor
        if frame_enhancer_processor.get_frame_enhancer_model() is not None:
            frame_enhancer_processor.clear_inference_pool()

        state_manager.set_item('frame_enhancer_model', model)
        state_manager.set_item('frame_enhancer_blend', frame_enhancer_blend)

        frame_enhancer_processor.pre_check()
        return model,

# 人脸增强
class FacefusionFaceEnhancerProcessor:
    def __init__(self):
        # print("FACE_SWAPPER_SET:",FACE_SWAPPER_SET)
        # print("pixel_boost:",FACE_SWAPPER_SET[default_face_swapper])
        pass

    @classmethod
    def INPUT_TYPES(cls):
        default_model = DEFAULT_SET["face_enhancer"]
        
        """定义输入参数"""
        return {
            "required": {
                "options":("FFPROCESSOROPTIONS", ),
                "model": (PROCESSOR_CHOICES_SET.face_enhancer_models, {"default":default_model}),
                "face_enhancer_blend":("INT", {"default":80,
                                               "min": PROCESSOR_CHOICES_SET.face_enhancer_blend_range[0],
                                               "max": PROCESSOR_CHOICES_SET.face_enhancer_blend_range[-1],
                                               "step": PROCESSOR_CHOICES_SET.face_enhancer_blend_range[1]-PROCESSOR_CHOICES_SET.face_enhancer_blend_range[0],
                                               "display":"slider"}),
                "face_enhancer_weight":("FLOAT", {"default":1.0,
                                                  "min":PROCESSOR_CHOICES_SET.face_enhancer_weight_range[0],
                                                  "max":PROCESSOR_CHOICES_SET.face_enhancer_weight_range[-1],
                                                  "step":PROCESSOR_CHOICES_SET.face_enhancer_weight_range[1]-PROCESSOR_CHOICES_SET.face_enhancer_weight_range[0],
                                                  "display":"slider"}),
            },
        }
    
    RETURN_TYPES = ("FFFACEENHANCERPROCESSOR",)
    RETURN_NAMES = ("face_enhancer_processor",)
    FUNCTION = "run"
    CATEGORY = GLOBAL_CATEGORY

    def run(self,options, model, face_enhancer_blend, face_enhancer_weight):
        for key, value in options.items():
            if value == 'none':
                options[key] = None
            elif(key == 'execution_providers'):
                state_manager.set_item('execution_providers', [options.get('execution_providers')])
            else:
                state_manager.set_item(key,options[key])

        import facefusion.processors.modules.face_enhancer as face_enhancer_processor
        if state_manager.get_item('face_enhancer_model') is not None:
            face_enhancer_processor.clear_inference_pool()
        
        state_manager.set_item('face_enhancer_model', model)
        state_manager.set_item('face_enhancer_blend', face_enhancer_blend)
        state_manager.set_item('face_enhancer_weight', face_enhancer_weight)

        face_enhancer_processor.pre_check()
        return model,

# 唇形同步
class FacefusionLipSyncerProcessor:
    def __init__(self):
        # print("FACE_SWAPPER_SET:",FACE_SWAPPER_SET)
        # print("pixel_boost:",FACE_SWAPPER_SET[default_face_swapper])
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                "options":("FFPROCESSOROPTIONS", ),
                "model": (PROCESSOR_CHOICES_SET.lip_syncer_models, {"default":DEFAULT_SET["lip_syncer"]}),
            },
        }
    
    RETURN_TYPES = ("FFLIPSYNCERPROCESSOR",)
    RETURN_NAMES = ("lip_syncer_processor",)
    FUNCTION = "run"
    CATEGORY = GLOBAL_CATEGORY

    def run(self,options, model,):
        for key, value in options.items():
            if value == 'none':
                options[key] = None
            elif(key == 'execution_providers'):
                state_manager.set_item('execution_providers', [options.get('execution_providers')])
            else:
                state_manager.set_item(key,options[key])

        import facefusion.processors.modules.lip_syncer as lip_syncer_processor
        if state_manager.get_item('lip_syncer_model') is not None:
            lip_syncer_processor.clear_inference_pool()
        
        state_manager.set_item('lip_syncer_model', model)
        return model,

# 配置节点
class FacefusionOptionsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        """定义输入参数"""
        requied_options = {
            "log_level":(CHOICES_SET.log_levels, {"default":CHOICES_SET.log_levels[0]}),
            "hr_1":("STRING",{"default":""}),
            "download_site":(["github","huggingface","hf-mirror"], {"default":"github"}),
            "skip_hash_check":(list(reversed(BOOL_SET)), {}),
            "hr_2":("STRING",{"default":""}),

            "face_selector_mode": (CHOICES_SET.face_selector_modes, {"default":"reference"}), # 选择人脸模式
            "face_selector_orders":(CHOICES_SET.face_selector_orders, {"default":"best-worst"}),# 选择人脸顺序
            "face_selector_genders":(["none"]+CHOICES_SET.face_selector_genders, {"default":"none"}), # 选择人脸性别
            "face_selector_races":(["none"]+CHOICES_SET.face_selector_races, {"default":"none"}), # 选择人脸种族
            "face_selector_age_start":("INT", {"default":0,"min":0, "max":100, "display":"slider"}), # 选择人脸最小年龄
            "face_selector_age_end":("INT", {"default":100,"min":0, "max":100,"display":"slider"}), # 选择人脸最大年龄
            "reference_face_index":("INT",{"default":0,}), # 参考脸序号,
            "reference_face_distance":("FLOAT", {"default":0.6,"min":0.0,"max":1.5,"step":0.05,"display":"slider"}), # 参考脸距离
            "reference_frame_number":("INT",{"default":0,}), # 参考帧序号
            "hr_3":("STRING",{"default":""}),

            "face_occluder_model":(CHOICES_SET.face_occluder_models, {"default":"xseg_1"}), # 遮挡人脸模型
            "face_parser_model":(CHOICES_SET.face_parser_models, {"default":"bisenet_resnet_34"}), # 人脸解析模型
            "hr_4":("STRING",{"default":""}),

            # "face_mask_types":(CHOICES_SET.face_mask_types, {"default":"box"}), # 人脸蒙版类型
            "use_face_box_mask": (BOOL_SET, ), # 是否使用人脸蒙版
            # face_box 属性
            "face_mask_blur":("FLOAT", {"default":0.3,"min":0.0,"max":1.0,"step":0.05,"display":"slider"}), # 人脸蒙版模糊度
            "face_mask_padding_top":("INT", {"default":0,"min":0,"max":100,"step":1,"display":"slider"}), # 人脸蒙版上边距
            "face_mask_padding_right":("INT", {"default":0,"min":0,"max":100,"step":1,"display":"slider"}), # 人脸蒙版右边距
            "face_mask_padding_bottom":("INT", {"default":0,"min":0,"max":100,"step":1,"display":"slider"}), # 人脸蒙版下边距
            "face_mask_padding_left":("INT", {"default":0,"min":0,"max":100,"step":1,"display":"slider"}), # 人脸蒙版左边距
            "hr_5":("STRING",{"default":""}),

            "use_face_occlusion_mask": (list(reversed(BOOL_SET)),), # 是否使用人脸蒙版
            "hr_6":("STRING",{"default":""}),

            "use_face_region_mask": (list(reversed(BOOL_SET)),), # 是否使用人脸蒙版
            # face_region属性
            "face_region_skin":(BOOL_SET, ), # 皮肤区域
            "face_region_left_eyebrow":(BOOL_SET, ), # 左眉区域
            "face_region_right_eyebrow":(BOOL_SET, ), # 右眉区域
            "face_region_left_eye":(BOOL_SET, ), # 左眼区域
            "face_region_right_eye":(BOOL_SET, ), # 右眼区域
            "face_region_glasses":(BOOL_SET, ), # 眼镜区域
            "face_region_nose":(BOOL_SET, ), # 鼻子区域
            "face_region_mouth":(BOOL_SET, ), # 嘴巴区域
            "face_region_upper_lip":(BOOL_SET, ), # 上唇区域
            "face_region_lower_lip":(BOOL_SET, ), # 下唇区域
            "hr_7":("STRING",{"default":""}),
            
            "face_detector_model":(CHOICES_SET.face_detector_models, {"default":DEFAULT_SET["face_detector"]}), # 人脸检测模型
            "face_detector_size":(CHOICES_SET.face_detector_set[DEFAULT_SET["face_detector"]],), # 人脸检测尺寸
            "face_detector_score":("FLOAT",{"default":0.5,"min":0.0,"max":1.0,"step":0.05,"display":"slider"}), # 人脸检测分数范围
        }
        for angle in CHOICES_SET.face_detector_angles:
            requied_options[f"face_detector_angle_{angle}deg"] = (BOOL_SET,) if angle==0 else (list(reversed(BOOL_SET)),)
        
        requied_options = requied_options | {
            "hr_7":("STRING",{"default":""}),

            "face_landmarker_model":(CHOICES_SET.face_landmarker_models,{"default":"2dfan4"}), # 人脸特征点模型
            "face_landmarker_score":("FLOAT",{"default":0.5,"min":0.0,"max":1.0,"step":0.05,"display":"slider"}), # 人脸特征点分数范围
            "hr_8":("STRING",{"default":""}),

            "skip_audio":(list(reversed(BOOL_SET)),), # 是否跳过音频
        }

        return {
            "required": requied_options,
            "optional":{
            },
        }
    
    RETURN_TYPES = ("FFOPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "run"
    CATEGORY = GLOBAL_CATEGORY

    def run(self, **options):
        # 清理用于分隔选项类型的值
        options = {k: v for k, v in options.items() if not k.startswith('hr_')}

        download_site = options.pop('download_site')
        if(download_site == "huggingface"):
            options["download_providers"] = [download_site]
            CHOICES_SET.download_provider_set["huggingface"]["urls"] = [HF_URL]
        elif(download_site == "hf-mirror"):
            options["download_providers"] = ["huggingface"]
            CHOICES_SET.download_provider_set["huggingface"]["urls"] = [HF_M_URL]
        else:
            options["download_providers"] = [download_site]
        # 将显示为face_index的值存为position
        state_manager.set_item('reference_face_position',options.pop('reference_face_index'))

        #设置mask类型
        face_mask_types=[]
        if options.pop('use_face_box_mask'):
            face_mask_types.append('box')
        if options.pop('use_face_occlusion_mask'):
            face_mask_types.append('occlusion')
        if options.pop('use_face_region_mask'):
            face_mask_types.append('region')
        state_manager.set_item('face_mask_types',face_mask_types)

        # 设置 mask padding
        options["face_mask_padding"]=(
            options.pop('face_mask_padding_top'),
            options.pop('face_mask_padding_right'),
            options.pop('face_mask_padding_bottom'),
            options.pop('face_mask_padding_left'),
        )

        # 设置 face_detector_angles
        face_detector_angles = []
        for angle in CHOICES_SET.face_detector_angles:
            if options.pop(f"face_detector_angle_{angle}deg"):
                face_detector_angles.append(angle)
        if len(face_detector_angles) == 0:
            face_detector_angles = [0]
        options["face_detector_angles"] = face_detector_angles

        face_mask_regions=[]
        #遍历options，将值为'none'的键的值设置成None, 并添加到state   
        for key, value in options.items():
            if value == 'none':
                options[key] = None
            
            # 整理face_region属性
            if(key.startswith('face_region_') and value):
                region_name = key.split('face_region_')[-1]
                region_name = region_name.replace('_','-')
                face_mask_regions.append(region_name)
            else:
                state_manager.set_item(key,options[key])

        state_manager.set_item('face_mask_regions',face_mask_regions)

        import facefusion.content_analyser as content_analyser
        content_analyser.pre_check()

        return options,

# processer 配置节点
class FacefusionProcesserOptionsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        """定义输入参数"""
        requied_options = {
            "log_level":(CHOICES_SET.log_levels, {"default":CHOICES_SET.log_levels[0]}),
            "hr_0":("STRING",{"default":""}),

            "download_site":(["github","huggingface","hf-mirror"], {"default":"github"}),
            "skip_hash_check":(list(reversed(BOOL_SET)), {}),
            "hr_1":("STRING",{"default":""}),

            "execution_providers": (CHOICES_SET.execution_providers, {"default":"cuda"}), # 选择执行设备
            "execution_device_id":(list(range(torch.cuda.device_count())), {"default":0}), # 选择执行设备序号
            "execution_thread_count":("INT", {"default":4,"min":CHOICES_SET.execution_thread_count_range[0],"max":CHOICES_SET.execution_thread_count_range[-1], "display":"slider"}), # 选择执行线程数
            "execution_queue_count":("INT", {"default":4,"min":CHOICES_SET.execution_queue_count_range[0],"max":CHOICES_SET.execution_queue_count_range[-1], "display":"slider"}), # 选择执行线程数
        }

        return {
            "required": requied_options,
            "optional":{
            },
        }
    
    RETURN_TYPES = ("FFPROCESSOROPTIONS",)
    RETURN_NAMES = ("processer_options",)
    FUNCTION = "run"
    CATEGORY = GLOBAL_CATEGORY

    def run(self, **options):
        download_site = options.pop('download_site')
        if(download_site == "huggingface"):
            options["download_providers"] = [download_site]
            CHOICES_SET.download_provider_set["huggingface"]["urls"] = [HF_URL]
        elif(download_site == "hf-mirror"):
            options["download_providers"] = ["huggingface"]
            CHOICES_SET.download_provider_set["huggingface"]["urls"] = [HF_M_URL]
        else:
            options["download_providers"] = [download_site]
        

        # 清理用于分隔选项类型的值
        options = {k: v for k, v in options.items() if not k.startswith('hr_')}

        options["execution_device_id"] = str(options.get("execution_device_id")) 

        return options,

from facefusion.core import conditional_process
from facefusion.vision import detect_image_resolution, detect_video_resolution,detect_video_fps
class FacefusionProcessingNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                "source_image_path":("STRING", {"default":"","lazy":True,"data-metadata":"image"}),
                "source_audio_path":("STRING", {"default":"","lazy":True,"data-metadata":"audio"}),
                "target_path":("STRING", {"default":""}),
                "options":("FFOPTIONS", {}),
                "output_path":("STRING", {"default":""}),
                "output_width":("INT", {"default":0,"min":0,"max":99999,"step":1,}),
                "output_height":("INT", {"default":0,"min":0,"max":99999,"step":1,}),
                "output_image_quality":("INT", {"default":100,"min":1,"max":100,"step":1,"display":"slider"}), # 输出图片质量
                "output_audio_encoder":(CHOICES_SET.output_audio_encoders, {"default": CHOICES_SET.output_audio_encoders[0]}), # 输出音频编码器
                "output_video_encoder":(CHOICES_SET.output_video_encoders, {"default": CHOICES_SET.output_video_encoders[0]}), # 输出视频编码器
                "output_video_preset":(CHOICES_SET.output_video_presets, {"default": CHOICES_SET.output_video_presets[0]}), # 输出视频预设
                "output_video_quality":("INT", {"default":100,"min":1,"max":100,"step":1,"display":"slider"}), # 输出视频质量
            },
            "optional":{
                "face_swapper":("FFFACESWAPPERPROCESSOR",),
                "lip_syncer":("FFLIPSYNCERPROCESSOR",),
                "face_enhancer":("FFFACEENHANCERPROCESSOR",),
                "frame_enhancer":("FFFRAMEENHANCERPROCESSOR",),
            },
            "hidden":{
                "test":("STRING", {"default":"1"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "run"
    CATEGORY = GLOBAL_CATEGORY

    OUTPUT_NODE = True

    def run(self,source_image_path,source_audio_path,target_path, options, output_path, output_width, output_height, output_image_quality, output_audio_encoder, output_video_encoder,output_video_preset,output_video_quality,**processors):
        input_root = folder_paths.get_input_directory()

        sources = []
        if source_image_path != "":
            source_image_path = source_image_path.strip('"')
            sources.append(os.path.join(input_root, source_image_path))
        if source_audio_path != "":
            source_audio_path = source_audio_path.strip('"')
            sources.append(os.path.join(input_root, source_audio_path))

        target_path = target_path.strip('"')
        target_path = os.path.join(input_root, target_path)
        output_path = output_path.strip('"')
        output_path = os.path.join(input_root, output_path)

        state_manager.set_item('source_paths', sources)
        state_manager.set_item('target_path', target_path)
        state_manager.set_item('output_path', output_path)

        if is_video(target_path):
            resolution = detect_video_resolution(target_path)
            fps = detect_video_fps(target_path)
            state_manager.set_item('output_video_fps',fps)
            state_manager.set_item('output_video_resolution',f"{resolution[0] if output_width==0 else output_width}x{resolution[1] if output_height==0 else output_height}")
        else:
            resolution = detect_image_resolution(target_path)
            state_manager.set_item('output_image_resolution',f"{resolution[0] if output_width==0 else output_width}x{resolution[1] if output_height==0 else output_height}")

        # 设置 output参数
        state_manager.set_item('output_image_quality',output_image_quality)
        state_manager.set_item('output_audio_encoder',output_audio_encoder)
        state_manager.set_item('output_video_encoder',output_video_encoder)
        state_manager.set_item('output_video_preset',output_video_preset)
        state_manager.set_item('output_video_quality',output_video_quality)

        state_manager.set_item('processors', list(processors.keys()))

        conditional_process()

        return output_path,# image_path, video_path,

    def check_lazy_status(self, source_image_path, source_audio_path,**kwargs):
        if source_image_path != "" or source_audio_path != "":
            return True
        else:
            return False

from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get("/hjh/get_facefusion_combo_set")
async def get_facefusion_combo_set(request):
    combo_set = {
        "face_swapper_models":PROCESSOR_CHOICES_SET.face_swapper_set,
        "face_detector_set":CHOICES_SET.face_detector_set,
    }
    return web.json_response(combo_set)

from .utils.extnames import check_type
@PromptServer.instance.routes.post("/hjh/facefusion/get_file_list")
async def facefusion_get_file_list(request):
    list = []
    input_root = folder_paths.get_input_directory()
    POST = await request.post() 
    cur_dir = POST.get('cur_dir')
    type = POST.get('type')
    if cur_dir == "":
        cur_dir = input_root
    else:
        cur_dir = os.path.join(input_root, cur_dir) 

    if os.path.exists(cur_dir) == False:
        return web.json_response([])

    # 遍历文件夹
    for file in os.listdir(cur_dir):
        file_path = os.path.join(cur_dir, file)
        path = file_path.replace(input_root,"").replace("\\","/")
        if path.startswith("//") is False:
            path = path.lstrip("/")

        if os.path.isdir(file_path):
            list.append({
                "name":file,
                "type":"folder",
                "path": path 
            })
        else:
            if check_type(file, type):
                list.append({
                    "name":file,
                    "type":"file",
                    "path":path,
                })
    # print(list)
    return web.json_response(list)

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "FacefusionOptionsNode":FacefusionOptionsNode,
    "FacefusionProcesserOptionsNode":FacefusionProcesserOptionsNode,
    "FacefusionProcessingNode":FacefusionProcessingNode,
    "FacefusionFaceSwapperProcessor":FacefusionFaceSwapperProcessor,
    "FacefusionFaceEnhancerProcessor":FacefusionFaceEnhancerProcessor,
    "FacefusionLipSyncerProcessor":FacefusionLipSyncerProcessor,
    "FacefusionFrameEnhancerProcessor":FacefusionFrameEnhancerProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FacefusionOptionsNode":"HJH-Facefusion-Node Options",
    "FacefusionProcesserOptionsNode":"HJH-Facefusion-Node Processer Options",
    "FacefusionProcessingNode":"HJH-Facefusion-Node Processing",
    "FacefusionFaceSwapperProcessor":"HJH-Facefusion-Node Face Swapper",
    "FacefusionFaceEnhancerProcessor":"HJH-Facefusion-Node Face Enhancer",
    "FacefusionLipSyncerProcessor":"HJH-Facefusion-Node Lip Syncer",
    "FacefusionFrameEnhancerProcessor":"HJH-Facefusion-Node Frame Enhancer",
}
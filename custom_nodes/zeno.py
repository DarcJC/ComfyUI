from PIL import Image, ImageOps, ImageSequence
from io import BytesIO
import numpy as np
import struct
import comfy.utils
import time
import sys
import argparse
import json
import os
import folder_paths
import safetensors.torch
import torch

#You can use this node to save full size images through the websocket, the
#images will be sent in exactly the same format as the image previews: as
#binary images on the websocket with a 8 byte header indicating the type
#of binary message (first 4 bytes) and the image format (next 4 bytes).

#Note that no metadata will be put in the images saved with this node.

CURRENT_FRAME = 0
FRAME_SHIFT = 0
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--placeholder", type=str, help="a json file path that provide placeholder value", required=False)
    parser.add_argument("--frameshift", type=int, help="frame shift for output", required=False, default=0)
    # parser.add_argument("--placeholder", type=str, help="a json string that provide placeholder value", default="{}")
    args, unknown = parser.parse_known_args()
    PLACEHOLDER_VALUES = None
    if args.placeholder:
        with open(args.placeholder, "r", encoding="UTF-8") as f:
            PLACEHOLDER_VALUES = json.loads(f.read())
    else:
        PLACEHOLDER_VALUES = {}
    print("Using placeholder list:", PLACEHOLDER_VALUES)
    CURRENT_FRAME = int(args.frameshift)
    FRAME_SHIFT = int(args.frameshift)
    print("Using frame shift: ", CURRENT_FRAME)
except Exception as err:
    print("Error!", err)
    os._exit(1)


def get_current_frame_and_add():
    t = CURRENT_FRAME
    t = CURRENT_FRAME + 1
    return t


class ZenoStringPlaceholder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "key": ("STRING", {
                            "default": "replace_me"
                        }),
                        "default": ("STRING", {
                            "default": ""
                        }),
                    }
                }

    RETURN_TYPES = ("STRING",)
    
    FUNCTION = "load_placeholder"
    CATEGORY = "Zeno/utils"
    OUTPUT_NODE = False

    def load_placeholder(self, key, default):
        value = PLACEHOLDER_VALUES.get(key, default)
        print(f"{key} is replacing to {value}")
        return (value,)


class ZenoStringToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "input": ("STRING", {"default": "0"}),
                        "base": ("INT", { "default": 10, "min": 2, "max": 16, "step": 1, "display": "slider" }),
                    }
                }

    RETURN_TYPES = ("INT",)

    FUNCTION = "convert"
    CATEGORY = "Zeno/utils"
    OUTPUT_NODE = False

    def convert(self, input: str, base: int):
        if isinstance(input, int):
            return (input,)
        return (int(input, base=base), )


class ZenoStringToFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "input": ("STRING", {"default": "0.0"}),
                    }
                }

    RETURN_TYPES = ("FLOAT",)

    FUNCTION = "convert"
    CATEGORY = "Zeno/utils"
    OUTPUT_NODE = False

    def convert(self, input: str):
        return (float(input),)


class ZenoAnyPlaceholder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "key": ("STRING", {
                            "default": "replace_me"
                        }),
                        "default": ("STRING", {
                            "default": ""
                        }),
                        "placeholder_type": (["string", "int", "float"], ),
                    }
                }

    RETURN_TYPES = ("*",)
    
    FUNCTION = "load_placeholder"
    CATEGORY = "Zeno/utils"

    def load_placeholder(self, key, default, pt):
        value = PLACEHOLDER_VALUES.get(key, default)
        try:
            if pt == 'string':
                value = str(value)
            if pt == 'int':
                value = int(value)
            if pt == 'float':
                value = float(value)
        except:
            if pt == 'string':
                value = ""
            elif pt == "int":
                value = 0
            elif pt == "float":
                value = 0.0
        print(f"{key} is replacing to {value}")
        return (value,)


class ZenoLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "key": ("STRING", {
                            "default": "replace_me"
                        }),
                        "default_image_path": ("STRING", {"default": "example.png"})
                    },
                }

    CATEGORY = "Zeno/image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, key, default_image_path):
        image = PLACEHOLDER_VALUES.get(key, folder_paths.get_annotated_filepath(default_image_path))
        print(f"{key} is replacing to {image}")

        image_path = image
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, key, default_image_path):
        image = PLACEHOLDER_VALUES.get(key, folder_paths.get_annotated_filepath(default_image_path))
        print(f"{key} is replacing to {image}")
        image_path = default_image_path
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, key, default_image_path):
        image = PLACEHOLDER_VALUES.get(key, folder_paths.get_annotated_filepath(default_image_path))
        print(f"{key} is replacing to {image}")
        if not os.path.exists(image):
            return "Invalid image file: {}".format(image)

        return True


class ZenoSaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "padding": ("INT", {"default": 5}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Zeno/image"

    def save_images(self, images, filename_prefix="ComfyUI", padding=5, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        counter = FRAME_SHIFT
        print(f"Saving images to {full_output_folder} with prefix {filename_prefix}")
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            # metadata = PngInfo()
            # if prompt is not None:
            #     metadata.add_text("prompt", json.dumps(prompt))
            # if extra_pnginfo is not None:
            #     for x in extra_pnginfo:
            #         metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", "")
            file = f"{filename_with_batch_num}_{counter:>0{padding}}.png"
            # filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            # file = f"{filename_with_batch_num}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }

class ZenoSaveSingleImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_path": ("STRING", {"default": "P:/AI_Demo/AIOutput/xxx.jpg"}),
                     },
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Zeno/image"

    def save_images(self, images, filename_path="P:/AI_Demo/AIOutput/xxx.jpg"):
        # filename_prefix += self.prefix_append
        # full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        # counter = get_current_frame_and_add()
        print(f"Saving images to {filename_path}")
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            # metadata = PngInfo()
            # if prompt is not None:
            #     metadata.add_text("prompt", json.dumps(prompt))
            # if extra_pnginfo is not None:
            #     for x in extra_pnginfo:
            #         metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # filename_with_batch_num = filename.replace("%batch_num%", "")
            # file = f"{filename_with_batch_num}_{counter:>0{padding}}.png"
            # filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            # file = f"{filename_with_batch_num}.png"
            img.save(filename_path, pnginfo=metadata, compress_level=self.compress_level)
            # results.append({
            #     "filename": file,
            #     "subfolder": subfolder,
            #     "type": self.type
            # })
            # counter += 1

        return { "ui": { "images": results } }


NODE_CLASS_MAPPINGS = {
    "ZenoPlaceholder": ZenoStringPlaceholder,
    "ZenoStringToInt": ZenoStringToInt,
    "ZenoStringToFloat": ZenoStringToFloat,
    "ZenoAnyPlaceholder": ZenoAnyPlaceholder,
    "ZenoLoadImage": ZenoLoadImage,
    "ZenoSaveImage": ZenoSaveImage,
    "ZenoSaveSingleImage": ZenoSaveSingleImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZenoPlaceholder": "String Placeholder Node",
    "ZenoStringToInt": "String To Int",
    "ZenoStringToFloat": "String To Float",
    "ZenoAnyPlaceholder": "Any Placeholder Node",
    "ZenoLoadImage": "Image Placeholder Node",
    "ZenoSaveImage": "Image Custom Save Node",
    "ZenoSaveSingleImage": "Save Single Image Node",
}

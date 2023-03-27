import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


import rembg
import numpy as np
import PIL


_MASK_MANUAL = 'Manually (Draw Mask)'
_MASK_REMOVE_BG = 'Use ML Model'
_MASK_PNG_ALPHA = 'Use PNG Alpha Channel'

_mask_methods = [_MASK_MANUAL, _MASK_REMOVE_BG, _MASK_PNG_ALPHA]



def get_mask_rembg(image):
    output = rembg.remove(image)
    img = np.array(output)
    mask_arr = 255-img[:, :, -1]
    mask = PIL.Image.fromarray(mask_arr, 'L')
    return mask


def get_mask_png_alpha(image):
    image_arr = np.array(image)
    alpha_channel = image_arr[:, :, -1]
    mask = PIL.Image.fromarray(alpha_channel)
    return mask


class Script(scripts.Script):
    def title(self):
        return 'Creative AI'

    def show(self, is_img2img):
        del(is_img2img)
        #return is_img2img
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        del(is_img2img)
        with gr.Accordion('Creative AI', open=True, elem_id='creative-ai'):
            mask_method = gr.Radio(label='Mask Method', choices=_mask_methods, value=_MASK_MANUAL)
            output_mask = gr.Checkbox(label='Output Mask for Debugging', value=False)

        return [mask_method, output_mask]

    def process(self, p, mask_method, output_mask, *args):
        del(args)
        if not hasattr(p, 'image_mask'):
            return
        if mask_method == _MASK_REMOVE_BG:
            p.image_mask = get_mask_rembg(p.init_images[0])
        elif mask_method == _MASK_PNG_ALPHA:
            pass
            #p.image_mask = get_mask_png_alpha(p.init_images[0])

        #print('mask image:', p.image_mask)
        #print('images:', p.init_images)

    def postprocess(self, p, processed, *args):
        del(args)
        if not hasattr(p, 'image_mask'):
            return
        processed.images.append(p.image_mask)



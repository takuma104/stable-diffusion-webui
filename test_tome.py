import json
import requests
import io
import base64
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from pprint import pprint

plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams['figure.facecolor'] = 'white'

url = "http://127.0.0.1:7860"

# prompt by @p1atdev_art
# https://twitter.com/p1atdev_art/status/1616557167087353856
prompt = """masterpiece, best quality, high quality, 1girl, sun hat, 
frilled white dress, looking at viewer, summer, sky, beach, semi-realistic"""
neg_prompt = """nsfw, worst quality, low quality, medium quality, deleted, 
lowres, bad anatomy, bad hands, text, error, missing fingers, extra digits, 
fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry,"""
t2i_payload = {
    "prompt": prompt,
    "negative_prompt": neg_prompt,
    "seed": 0,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 11,
    "width": 512,
    "height": 512,
    "sampler_index": "DPM++ 2S a Karras",
    "model": "wd-1-5-beta2-aesthetic-fp16",
}

def t2i_repeate(fn_prefix='run', seed=0, runs=4, size=512):
    t2i_payload['seed'] = seed
    t2i_payload['width'] = size
    t2i_payload['height'] = size
    infotext = ''
    for run in range(runs):
        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=t2i_payload)
        r = response.json()
        infotext = json.loads(r['info'])['infotexts'][0]
        i = r['images'][0]
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        fn = f"{fn_prefix}_{run}.png"
        image.save(fn)
        print(f'{fn} generated.')
    return infotext

if __name__ == '__main__':
    size = int(sys.argv[1])
    fn_prefix='run'
    t2i_repeate(fn_prefix=fn_prefix, seed=7, runs=4, size=size)

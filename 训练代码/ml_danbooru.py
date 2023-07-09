import json
import logging
import os
import shutil
from functools import lru_cache
from threading import Lock
from typing import Optional

import numpy as np
from PIL import Image
from hbutils.system import pip_install
from huggingface_hub import hf_hub_download


def _ensure_onnxruntime():
    try:
        import onnxruntime
    except (ImportError, ModuleNotFoundError):
        logging.warning('Onnx runtime not installed, preparing to install ...')
        if shutil.which('nvidia-smi'):
            logging.info('Installing onnxruntime-gpu ...')
            pip_install(['onnxruntime-gpu'], silent=True)
        else:
            logging.info('Installing onnxruntime (cpu) ...')
            pip_install(['onnxruntime'], silent=True)


_ensure_onnxruntime()
from onnxruntime import get_available_providers, get_all_providers, InferenceSession, SessionOptions, \
    GraphOptimizationLevel

alias = {
    'gpu': "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
}


def get_onnx_provider(provider: Optional[str] = None):
    if not provider:
        if "CUDAExecutionProvider" in get_available_providers():
            return "CUDAExecutionProvider"
        else:
            return "CPUExecutionProvider"
    elif provider.lower() in alias:
        return alias[provider.lower()]
    else:
        for p in get_all_providers():
            if provider.lower() == p.lower() or f'{provider}ExecutionProvider'.lower() == p.lower():
                return p

        raise ValueError(f'One of the {get_all_providers()!r} expected, '
                         f'but unsupported provider {provider!r} found.')


def resize(pic: Image.Image, size: int, keep_ratio: float = True) -> Image.Image:
    if not keep_ratio:
        target_size = (size, size)
    else:
        min_edge = min(pic.size)
        target_size = (
            int(pic.size[0] / min_edge * size),
            int(pic.size[1] / min_edge * size),
        )

    target_size = (
        (target_size[0] // 4) * 4,
        (target_size[1] // 4) * 4,
    )

    return pic.resize(target_size, resample=Image.Resampling.BILINEAR)


def to_tensor(pic: Image.Image):
    img: np.ndarray = np.array(pic, np.uint8, copy=True)
    img = img.reshape(pic.size[1], pic.size[0], len(pic.getbands()))

    # put it from HWC to CHW format
    img = img.transpose((2, 0, 1))
    return img.astype(np.float32) / 255


def fill_background(pic: Image.Image, background: str = 'white') -> Image.Image:
    if pic.mode == 'RGB':
        return pic
    if pic.mode != 'RGBA':
        pic = pic.convert('RGBA')

    background = background or 'white'
    result = Image.new('RGBA', pic.size, background)
    result.paste(pic, (0, 0), pic)

    return result.convert('RGB')


def image_to_tensor(pic: Image.Image, size: int = 512, keep_ratio: float = True, background: str = 'white'):
    return to_tensor(resize(fill_background(pic, background), size, keep_ratio))


DEFAULT_MODEL = 'ml_caformer_m36_dec-5-97527.onnx'


def get_onnx_model_file(name=DEFAULT_MODEL):
    return hf_hub_download(
        repo_id='deepghs/ml-danbooru-onnx',
        filename=name,
    )


def _open_onnx_model(ckpt: str, provider: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "CPUExecutionProvider":
        options.intra_op_num_threads = os.cpu_count()

    logging.info(f'Model {ckpt!r} loaded with provider {provider!r}')
    return InferenceSession(ckpt, options, [provider])


CLASSES = json.load(open(hf_hub_download(repo_id='deepghs/ml-danbooru-onnx', filename='classes.json'), 'r', encoding='utf-8'))


model = None
_L = Lock()


def get_tags_from_image(pic: Image.Image, threshold: float = 0.7, size: int = 512, keep_ratio: bool = False):
    global model
    real_input = image_to_tensor(pic, size, keep_ratio)
    real_input = real_input.reshape(1, *real_input.shape)
    with _L:
        if model is None:
            model = _open_onnx_model(get_onnx_model_file(DEFAULT_MODEL), get_onnx_provider('cpu'))
        native_output, = model.run(['output'], {'input': real_input})
    output = (1 / (1 + np.exp(-native_output))).reshape(-1)
    pairs = sorted([(CLASSES[i], ratio) for i, ratio in enumerate(output)], key=lambda x: (-x[1], x[0]))
    return {tag: float(ratio) for tag, ratio in pairs if ratio >= threshold}

import json
import lzma

import numpy as np
from PIL import Image

import ml_danbooru


人标签 = ['animal_ears', 'animal_ear_fluff', 'pointy_ears', 'cat_ears', 'fox_ears', 'dog_ears', 'horse_ears', 'multiple_tails', 'rabbit_tail', 'cat_tail', 'fox_tail', 'dog_tail', 'demon_tail', 'blue_eyes', 'hair_between_eyes', 'purple_eyes', 'green_eyes', 'brown_eyes', 'red_eyes', 'closed_eyes', 'pink_eyes', 'yellow_eyes', 'aqua_eyes', 'grey_eyes', 'black_eyes', 'orange_eyes', 'white_eyes', 'glowing_eyes', 'hair_over_eyes', 'hair_over_shoulder', 'long_hair', 'short_hair', 'brown_hair', 'eyebrows_visible_through_hair', 'black_hair', 'blonde_hair', 'very_long_hair', 'blue_hair', 'purple_hair', 'silver_hair', 'white_hair', 'pink_hair', 'grey_hair', 'pubic_hair', 'medium_hair', 'multicolored_hair', 'red_hair', 'two-tone_hair', 'shiny_hair', 'streaked_hair', 'orange_hair', 'gradient_hair', 'aqua_hair', 'green_hair', 'light_purple_hair', 'light_brown_hair', 'tied_hair', 'straight_hair', 'asymmetrical_hair', 'spiked_hair', 'light_blue_hair', 'platinum_blonde_hair', 'eyebrows_behind_hair', 'colored_inner_hair', 'drill_hair', 'wavy_hair', 'low-tied_long_hair', 'antenna_hair', 'medium_breasts', 'large_breasts', 'small_breasts', 'huge_breasts', 'gigantic_breasts', 'flat_chest', 'ponytail', 'high_ponytail', 'ribbon', 'two_side_up', 'twintails', 'short_twintails', 'low_twintails', 'one_side_up', 'double_bun', 'hair_bun', 'bangs', 'blunt_bangs', 'parted_bangs', 'swept_bangs', 'asymmetrical_bangs', 'hair_ornament', 'sidelocks', 'short_hair_with_long_locks', 'virtual_youtuber', 'braid', 'tail', 'hairband', 'hairclip', 'hair_bow', 'hair_ribbon', 'side_ponytail', 'glasses', 'heterochromia', 'elf', 'ahoge', 'halo', 'hair_over_one_eye', 'horns', 'hime_cut', 'hair_intakes', 'headgear', 'short_eyebrows', 'thick_eyebrows', 'mole', 'mole_under_eye', 'bow', 'dark_skin', 'colored_skin', 'wings', 'jewelry', 'necktie', 'coat', 'elbow_gloves', 'hat', 'weapon', 'white_shirt', 'armor', 'black_neckwear', 'yellow_bow', 'emblem', 'hood']


with lzma.open('人均值.json.xz') as f:
    _人均值 = json.load(f)
_人均值 = {k: np.array(v) for k, v in _人均值.items()}


_人阵, _人均值阵 = [], []
for k, v in _人均值.items():
    _人阵.append(k)
    _人均值阵.append(v)
_人均值阵 = np.array(_人均值阵)


def _标签转特征(t: dict) -> np.array:
    return np.array([t.get(s, 0) for s in 人标签])


def predict(image, top_n=3) -> list:
    tags = ml_danbooru.get_tags_from_image(image, threshold=0.4, keep_ratio=True, size=256)
    特征 = _标签转特征(tags)
    距离 = np.linalg.norm(特征 - _人均值阵, axis=1)
    预测人 = [(_人阵[i], 距离[i]) for i in np.argsort(距离)[:top_n]]
    return 预测人

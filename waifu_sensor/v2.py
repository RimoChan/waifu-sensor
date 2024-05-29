import json
import lzma
from pathlib import Path

import numpy as np
from PIL import Image

from . import ml_danbooru


人标签 = ['animal_ears', 'animal_ear_fluff', 'pointy_ears', 'cat_ears', 'fox_ears', 'dog_ears', 'horse_ears', 'multiple_tails', 'rabbit_tail', 'cat_tail', 'fox_tail', 'dog_tail', 'demon_tail', 'blue_eyes', 'hair_between_eyes', 'purple_eyes', 'green_eyes', 'brown_eyes', 'red_eyes', 'closed_eyes', 'pink_eyes', 'yellow_eyes', 'aqua_eyes', 'grey_eyes', 'black_eyes', 'orange_eyes', 'white_eyes', 'glowing_eyes', 'hair_over_eyes', 'hair_over_shoulder', 'long_hair', 'short_hair', 'brown_hair', 'eyebrows_visible_through_hair', 'black_hair', 'blonde_hair', 'very_long_hair', 'blue_hair', 'purple_hair', 'silver_hair', 'white_hair', 'pink_hair', 'grey_hair', 'pubic_hair', 'medium_hair', 'multicolored_hair', 'red_hair', 'two-tone_hair', 'shiny_hair', 'streaked_hair', 'orange_hair', 'gradient_hair', 'aqua_hair', 'green_hair', 'light_purple_hair', 'light_brown_hair', 'tied_hair', 'straight_hair', 'asymmetrical_hair', 'spiked_hair', 'light_blue_hair', 'platinum_blonde_hair', 'eyebrows_behind_hair', 'colored_inner_hair', 'drill_hair', 'wavy_hair', 'low-tied_long_hair', 'antenna_hair', 'medium_breasts', 'large_breasts', 'small_breasts', 'huge_breasts', 'gigantic_breasts', 'flat_chest', 'ponytail', 'high_ponytail', 'ribbon', 'two_side_up', 'twintails', 'short_twintails', 'low_twintails', 'one_side_up', 'double_bun', 'hair_bun', 'bangs', 'blunt_bangs', 'parted_bangs', 'swept_bangs', 'asymmetrical_bangs', 'hair_ornament', 'sidelocks', 'short_hair_with_long_locks', 'virtual_youtuber', 'braid', 'tail', 'hairband', 'hairclip', 'hair_bow', 'hair_ribbon', 'side_ponytail', 'glasses', 'heterochromia', 'elf', 'ahoge', 'halo', 'hair_over_one_eye', 'horns', 'hime_cut', 'hair_intakes', 'headgear', 'short_eyebrows', 'thick_eyebrows', 'mole', 'mole_under_eye', 'bow', 'dark_skin', 'colored_skin', 'wings', 'jewelry', 'necktie', 'coat', 'elbow_gloves', 'hat', 'weapon', 'white_shirt', 'armor', 'black_neckwear', 'yellow_bow', 'emblem', 'hood', 'looking_at_viewer', 'blush', 'breasts', 'simple_background', 'smile', 'white_background', 'open_mouth', 'shirt', 'shiny', 'standing', 'long_sleeves', 'artist_name', 'thighs', 'closed_mouth', 'skirt', 'alternate_costume', 'collarbone', 'cowboy_shot', 'bare_shoulders', 'dress', 'looking_to_the_side', 'eyelashes', 'hand_up', 'head_tilt', 'holding', 'twitter_username', 'cleavage', 'eyebrows', 'parted_lips', 'dated', 'watermark', 'upper_body', 'signature', 'underwear', 'shiny_skin', 'fingernails', 'gloves', ':d', 'navel', 'sleeveless', 'frills', 'eyes_visible_through_hair', 'dutch_angle', 'thighhighs', 'web_address', 'grey_background', 'full_body', 'teeth', 'heart', 'jacket', 'sketch', 'from_side', 'arm_up', 'miniskirt', 'sitting', 'black_legwear', 'lips', 'short_sleeves', 'groin', 'collar', 'collared_shirt', 'blurry', 'open_clothes', 'panties', 'pleated_skirt', 'alternate_hairstyle', 'legs', 'bare_arms', 'midriff', 'leaning_forward', 'puffy_sleeves', 'light_smile', 'hands_up', 'stomach', 'looking_away', 'v-shaped_eyebrows', 'shadow', 'skindentation', 'character_name', 'multicolored', ':o', 'cosplay', 'gradient', 'depth_of_field', 'detached_sleeves', 'floating_hair', 'ass', 'school_uniform', 'half-closed_eyes', 'wide_sleeves', 'shoes', 'bare_legs', 'tareme', 'fang', 'choker', 'feet_out_of_frame', 'looking_down', 'flower', 'buttons', 'black_skirt', 'belt', 'gradient_background', 'sweat', 'sleeves_past_wrists', 'outdoors', 'boots', 'swimsuit', 'covered_nipples', 'wing_collar', 'striped', 'red_ribbon', 'nail_polish', 'one-hour_drawing_challenge', 'blouse', 'expressionless', 'light_blush', 'day', 'frown', 'armpits', 'see-through', 'from_above', 'earrings', 'tsurime', 'no_bra', 'off_shoulder', 'embarrassed', 'bowtie', 'dress_shirt', 'tongue', 'halterneck', 'star_(symbol)', 'blurry_background', 'white_legwear', 'nose_blush', 'sideboob', 'leg_up', 'symbol-shaped_pupils', 'arm_support', 'black_gloves', 'light_particles', 'uniform', 'sky', 'sleeveless_dress', 'open_jacket', 'indoors', 'white_dress', 'black_ribbon', 'neck_ribbon', 'english_text', 'short_dress', 'black_footwear', 'foreshortening', 'sweatdrop', 'looking_up', 'sleeveless_shirt', 'looking_back', 'pantyhose', 'from_below', 'vest', 'contrapposto', 'two-tone_background', 'clothing_cutout', 'sparkle', 'red_bow', 'sunlight', 'black_dress', 'areolae', 'outstretched_arm', 'bikini', 'slit_pupils', 'clothes_lift', 'happy', 'arm_at_side', 'bra', 'patreon_username', 'upper_teeth', 'makeup', 'ass_visible_through_thighs', 'arms_up', 'wind', 'legs_together', 'red_neckwear']


w = np.array([1.0, 1.0, 1.0, 0.6196702088134012, 0.8136686501354825, 0.7139569621037521, 1.0, 0.9105899697368581, 0.30646884142682057, 0.6106608786111534, 0.6860634605733226, 1.0, 0.8103552847941861, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.31737839664916456, 0.8841340491818594, 1.0, 0.7753598177644933, 1.0, 1.0, 1.0, 0.7413820250147418, 0.7092589208106523, 0.9230816768707234, 0.6585577944961893, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9058255528251736, 1.0, 1.0, 1.0, 0.4635535679953233, 1.0, 1.0, 1.0, 0.34483096324604084, 0.2916921703569786, 0.0, 1.0, 1.0, 0.7128496350822201, 1.0, 1.0, 1.0, 0.702407301523928, 1.0, 1.0, 0.7943012113417053, 1.0, 0.5816925503668819, 0.7999461898881363, 1.0, 1.0, 1.0, 0.9658539834448508, 0.9104754954678922, 1.0, 1.0, 0.31610163349413306, 0.4043143693601816, 0.8001292237326092, 0.8239613825746457, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.9397807751193408, 0.9269724046458486, 1.0, 1.0, 0.9111096299864008, 1.0, 1.0, 1.0, 0.9597040727388867, 1.0, 1.0, 0.0, 1.0, 1.0, 0.7671671792688803, 0.1502205392941192, 1.0, 1.0, 1.0, 0.7547359919177995, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.1744619263394497, 1.0, 0.38957132701157343, 0.1827066917252549, 1.0, 1.0, 0.0, 0.9879096387209226, 0.031205079541986563, 0.06947243616658358, 0.491044241546869, 1.0, 0.0, 0.6029935219851833, 0.0, 0.051717915878040505, 0.9219745306809581, 0.0, 0.0, 0.0, 0.550414154996014, 0.0, 0.0, 0.10152895673205846, 0.0, 0.0, 0.3242058125156621, 0.6247064473087517, 0.0, 0.2740212861441173, 0.0, 0.0, 0.4706842503032639, 0.6878584550426591, 0.0, 0.16764321660854187, 0.0, 0.04964850159481517, 0.0, 1.0, 0.2736539079322769, 0.7011270008430941, 0.07256775648444608, 0.0, 1.0, 0.0, 0.22498130470245575, 0.3398922042454251, 0.0, 0.3159457724462025, 0.0, 1.0, 0.25107012549881863, 0.8923066814265749, 0.0, 0.0, 0.30632979220142365, 0.0, 0.0, 0.7477732121232057, 0.18735313089151326, 0.0, 0.0, 0.0, 0.0946112657852323, 0.0, 0.09247864286481501, 0.5762492757978814, 0.011859447458120148, 0.2940252896294318, 0.0, 0.5701255050393726, 0.2621359770425037, 0.26648942799274733, 0.0, 0.3426102961424208, 0.0, 0.3218340523032685, 0.026907647728170798, 0.4819719772832961, 0.0, 0.6155501255866447, 0.4854250327643437, 0.21967773990091544, 0.3988230843357877, 0.0, 0.9347563847792025, 0.24152554722290728, 0.9401076349665487, 0.5559111373751726, 0.12478627073647236, 0.2415150306090192, 0.0, 0.4299239700433099, 0.6205684792348696, 0.06486790960334057, 0.7822859179069445, 0.7616197858464084, 0.0, 0.0, 0.14297900110501896, 1.0, 0.0, 0.8778100137827116, 1.0, 1.0, 0.8397289887446975, 0.5692549652538004, 0.17470530673141568, 0.6966194044493337, 0.20052150536124722, 0.0, 1.0, 0.7099815226016615, 0.32688453506052084, 0.0, 0.13420223649450316, 0.0, 0.0, 0.519638581821267, 0.3858239162052936, 1.0, 1.0, 0.061388885562528826, 0.5100978187501516, 0.19695743101273117, 0.4478749032973049, 0.06777944957116754, 0.28229172786387546, 0.6683301507042286, 0.07470283977621422, 0.2755022196112187, 0.0, 1.0, 0.0, 0.5317774071377955, 0.6882267864825502, 0.2200050559640987, 0.3994758715588478, 0.15769302419402892, 0.396446757681362, 0.0, 0.12325178697833587, 0.0619598302184682, 0.1889014112901315, 0.0, 0.0, 0.733872050744509, 0.6656674374417245, 0.6210987384018096, 0.20588763211404382, 0.013271390968714422, 1.0, 0.0, 0.24518247901522117, 0.3016117020130831, 0.37113295857118445, 0.6651285447537479, 0.1356897736509934, 0.3278209101678399, 0.3215727482713173, 0.6001876964313091, 0.1753805771974099, 0.14293516253694727, 0.1381482919159749, 0.20404075393523044, 0.09235068296588303, 0.0, 0.0, 0.9788722225287347, 0.0, 0.49460332137406776, 0.5718950601717618, 0.5573135022370715, 0.8429915341169456, 0.21857127224881873, 0.10182879743720456, 0.00924752053633909, 0.3933412379217748, 0.15903185805932032, 0.0, 0.9695725179413723, 1.0, 0.5256315602688051, 0.38946698648646233, 0.2817986167568704, 0.31981526693020507, 0.2929498601844566, 1.0, 0.17154156327645056, 0.004967242941380549, 0.6914274012095183, 0.0, 0.3357773148257881])


with lzma.open(Path(__file__).parent/'人均值v2.json.xz') as f:
    _人均值 = json.load(f)
_人均值 = {k: np.array(v) for k, v in _人均值.items()}


assert len(人标签) == len(set(人标签)) == len(w) == len(next(iter(_人均值.values())))


_人阵, _人均值阵 = [], []
for k, v in _人均值.items():
    _人阵.append(k)
    _人均值阵.append(v)
_人均值阵 = np.array(_人均值阵)


def _标签转特征(t: dict) -> np.array:
    return np.array([t.get(s, 0) for s in 人标签])


def predict(image, top_n=3, size=512) -> list:
    tags = ml_danbooru.get_tags_from_image(image, threshold=0.4, keep_ratio=True, size=size)
    特征 = _标签转特征(tags)
    距离 = np.linalg.norm(w * (特征 - _人均值阵), axis=1)
    预测人 = [(_人阵[i], 距离[i]) for i in np.argsort(距离)[:top_n]]
    return 预测人


def why_not(image, name, top_n=3, size=512):
    目标特征 = [均 for 人, 均 in zip(_人阵, _人均值阵) if 人==name][0]
    tags = ml_danbooru.get_tags_from_image(image, threshold=0.4, keep_ratio=True, size=size)
    特征 = _标签转特征(tags)
    差 = w * (特征 - 目标特征)
    return [(人标签[i], 差[i]) for i in np.argsort(差**2)[::-1][:top_n]]

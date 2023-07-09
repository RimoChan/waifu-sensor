import json
import random
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
import lxml.html
from tqdm import tqdm


human = []

def f(_):
    i = random.randint(1, 6400000)
    try:
        r = requests.get(f'https://danbooru.donmai.us/posts/{i}', headers={'user-agent': 'rimochan'})
        r.raise_for_status()
        h = lxml.html.document_fromstring(r.text)
        for j in h.cssselect('ul.character-tag-list .search-tag'):
            human.append(j.text)
    except Exception as e:
        logging.exception(e)


for k, _ in tqdm(enumerate(ThreadPoolExecutor(max_workers=4).map(f, range(30000))), total=30000):
    if k % 100 == 0:
        with open('human.json', 'w', encoding='utf8') as f:
            f.write(json.dumps(sorted(human)))

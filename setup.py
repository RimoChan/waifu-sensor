import setuptools


setuptools.setup(
    name='waifu-sensor',
    version='3.0.1',
    author='RimoChan',
    author_email='the@librian.net',
    description='waifu-sensor',
    long_description=open('readme.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RimoChan/waifu-sensor',
    packages=['waifu_sensor'],
    package_data={'waifu_sensor': ['人均值.json.xz', '人均值v2.json.xz', '人均值v3.json.xz']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'huggingface-hub>=0.15,<2.0',
        'numpy>=1.24,<2.0',
        'hbutils>=0.9,<1.0',
        'Pillow>=10.0,<12.0',
    ],
    python_requires='>=3.9',
)

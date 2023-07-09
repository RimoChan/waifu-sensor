# 【老婆探测器】一个普通的动漫角色分类模型

事情是这样的，网上有一些别人训练好的动漫角色分类模型。但是这些模型往往都有一个问题，那就是它们不认得新角色，所以还得隔一段时间训练一次，然后把这段时间内的角色标签重打一遍。

于是我就想，要不就不分类了，用embedding召回的方法来做，这样一来，出了新角色以后，只需要把这个角色的embedding入库，不就可以识别出这个角色的其他图了嘛！

不过我也没有embedding模型，所以这次直接用[ML-Danbooru](https://github.com/7eu7d7/ML-Danbooru)凑合一下(这是一个标签模型)。把标签用手凑一凑，拼成一个embedding吧！


## 使用方法

把这个仓库clone回去，然后把1张图片输入`predict`里就可以了:

```python
from PIL import Image
from predict import predict

print(predict(Image.open('urisai.jpg')))   # [('momoi (blue archive)', 1.4793390460772633), ('midori (blue archive)', 2.2018390494738482), ('iijima yun', 2.309663538692209)]
```

|  图片  | 预测结果 1  | 预测结果 2  | 预测结果 3  |
|  ----  | ----  | ----  | ----  |
| ![urisai.jpg](urisai.jpg)  | momoi (blue archive), 1.4793390460772633) | midori (blue archive), 2.2018390494738482 | iijima yun, 2.309663538692209)] |


## 关于训练

这次用的数据集是[danbooru2022](https://huggingface.co/datasets/animelover/danbooru2022)。

下了4%的数据出来训练，因为数据太多了，下不动啦。然后过滤出只包含一个女性角色的图片，总的训练样本数大概是60000。

- 训练集下了36个包，是 `data-0\d[0257]1.zip`。 
- 测试集是 `data-0000.zip`。

测完发现这个瞎搞的准确率其实没有很高，top1命中74%，top3命中80%。

嘛，毕竟有5684个分类，长尾的分类太多了。我自己都认不出74%的图，那它已经比我认得多啦！

不过因为只给所有的图打一次标签，相当于只需要炼1个epoch，训练很快。


## 结束

就这样，我要去和电脑里的老婆亲热了，大家88！

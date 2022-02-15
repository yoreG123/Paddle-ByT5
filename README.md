# Paddle-ByT5
Paddle-ByT5
## 1 简介 

**本项目基于PaddlePaddle复现的Byt5，完成情况如下:**

- 在tweetqa和xsum数据集上均达到论文精度
- 我们复现的ByT5是基于paddlenlp
- 我们提供aistudio notebook, 帮助您快速验证模型

**项目参考：**
- [https://github.com/huggingface/transformers/tree/master/src/transformers/models/byt5](https://github.com/huggingface/transformers/tree/master/src/transformers/models/byt5)
- [https://github.com/google-research/byt5](https://github.com/google-research/byt5)
- [https://github.com/JunnYu/paddle_t5](https://github.com/JunnYu/paddle_t5)

## 2 复现精度
>#### 在TweetQA数据集的测试效果如下表。

|模型 |opt|数据集|BLEU-1|BLEU-1(原论文)|rougeL|rougeL(原论文)
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|ByT5-small|AdamW|TweetQA|65.68|65.7|69.67|69.7|

>复现代码训练日志：
[复现代码训练日志](https://github.com/yoreG123/Paddle-ByT5/blob/main/logs/tweetqa.log)

>
>#### 在Xsum数据集的测试效果如下表。

|模型 |opt|数据集|BLEU|BLEU(原论文)
| :---: | :---: | :---: | :---: | :---: 
|ByT5-small|AdamW|Xsum|9.36|9.1

>复现代码训练日志：
[复现代码训练日志](https://github.com/yoreG123/Paddle-ByT5/blob/main/logs/xsum.log)

同时logs目录之下提供visualDL日志

## 3 数据集
我们主要复现tweetqa和xsum数据集的精度, 数据集，

tweetqa数据集可以前往此处下载:
[地址](https://tweetqa.github.io/)

xsum数据集可在此处下载: 
[地址](https://aistudio.baidu.com/aistudio/datasetdetail/122619)


## 4环境依赖
运行以下命令即可配置环境(由于nltk在源码中复制，所以可以不安装)
```bash
pip install paddlepaddle-gpu
pip install sacrebleu
pip install rouge_score
```

## 5 快速开始
由于代码在paddlenlp原始代码中增加一些代码，比如tweetqa以及xsum数据集的加载，以及byt5Tokenizer的部分代码，所以首先执行pip uninstall paddlenlp卸载原始paddlenlp，之后cd到byt5目录之下，便可以引入加入本论文代码的paddlenlp。
1. 权重转换对齐：执行compare.py，注意修改模型路径。结果发现平均误差3.3157e-07符合精度要求。compare.py脚本参考https://github.com/JunnYu/paddle_t5/blob/main/compare.py
2. 微调：进入byt5目录，微调tweetqa执行python tasks/tweetqa/run.py --model_name_or_path 初始模型路径；微调xsum执行python tasks/xsum/run.py --model_name_or_path 初始模型路径，其他参数参考args.py并自行调整。
微调之后的预训练模型链接：
tweetqa18450：https://aistudio.baidu.com/aistudio/datasetdetail/128224
xsum380000（由于训练资源有限，batchsize设为1，所以训练步数较大）：https://aistudio.baidu.com/aistudio/datasetdetail/127876
3. 验证：进入byt5目录，要验证xsum模型执行python tasks/xsum/eval.py --model_name_or_path 微调后模型路径；要验证tweetqa模型执行python tasks/tweetqa/eval.py --model_name_or_path 微调后模型路径；注意修改--evaluate_file 为 相应数据集的dev.json文件。
最终结果截图如下
4. aistudio链接：

## 6 主要代码路径
1. tokenizer代码（modeling代码同t5）
byt5tokenizer：byt5/paddlenlp/transformers/byt5
2. 数据集加载：
tweetqa18450：byt5/paddlenlp/datasets/tweetqa_paddle.py
xsum：byt5/paddlenlp/datasets/xsum.py
注意，如需加载数据集，应该将两个py文件中的数据集路径加以修改
3. tasks目录中包含两个微调任务的训练与测试脚本，run.py执行微调，args.py设置微调参数，eval.py验证结果

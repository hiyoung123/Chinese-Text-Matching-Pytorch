# Chinese-Text-Matching-Pytorch
基于Pytorch实现的中文文本匹配脚手架，以及常用模型对比。

## Structure
```python
├─config              # 配置文件目录
├─data                # 数据目录
├─log                 # log 目录
├─output              
│  ├─model_data       # 模型存放目录
│  └─result           # 生成结果目录
├─pretrain            # 预训练模型存放目录
├─src                 # 主要代码
│  ├─datasets         # dataset 
│  ├─models           # model
│  └─tricks       
│     ├─adversarial   # 对抗训练
│     └─loss          # 特殊 loss
└─utils               # 工具代码
    
```

## Data
数据预处理
```bash
python src\process.py \
    --data_dir data/THUCNews \
    --out_dir data/THUCNews/processed \
    --max_vocab_size 2000000 \
    --min_freq 0 \
    --vocab_path data/THUCNews/vocab.pkl \
    --vector_path data/THUCNews/sgns.sogou.word/sgns.sogou.word \
    --embedding_path data/THUCNews/embedding.pkl
```
参数说明：
* --data_dir： 源数据存储文件夹，文件夹下需要有 train.txt, dev.txt, test.txt 三个文件。
* --out_dir： 输出文件夹，处理后的数据存储位置。
* --max_vocab_size： 词汇表最大容量。
* --min_freq： 最小词频。
* --vocab_path： 词汇表存储路径。
* --vector_path： 预训练词向量路径。
* --embedding_path： 对齐后的词向量矩阵存储路径。

## Model

## Trick
1. Flooding
2. FGM
3. Focal Loss
4. Dice Loss
5. Label Smooth
6. scheduler
7. max_gradient_norm
8. fp16
9. lookahead
10. MSD
11. 初始化权重

## Pre Trained

## Run Code

```bash
# 使用 ESIM 模型 训练并做验证 按照 epoch 计算 val score
python run.py --model ESIM

# 使用 ESIM 模型 训练并做验证 按照 step 计算 val score， 早停 限制 1000 步
python run.py --model ESIM --save_by_step True --patience 1000

# 使用 ESIM 模型 训练并做验证 按照 step 计算 val score，使用 label_smooth, 并指定任务名为 label_smooth.
python run.py --model ESIM --save_by_step True --patience 1000 --loss_type label_smooth --task_name label_smooth
```
训练参数过多，可以使用 --help 进行查询使用。

## Result

## Note
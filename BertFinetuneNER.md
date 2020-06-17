# BERT Fine-tune model

### Abstract

这词使用的是huggingface/transformers库来对bert模型进行fine-tune，用于命名实体识别(NER)任务。NER和POS的本质都是字符级的分类任务(token-classification)

### Data preprocessing

本次使用的数据集来自开源的中文NER[数据集](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master/ner)

- validation.txt示例如下，train.txt, test.txt的格式都是这样，每行word+一个空格+label

  ~~~txt
  记 O
  得 O
  上 B_Time
  世 I_Time
  纪 I_Time
  5 I_Time
  0 I_Time
  年 I_Time
  代 I_Time
  中 I_Time
  期 I_Time
  ， O
  ~~~

- 但是这个数据集缺少一个label.txt, 所以需要我们去制作一个

  首先需要学习几个文本操作命令(Linux命令，在win下可以用git CMD来模拟)

  1. ##### cut命令

     基本说明：
     cut 命令从文件的每一行剪切字节、字符和字段并将这些字节、字符和字段写至标准输出

     参数说明：
     -b ：以字节为单位进行分割。这些字节位置将忽略多字节字符边界，除非也指定了 -n 标志
     -c ：以字符为单位进行分割
     -d ：自定义分隔符，默认为制表符也就是\t
     -f ：与-d一起使用，指定显示哪个区域
     -n ：取消分割多字节字符。仅和 -b 标志一起使用。如果字符的最后一个字节落在由 -b 标志的 List 参数指示的范围之内，该字符将被写出；否则，该字符将被排除

  2. ##### grep命令

     基本说明：
     grep 指令用于查找内容包含指定的范本样式的文件，如果发现某文件的内容符合所指定的范本样式，预设 grep 指令会把含有范本样式的那一列显示出来。*简单来说，就是查找文本文件中符合或者不符合某些规则的行并把它打印出来*

     参数说明：略

     使用例子：
     查找后缀有 file 字样的文件中包含 test 字符串的文件，并打印出该字符串的行
     `grep test *file `

     查找指定目录/etc/acpi 及其子目录（如果存在子目录的话）下所有文件中包含字符串"update"的文件，并打印出该字符串所在行的内容
     `grep -r update /etc/acpi `

     前面各个例子是查找并打印出符合条件的行，通过"-v"参数可以打印出不符合条件行的内容。

     查找文件名中包含 test 的文件中不包含test 的行
     `grep -v test *test*`

  3. ##### tr命令

     基本说明：
     有点像python字符串的replace方法

     使用例子：
     将文件testfile中的小写字母全部转换成大写字母
     `cat testfile |tr a-z A-Z ` 或者`cat testfile |tr [:lower:] [:upper:] `

     将train.txt文件中的制表符(\t)转变为一个空格
     `cat train.txt | tr '\t' ' '`

  4. ##### sort命令

     基本说明：
     就是排序，以默认的方式将文本文件的第一列以ASCII 码的次序排列，并将结果输出到标准输出

     参数说明：
     -r : 反序
     -u: 排序并去重

- 生成label.txt

  需要取label也就是空格分割后的第二个，取不含有"^$"的行，排序并去重

  `cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort -u > labels.txt`

  得到的label.txt是应该如下的内容，每行只含有一个label

  ~~~
  B_Abstract
  B_ABstract
  B_Abstract
  B_ABstract
  B_Abstract
  B_ABstract
  B_Abstract
  B_Location
  B_Metric
  B_Organization
  B_Person
  B_Physical
  B_Term
  B_Thing
  B_Time
  I_Abstract
  I_ABstract
  I_Abstract
  I_ABstract
  I_Abstract
  I_ABstract
  I_Abstract
  I_ABstract
  I_Abstract
  I_Location
  I_Metric
  I_Organization
  I_Person
  I_Physical
  I_Term
  I_Thing
  I_Time
  O
  ~~~

### Configuration and prepare to fine-tune

- ##### configuration and run

  ~~~bash
  export MAX_LENGTH=128
  export BERT_MODEL=bert-base-chinese
  export OUTPUT_DIR=ner-model-zh
  export BATCH_SIZE=32
  export NUM_EPOCHS=3
  export SAVE_STEPS=750
  export SEED=1
  ~~~

  ~~~
  python run_ner.py --data_dir ./ \
  --labels ./labels.txt \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed $SEED \
  --do_train \
  --do_eval \
  --do_predict
  ~~~

- ##### or use json file(recommended)

  config.json文件

  ~~~json
  {
      "data_dir": ".",
      "labels": "./labels.txt",
      "model_name_or_path": "bert-base-chinese",
      "output_dir": "ner-model-zh",
      "max_seq_length": 128,
      "num_train_epochs": 3,
      "per_device_train_batch_size": 32,
      "save_steps": 750,
      "seed": 1,
      "do_train": true,
      "do_eval": true,
      "do_predict": true
  }
  ~~~

  `python run_ner.py path/to/config.json`

### Predictions

- 训练完成后，将会在运行的目录下得到**ner-model-zh**这个模型文件夹（前面的config指定的output_dir）

- 编写predict.py文件

  ~~~python
  import torch
  from transformers import (
      AutoModelForTokenClassification,
      AutoTokenizer
  )
  
  # 模型的路径和标签路径
  MODEL_PATH = r"path/to/ner-model-zh"
  LABEL_PATH = r"path/to/label.txt"
  
  # 模型对象
  model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
  # 分词器对象
  tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
  # 标签列表
  with open(PATH, "r",encoding="utf8") as f:
      label_list = [i.strip() for i in f.readlines()]
  
  # 输入的句子
  sequence = "我今天想吃麦当劳随心搭"
  
  tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
  inputs = tokenizer.encode(sequence, return_tensors="pt")
  
  outputs = model(inputs)[0]
  predictions = torch.argmax(outputs, dim=2)
  
  res = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]
  for token, label in res:
      print(token, label)
  ~~~

  output:

  ~~~bash
  [CLS] O
  我 B_Person
  今 B_Time
  天 I_Time
  想 O
  吃 O
  麦 B_Thing
  当 I_Thing
  劳 I_Thing
  随 O
  心 O
  搭 O
  [SEP] O
  ~~~

  
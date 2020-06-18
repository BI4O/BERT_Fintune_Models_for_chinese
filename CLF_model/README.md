# 文本分类 text classification

## GLUE

一般来说，NLP任务分为NLU(自然语言理解)和NLG(自然语言生成)，在NLU方面通常会用
GLUE(General Language Understanding Evaluation)排行榜作为例子，它包括了九项NLU任务

1. **CoLa**(Corpus of Linguitsic Acceptability): 纽约大学发布的有关**语法**的数据集，给定一个句子，判断其语法是否正确，属于**单个句子的文本二分类任务**
2. **SST**(Stanford Sentiment Treebank): 斯坦福大学发布的一个**情感分析**数据集，也是属于**单句子的分类任务**，主要针对电影评论（SST-2是二分类， SST-5是五分类）
3. **MRPC**(Microsoft Research Paraphrase Corpus): 微软发布的，判断两个给定句子是否具有相同的语义，属于句子对的二分类任务
4. **STS-B**(Semantic Textual Similarity Benchmark): 用1-5分来表征两个句子的相似性，本质是一个回归问题，但依然可以归为句子对的**句子对的五分类任务**
5. **QQP**(Quora Question Pairs): 是由Quora发布的两个句子是否语义一致的数据集，属于句子对的文本二分类任务
6. **MNLI**(Multi-Genre Natural Language Inference): 同样由纽约大学发布的，是一个文本蕴含的任务，在给定前提下，需要判断假设是否成立。属于**句子对的三分类任务**
7. **RTE**(Recognizing Textual Entailment): 和MNLI类似，也是一个文本蕴含任务，不过MNLI是三分类任务，RTE至于要判断两个句子是否能推断或对齐，属于**句子对的文本二分类任务**
8. **QNLI**(Question Natural Language Inference): 前身是SQuAD1.0数据集，给定一个问句，需要判断给定文本中是否包含该问句的真确答案，属于**句子对的文本二分类任务**
9. **WNLI**(Winograd Natural Language Inference): 也是一个文本蕴含任务，

## BERT fine tune for multi-label classification

- ### Abstract

  首先确保下列的依赖库已经安装好

  1. Pandas
  2. Pytorch
  3. Pytorch Utils for Dataset and Dataloader
  4. Transformers
  5. BERT Model and Tokenizer

  `pip install transformers`

  ~~~python
  import numpy  as np
  import pandas as pd
  from sklearn import metrics
  import transformers
  import torch
  from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
  from transformers import BertTokenizer, BertModel, BertConfig
  
  # gpu配置
  from torch import cuda
  device = 'cuda' if cuda.is_available() else 'cpu'
  ~~~

- ### Data preprocessing

  假设你已经下载了`train.csv`并已经解压在了`data`文件夹下

  ~~~python
  df = pd.read_csv('./data/train.csv')
  df['list'] = df[df.columns[2:]].values.tolist()
  new_df = df[['comment_text', 'list']].copy()
  new_df.head()
  ~~~

  |      |                                      comment_text |               list |
  | ---: | ------------------------------------------------: | -----------------: |
  |    0 | Explanation\nWhy the edits made under my usern... | [0, 0, 0, 0, 0, 0] |
  |    1 | D'aww! He matches this background colour I'm s... | [0, 0, 0, 0, 0, 0] |
  |    2 | Hey man, I'm really not trying to edit war. It... | [0, 0, 0, 0, 0, 0] |
  |    3 | "\nMore\nI can't make any real suggestions on ... | [0, 0, 0, 0, 0, 0] |
  |    4 | You, sir, are my hero. Any chance you remember... | [0, 0, 0, 0, 0, 0] |

  ##### 准备Dataset和Dataloader

  ###### Dataset类: 决定了待分类文本的是怎么经过处理的

  - 接受`tokenizer`,`dataframe` 和`max_length`作为输入和生成BERT需要的的，已经被分词器处理的分词序列和文本标签

  - 这次使用的是BERT的分词器来处理`comment_text`列的内容

  - 这个类可以使用`encode_plus`方法去做分词，这样生成的输出是：`ids`,`attention_mask`,`token_type_ids`

    （注意这里distillbert和bert有点不一样，bert只会生成`token_type_ids`）

  - `targest`是一个列表，表示多标签的独热编码向量（元素为0或者1）

  - 一般CustomDataset会创建两个dataset，一个用于训练，一个用于测试

  - 一般训练数据占总数据集的80%

  - 测试集是用来测试这个模型的最终表现的，需要是训练的时候没有看过的数据

  ###### Dataloader类: 决定了已经经过预处理的文本是怎么通过batch方式喂进神经网络模型的

  - Dataloader是用来创建training dataloader和validation dataloader的，他们将作为模型的直接输入。Dataloader是必须的因为主机的内存是有限的，不可能一开始就把全部的数据load到内存，所以需要对数据进行分批放入，这就是dataloader做的事情
  - Dataloader接受的基本参数为`batch_size`和`max_len`

  ~~~python
  # 定义一些会用到的常量
  MAX_LEN = 200
  TRAIN_BATCH_SIZE = 8
  VALID_BATCH_SIZE = 4
  EPOCH = 1
  LEARNING_RATE = 1e-05
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  ~~~

  定义CustomDataset

  ~~~python
  class CustomDataset(Dataset):
      def __init__(self, dataframe, tokenizer, max_len):
          self.tokennizer = tokenizer
          self.data = dataframe
          self.comment_text = self.data.comment_text
          self.targets = self.data.list
          self.max_len = max_len
      
      def __len__(self):
          return len(self.comment_text)
      
      def __getitme__(self, index):
          comment_text = str(self.comment_text)
          comment_text = " ".join(comment_text.split())
          
          inputs = self.tokenizer.encode_plus(
          	comment_text,
              None,
              add_special_tokens=True,
              max_length=self.max_len,
              pad_to_max_length=True,
              return_token_type_ids=True
          )
          ids = inputs['input_ids']
          mask = inputs['attention_mask']
          token_type_ids = inputs['token_type_ids']
          
          return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(mask, dtype=torch.long),
              'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
              'targets': torch.tensor(self.targets[index], dtype=torch.float)
          }
  ~~~

  创建Dataloader

  ~~~python
  train_size = 0.8
  # 从源数据中随机抽取80%的数据作为训练集
  train_dataset = new_df.sample(frac=train_size, random_state=200).reset_index(drop=True)
  test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
  
  # 看看分好的数据各有多少条
  print("FULL Dataset: {}".format(new_df.shape))
  print("TRAIN Dataset: {}".format(train_dataset.shape))
  print("TEST Dataset: {}".format(test_dataset.shape))
  
  # 创建CUstomDataset
  training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
  testingt_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
  
  # 创建DataLoader
  training_loader = DataLoader(
      training_set, shuffle=True, num_workers=0
  )
  testing_loader = DataLoader(
  	testing_set, shuffle=True, num_workers=0
  )
  ~~~

- ### Creating Neural Network for Fine Tuning

  ##### Neural Network

  - Bert模型后面需要接`Dropout`和`Linear Layer`，作用分别是**正则化**和**分类**
  - 在正向传播的循环中，将会有两个输出来自`BertModel`层
  - 其中第二个输出或者叫`pooled output`会被接到`Dropout` 层，再然后得到的输出会被接到`Linear`全连接层
  - 注意`Linear`层的维度是**6**因为这就是分类类别的总数

  ##### Loss Funtion and Optimizer

  - 损失函数将会定义为`loss_fn`
  - 本次多标签分类任务的损失函数实际上将会是6个`Binary Cross Entropy`二元交叉熵
  - 优化器是用来根据损失来优化模型参数的

  ~~~python
  class BERTClass(torch.nn.Module):
      def __init__(self):
          super(BERTClass, self).__init__()
          self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
          self.l2 = torch.nn.Dropout(0.3)
          self.l3 = torch.nn.Linear(768, 6)
          
      def forward(self, idx, mask, token_type_ids):
          _, output_1 = self.l1(
              ids,
              attnetion_mask=mask,
          	token_type_ids=token_type_ids
          )
          output_2 = self.l2(output_1)
          output = self.l3(output_2)
          return output
      
  # 损失函数
  def loss_fn(outputs, targets):
      return torch.nn.BCEWithLogitsLoss()(outputs, targets)
  
  # 优化器
  optimizer = torch.optim.Adam(para=model.parameters(),lr=LEARNING_RATE)
      
  
  # 创建模型实例
  model = BERTClass()
  model.to(device)
  ~~~

- ### Fine Tuning the Model

  完成了上述的所有步骤之后，最后一步的微调是比较轻松的。首先定义训练方法（需要指定训练的Epoch）一个Epoch代表全部的训练数据被遍历一遍

  - DataLoader会以batch的形式向模型输送数据
  - 损失函数是以模型计算出来的数值和实际的分类编码进行比对而计算出来的
  - 优化器会根据损失值不断对模型的参数进行优化改进，使得模型的损失越来越小
  - 每5000steps(一个batch输入了模型就代表一个step)损失值将会被打印在控制台上

  ~~~python
  def train(epoch):
      model.train()
      for steps, data in enumerate(training_loader, 0):
          ids = data['ids'].to(device, dtype=torch.long)
          mask = data['mask'].to(device, dtype=torch.long)
          token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
          targets = data['targets'].to(device, dtype = torch.float)
  
          outputs = model(ids, mask, token_type_ids)
  
          optimizer.zero_grad()
          loss = loss_fn(outputs, targets)
          if _%5000==0:
              print(f'Epoch: {epoch}, Loss:  {loss.item()}')
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
  # 开始训练
  for epoch in range(EPOCHS):
      train(epoch)
  ~~~

  ###### output

  ~~~bash
  Epoch: 0, Loss:  0.8253790140151978
  Epoch: 0, Loss:  0.1364113688468933
  Epoch: 0, Loss:  0.06799022853374481
  Epoch: 0, Loss:  0.022630181163549423
  ~~~

- ### Validation the Model

  拿剩下的20%未经训练的数据作为检验模型的测试集，测试模型的指标捡回是：

  - Accuracy Score
  - F1 Micro
  - F1 Macro

  ~~~python
  def validation(epoch):
      model.eval()
      fin_targets=[]
      fin_outputs=[]
      with torch.no_grad():
          for _, data in enumerate(testing_loader, 0):
              ids = data['ids'].to(device, dtype = torch.long)
              mask = data['mask'].to(device, dtype = torch.long)
              token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
              targets = data['targets'].to(device, dtype = torch.float)
              outputs = model(ids, mask, token_type_ids)
              fin_targets.extend(targets.cpu().detach().numpy().tolist())
              fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
      return fin_outputs, fin_targets
  
  # 开始测试
  for epoch in range(EPOCHS):
      outputs, targets = validation(epoch)
      outputs = np.array(outputs) >= 0.5
      accuracy = metrics.accuracy_score(targets, outputs)
      f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
      f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
      print(f"Accuracy Score = {accuracy}")
      print(f"F1 Score (Micro) = {f1_score_micro}")
      print(f"F1 Score (Macro) = {f1_score_macro}")
  ~~~

  ###### output

  ~~~bash
  Accuracy Score = 0.9354828601867519
  F1 Score (Micro) = 0.8104458787743897
  F1 Score (Macro) = 0.6943681099377335
  ~~~

  
import os
import sys
import pandas as pd
from torch.utils.data import Dataset,random_split,DataLoader
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.optim import Adam

sys.path.append(os.path.dirname(os.path.dirname(__file__)))



class MyDataset(Dataset):
    """
    创建Dataset
    """
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("transformers-code-master/01-Getting Started/04-model/ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()
        
    def __getitem__(self, index):
        return self.data.loc[index]['review'], self.data.loc[index]['label']
    
    def __len__(self):
        return len(self.data)
        
def collate_fuc(batch):
    """
    用于在构建dataloader时，批量进行分词处理，这种做法会更快
    """
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokernizer(texts, max_length=128,padding="max_length",truncation=True,return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs


def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in valloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()
    return acc_num/len(valset)
    

def train(epoch = 3,log_step = 100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k,v in batch.items()}
            # 梯度归零
            optimizer.zero_grad()
            output = model(**batch)
            # 计算损失
            output.loss.backward()
            # 更新梯度
            optimizer.step()
            # 打印日志
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        acc = evaluate()
        print(f"ep: {ep}, acc:{acc}")
    

if __name__ == "__main__":
    
    # 构造dataset
    dataset = MyDataset()
    
    # 划分数据集
    trainset, valset = random_split(dataset,lengths=[0.9, 0.1])
    
    # 构建tokernizer
    tokernizer = AutoTokenizer.from_pretrained("hfl/rbt3")
    
    # 创建dataloader
    trainloader = DataLoader(trainset,batch_size=32,shuffle=True,collate_fn=collate_fuc)
    valloader = DataLoader(valset,batch_size=64,shuffle=False,collate_fn=collate_fuc)
    
    # print(next(enumerate(valloader))[1])
    
    # 创建模型以及优化器、学习率
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
    optimizer = Adam(model.parameters(),lr=2e-5)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 训练与验证
    train()
    
    
    # 模型预测
    sen = "我觉得这家酒店不错，很好吃"
    id2_label = {0:"差评",1:"好评"}
    with torch.inference_mode():
        inputs = tokernizer(sen,return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k : v.cuda() for k,v in inputs.items()}
        logits = model(**inputs).logits
        pred = torch.argmax(logits,dim=-1)
        print(f"输入:{sen}\n模型预测结果:{id2_label.get(pred)}")
    
    # transformers实现
    from transformers import pipeline
    pipe = pipeline("text-classification",model="hfl/rbt3",tokenizer=tokernizer)
    pipe(sen)
    
    
    
    
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class SimpleGPT(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(SimpleGPT, self).__init__()
        
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def forward(self, prompt):
        # 将输入文本转化为模型可以理解的形式
        inputs = self.tokenizer(prompt, return_tensor = 'pt', padding = True, truncation=True)
        inputs = inputs.to(self.device)
        
        # 获取输出
        outputs = self.model(**inputs)
        hidden_states = outputs[0]     #  隐藏层状态，包含文本表示和自回归结果
        
        # 从最后一个token生成一个词（假设我们的任务是根据给定的提示生成写一个词）
        last_token_hidden = hidden_states[:, -1]
        
        # 取最后一个token的隐藏状态
        probs = self.model.get_logits(last_token_hidden)
        
        # 获取生成下一个词的概率分布
        next_word = self.tokenizer.decode([torch.argmax(probs).item()])
        # 根据最大概率选择下一个词并解码
        
        return next_word

        
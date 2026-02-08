import gc
import itertools
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import sys
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from pathlib import Path
from peft import PeftModel


def ensure_directory_exists(directory_path):
    """
    检查给定的路径是否存在，如果不存在则创建该文件夹。
    :param directory_path: 要检查和创建的文件夹路径（绝对或相对）
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    
 
    
    
class TextDataset(Dataset):
    def __init__(self ,usedf_,processor_, random_seed_, salectn_  ):
        random.seed(random_seed_)
        self.usedf_ = usedf_.sample(salectn_ if len(usedf_)>salectn_ else len(usedf_) ,  random_state = random_seed_ ).reset_index()
        self.random_seed_ =  random_seed_
        self.processor_ =  processor_
    def __len__( self ):
        return len(self.usedf_ )
    def content_2_message(self,title_, domain_):
        content_="""我这里有一些从各个网站抓取的数据集，网站存在一些错误/谣言信息，你需要甄别它（二分类任务），1代表为是错误/谣言信息， 0 代表不是
        一下是一些案例,content是抓取的信息，domain为对应网站，
        下面是你需要判别的信息:
        {"content":"%%content%%","domain":"%%domain%%"}
        请输出你的判别（0/1）:
        """
        content_= content_.replace(  "%%content%%",str(title_) ).replace(  "%%domain%%",str(domain_) )
        messages = [   {
        "role": "user",
        "content": [
            {"type": "text", "text": content_},
            ],
        }   ]
        return  messages
    def message_2_inputids_labelids(self,messages_, label_):
        text = self.processor_.apply_chat_template(   messages_, tokenize=False, add_generation_prompt=True)
        #print( text ,  label_)
        inputs = self.processor_(  text = [text],   max_length=512 , padding='max_length' ,truncation=True,  return_tensors="pt").to("cuda")
        answers  = self.processor_(   text=[str(label_)],   max_length=8 , padding='max_length' ,truncation=True,  return_tensors="pt").to("cuda")
        #labels = [lab_["input_ids"].clone() for lab_ in  model_inputs ]
        #for le in range(0, len(labels ) ):
        #    labels[le][model_inputs[le]["attention_mask"] == 0]= -100
        input_ids = torch.cat([inputs.input_ids, answers.input_ids], dim=1)
        attention_mask = torch.cat([inputs.attention_mask, answers.attention_mask], dim=1)
        labels = input_ids.clone()
        labels[:, :inputs.input_ids.shape[1]] = -100 
        labels[attention_mask == 0] = -100
        return input_ids,attention_mask,labels
        
    def __getitem__(self, idx):   
        #start_ ,end_= self.make_data( self.key_[idx], self.title_[idx], self.content_[idx] )
        id_ = self.usedf_['id'].to_list()
        title_ = self.usedf_.title.to_list()
        domain_ = self.usedf_.domain.to_list()
        label_ = self.usedf_['2_way_label'].to_list()
        messages =self.content_2_message( title_[idx],  domain_[idx] )
        input_ids,attention_mask,labelsid =self.message_2_inputids_labelids( messages, label_[idx ] )
        return id_[idx], title_[idx], domain_[idx],messages,label_[idx ],input_ids,attention_mask,labelsid



def custom_collate(batch):
    id_  =[item[0] for item in batch]
    title_  =[item[1] for item in batch]
    domain_  =[item[2] for item in batch]
    messages  =[item[3] for item in batch]
    label_  =[item[4] for item in batch]
    input_ids  =[item[5] for item in batch]
    attention_mask  =[item[6] for item in batch]
    labelsid  =[item[7] for item in batch]
    return  id_, title_, domain_, messages,label_,input_ids,attention_mask,labelsid




def do_eval(model,processor,train_dataloader,epochs_ ,dir_):
    start_time = time.time()
    return_ans= []
    return_label = []
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epochs+1}") 
    for step, batch in enumerate(progress_bar):
        #if step>50:return
        #batch = {k: v.to(  model.device) for k, v in batch.items()}
        #print( batch[-2])
        #print( len(batch[-2] ))
        text = processor.apply_chat_template(   batch[-5], tokenize=False, add_generation_prompt=True)
        inputs = processor(   text=text,   max_length=512 , padding='max_length' ,truncation=True,  return_tensors="pt").to("cuda")
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=1)
        generated_ids_trimmed = [ out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)  ]
        output_text = processor.batch_decode(    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return_ans+= output_text
        return_label+=batch[-4]
    return return_ans , return_label
    print(f"infer running time：{time.time() - start_time}秒")




df =  pd.read_table("multimodal_train.tsv", encoding='utf-8')


img_folder_path="trian_image/"
save_res_path="/data/home/zhangxian/论文_DCRs/data/Fakeddit/Lora" 
model_address= ['/data_nas/model_hub/Qwen2.5-VL-7B-Instruct']
idlst_= [f.replace(".jpg",'') for f in os.listdir(img_folder_path) if".jpg" in f ]
df = df[ df.id.isin(idlst_)   ]
train_id_list= df.sample(300, random_state = 55116 ).id.tolist()
df= df[~df.id.isin(train_id_list)  ]
print(df.head() ) 
print( 'the len of the dataset_',len( df ))
model_name= '/data_nas/model_hub/Qwen2.5-VL-7B-Instruct'
processor = AutoProcessor.from_pretrained(model_name)
textds  = TextDataset( df  ,processor ,  55116, 2000) 
#textds[2]
train_dataloader = DataLoader(textds, batch_size=6, shuffle=True, collate_fn=custom_collate ,drop_last=True )


global_step = 0
epochs=  2
grad_accum_steps = 2 
total_steps = len(train_dataloader) * epochs // grad_accum_steps


lora_path="/data/home/zhangxian/论文_DCRs/data/Fakeddit/Lora/" 


for model_name  in model_address:  
    print(f'train model {model_name}')
    model2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
       torch_dtype=torch.float16,
        device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name) 
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
        )
    tuned_model = PeftModel.from_pretrained(model, lora_path+model_name+'_lora_adapters_ep1')
    merged_model = tuned_model.merge_and_unload()
    merged_model.eval()
    #ensure_directory_exists( save_res_path  )
    ans,label = do_eval(merged_model,processor,train_dataloader,epochs ,save_res_path)
    ans_int =[ int(t) for t in ans ]




ans2,label2 = do_eval(model2,processor,train_dataloader,epochs ,save_res_path)
ans2_int =[ int(t) for t in ans2 ]

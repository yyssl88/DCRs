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
from qwen_vl_utils import process_vision_info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from pathlib import Path


def ensure_directory_exists(directory_path):
    """
    检查给定的路径是否存在，如果不存在则创建该文件夹。
    :param directory_path: 要检查和创建的文件夹路径（绝对或相对）
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    
    
class TextDataset(Dataset):
    def __init__(self ,usedf_,processor_, random_seed_, salectn_ , image_path_ ):
        random.seed(random_seed_)
        self.usedf_ = usedf_.sample(salectn_ if len(usedf_)>salectn_ else len(usedf_) ,  random_state = random_seed_ ).reset_index()
        self.random_seed_ =  random_seed_
        self.processor_ =  processor_
        self.image_path_ =  image_path_
    def __len__( self ):
        return len(self.usedf_ )
    def content_2_message(self,title_, domain_,id_):
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
             {"type": "image", "image":self.image_path_ + id_ +".jpg" },
            {"type": "text", "text": content_},
            ],
        }   ]
        return  messages
    def message_2_inputids_labelids(self,messages_, label_):
        text = self.processor_.apply_chat_template(   messages_, tokenize=False, add_generation_prompt=True)
        #print( text ,  label_)
        image_inputs, video_inputs = process_vision_info(messages_)
        inputs = self.processor_(  text = [text], max_pixels=152974 , images=image_inputs,videos=video_inputs,  max_length=512 , padding='max_length' ,truncation=True,  return_tensors="pt").to("cuda")
        answers  = self.processor_(   text=[str(label_)], max_pixels=152974 ,  max_length=8 , padding='max_length' ,truncation=True,  return_tensors="pt").to("cuda")
        #labels = [lab_["input_ids"].clone() for lab_ in  model_inputs ]
        #for le in range(0, len(labels ) ):
        #    labels[le][model_inputs[le]["attention_mask"] == 0]= -100
        input_ids = torch.cat([inputs.input_ids, answers.input_ids], dim=1)
        attention_mask = torch.cat([inputs.attention_mask, answers.attention_mask], dim=1)
        labels = input_ids.clone()
        labels[:, :inputs.input_ids.shape[1]] = -100 
        labels[attention_mask == 0] = -100
        return input_ids,attention_mask,inputs.pixel_values,inputs.image_grid_thw ,labels
    def __getitem__(self, idx):   
        #start_ ,end_= self.make_data( self.key_[idx], self.title_[idx], self.content_[idx] )
        id_ = self.usedf_['id'].to_list()
        title_ = self.usedf_.title.to_list()
        domain_ = self.usedf_.domain.to_list()
        label_ = self.usedf_['2_way_label'].to_list()
        messages =self.content_2_message( title_[idx],  domain_[idx] ,id_[idx] )
        input_ids,attention_mask,pixel_values ,image_grid_thw ,labelsid =self.message_2_inputids_labelids( messages, label_[idx ] )
        return id_[idx], title_[idx], domain_[idx],messages,label_[idx ],input_ids,attention_mask,pixel_values ,image_grid_thw ,labelsid



def custom_collate(batch):
    id_  =[item[0] for item in batch]
    title_  =[item[1] for item in batch]
    domain_  =[item[2] for item in batch]
    messages  =[item[3] for item in batch]
    label_  =[item[4] for item in batch]
    input_ids  =[item[5] for item in batch]
    attention_mask  =[item[6] for item in batch]
    pixel_values  =[item[7] for item in batch]
    image_grid_thw  =[item[8] for item in batch]
    labelsid  =[item[9] for item in batch]
    return  id_, title_, domain_, messages,label_,input_ids,attention_mask,pixel_values ,image_grid_thw ,labelsid




def do_train(model,processor,train_dataloader,epochs_ ,dir_):
    start_time = time.time()
    for epochs in range(epochs_):
        trainloss = [ ]
        epoch_loss = 0.0
        best_accuracy= np.inf
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epochs+1}") 
        for step, batch in enumerate(progress_bar):
            #if step>50:return
            #batch = {k: v.to(  model.device) for k, v in batch.items()}
            #print( batch[-2])
            #print( len(batch[-2] ))
            #text = processor.apply_chat_template(   batch[-2], tokenize=False, add_generation_prompt=True)
            #inputs = processor(   text=[text],   max_length=512 , padding='max_length' ,truncation=True,  return_tensors="pt").to("cuda")
            #answers  = processor(   text=[batch[-1]],   max_length=8 , padding='max_length' ,truncation=True,  return_tensors="pt").to("cuda")
            ##labels = [lab_["input_ids"].clone() for lab_ in  model_inputs ]
            ##for le in range(0, len(labels ) ):
            ##    labels[le][model_inputs[le]["attention_mask"] == 0]= -100
            #input_ids = torch.cat([inputs.input_ids, answers.input_ids], dim=1)
            #attention_mask = torch.cat([inputs.attention_mask, answers.attention_mask], dim=1)
            #labels = input_ids.clone()
            #labels[:, :inputs.input_ids.shape[1]] = -100 
            #labels[attention_mask == 0] = -100
            ####
            #print( torch.stack(batch[-3]).to("cuda")[0].shape )
            #print( torch.stack(batch[-2]).to("cuda")[0].shape )
            #print( torch.stack(batch[-1]).to("cuda")[0].shape )
            #print( batch[-3].shape )
            #print( batch[-2].shape )
            print(step, batch[ 0 ])
            outputs = model( input_ids= torch.stack(batch[-5]).squeeze(1).to("cuda")
                            ,attention_mask= torch.stack(batch[-4]).squeeze(1).to("cuda")
                            ,pixel_values  =  torch.concat(batch[-3]).to("cuda")
                            ,image_grid_thw =torch.stack(batch[-2 ]).squeeze(1).to("cuda")
                                , labels=  torch.stack( batch[-1]).squeeze(1).to("cuda")  )
            ##
            loss = outputs.loss
            loss = loss / grad_accum_steps 
            #trainloss.append(loss.cpu().detach().numpy()  )
            epoch_loss += loss.item()   
            loss.backward() 
            global global_step
            global_step += 1 
            gc.collect()
            if (step + 1) % grad_accum_steps == 0 or step == len(train_dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                #scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_postfix({
                    "loss": f"{loss.item()*grad_accum_steps:.4f}",
                 #   "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
        avg_loss = epoch_loss / step
        print(f"Epoch {epochs+1} | Average Loss: {avg_loss:.4f}")
        model.save_pretrained(f"{dir_+model_name}_textImg_lora_adapters_ep{epochs}")
    print(f"train running time：{time.time() - start_time}秒")




df =  pd.read_table("multimodal_train.tsv", encoding='utf-8')


img_folder_path="trian_image/"
save_res_path="/data/home/zhangxian/论文_DCRs/data/Fakeddit/Lora" 
model_name = '/data_nas/model_hub/Qwen2.5-VL-7B-Instruct'
image_path = "/data/home/zhangxian/论文_DCRs/data/Fakeddit/trian_image/"

idlst_= [f.replace(".jpg",'') for f in os.listdir(img_folder_path) if".jpg" in f ]
df = df[ df.id.isin(idlst_)   ]
print(df.head()) 
print( 'the len of the dataset_',len( df ))
processor = AutoProcessor.from_pretrained(model_name)
textds  = TextDataset( df  ,processor ,  55116, 3000  ,image_path ) 
#textds[2]
#torch.manual_seed(88) 
#fixed_generator = torch.Generator().manual_seed(88)
train_dataloader = DataLoader(textds, batch_size = 6,   shuffle= False, collate_fn=custom_collate ,drop_last=True )
global_step = 0
epochs=  2
grad_accum_steps = 2 
total_steps = len(train_dataloader) * epochs // grad_accum_steps






print(f'train model {model_name}')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
model_lora = get_peft_model(model, config)
model_lora.print_trainable_parameters()
model_lora.train()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.00013, weight_decay=0.000001)
ensure_directory_exists( save_res_path  )
do_train(model_lora,processor,train_dataloader,epochs ,save_res_path)

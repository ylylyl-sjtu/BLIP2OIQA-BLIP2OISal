import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sys
sys.path.append('/DATA/DATA1/yangliu/code/config')
from options import *
import sys
sys.path.append('/DATA/DATA1/yangliu/code/config')
from utils import *
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from models.BLIP2OIQA import *
from transformers import BertTokenizer
import clip
from ERP_Process import Equirec2Perspec as E2P

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")



def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    
    tokenizer = BertTokenizer.from_pretrained('/DATA/DATA1/yangliu/pretrained_models/bert-base-uncased')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
        
    def forward(self, input):
        return self.layers(input)


class blip2oiqa(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        #self.preprocess = _transform(config['BLIP']['image_size'])
        self.tokenizer = init_tokenizer()
        self.device = device
        self.blip2 = BLIP2OIQA_qformer()
        checkpoint = torch.load('/DATA/DATA1/yangliu/pretrained_models/blip2_pretrained.pth',
                                map_location="cpu")
        state_dict = checkpoint["model"]
        msg = self.blip2.load_state_dict(state_dict, strict=False)
        print(msg)
        for name, parms in self.blip2.named_parameters():
            print(name)
            if 'Qformer' in name:
                parms.requires_grad_(False)
       
        self.preprocess = _transform(config['BLIP']['image_size'])
        
        if opts.fix_base:
            self.blip.requires_grad_(False)
        
        # for name, parms in self.blip2.named_parameters():
        #     if '_proj' in name:
        #         parms.requires_grad_(False)
        # for name, parms in self.blip2.named_parameters():
        #     if 'Qformer1' in name:
        #         parms.requires_grad_(False)
        
        # # fix certain ratio of layers
        # self.image_layer_num = 24 if config['BLIP']['vit'] == 'large' else 12
        # if opts.fix_rate > 0:
        # #     text_fix_num = "layer.{}".format(int(12 * opts.fix_rate))
        #     image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
        # #     # for name, parms in self.blip2.text_encoder.named_parameters():
        # #     #     parms.requires_grad_(False)
        # #     #     if text_fix_num in name:
        # #     #         break
        #     for name, parms in self.blip2.visual_encoder.named_parameters():
        #         parms.requires_grad_(False)
        #         if image_fix_num in name:
        #             break


    def loose_layer(self, fix_rate):
        text_layer_id = [f"layer.{id}" for id in range(int(12 * fix_rate), 13)]
        image_layer_id = [f"blocks.{id}" for id in range(int(24 * fix_rate), 25)]
        for name, parms in self.blip.text_encoder.named_parameters():
            for text_id in text_layer_id:
                if text_id in name:
                    parms.requires_grad_(True)
        for name, parms in self.blip.visual_encoder.named_parameters():
            for image_id in image_layer_id:
                if image_id in name:
                    parms.requires_grad_(True)


    def forward(self, batch_data):

        batch_data = self.encode_pair(batch_data)
        loss = batch_data['loss']
        
        emb_better1 = batch_data['emb_better1']
        reward_better1 = emb_better1
        reward_better1 = reward_better1[:,None]
        reward_worse1 = batch_data['emb_worse1']
        reward_worse1 = reward_worse1[:,None]
        # print("---------------------")
        # print("reward:worse & better")
        # print("rewardworse1: " ,reward_worse1)
        # print("rewardbetter1: " ,reward_better1)
        reward1 = torch.concat((reward_better1, reward_worse1), dim=1)
       
        emb_better2 = batch_data['emb_better2']
        reward_better2 = emb_better2
        reward_better2 = reward_better2[:,None]
        reward_worse2 = batch_data['emb_worse2']
        reward_worse2 = reward_worse2[:,None]
        # print("rewardworse2: " ,reward_worse2)
        # print("rewardbetter2: " ,reward_better2)
        reward2 = torch.concat((reward_better2, reward_worse2), dim=1)

        emb_better3 = batch_data['emb_better3']
        reward_better3 = emb_better3
        reward_better3 = reward_better3[:,None]
        reward_worse3 = batch_data['emb_worse3']
        reward_worse3 = reward_worse3[:,None]
        # print("rewardworse3: " ,reward_worse3)
        # print("rewardbetter3: " ,reward_better3)
        reward3 = torch.concat((reward_better3, reward_worse3), dim=1)

        return reward1, reward2, reward3, loss

    def inferenceall(self, batch_data):
        for id in range(5):
            id2 = id * 100
            img_path = f'/DATA/DATA1/yangliu/CNNIQA/dataset1/indoor_5_621.bmp'
            pil_image = Image.open(img_path)
            prompt2 = "What is the authenticity of the image?"
            prompt = "What is the quality of the image?"
            prompt3 = "What is the correspondence of the image?"
            prompt4 = "Assess the image from three perspectives: quality, authenticity, and correspondence"
            image = self.preprocess(pil_image)
            imager = image.unsqueeze(0)
            image2 = image.unsqueeze(0).to(self.device)
            out = self.blip2.generate({"image": image2, "prompt": prompt}) 
            out2 = self.blip2.generate({"image": image2, "prompt": prompt2}) 
            out3 = self.blip2.generate({"image": image2, "prompt": prompt3}) 
            out4 = self.blip2.generate({"image": image2, "prompt": prompt4}) 
            with open('train2-out1.txt','a') as f:
                f.write(str(id))
                f.write('\n')
                f.write(str(out))
                f.write('\n')
            with open('train2-out2.txt','a') as f2:
                f2.write(str(id))
                f2.write('\n')
                f2.write(str(out2))
                f2.write('\n')
            with open('train2-out3.txt','a') as f:
                f.write(str(id))
                f.write('\n')
                f.write(str(out3))
                f.write('\n')
            with open('train2-out4.txt','a') as f2:
                f2.write(str(id))
                f2.write('\n')
                f2.write(str(out4))
                f2.write('\n')



    def encode_pair(self, batch_data):

        text_ids = batch_data['text_ids']
        img_name = batch_data['img_name']
        #tq_ids = batch_data['tq_ids']
        #tq_mask = batch_data['tq_mask']
        #ti_ids = batch_data['ti_ids']
        #ti_mask = batch_data['ti_mask']
        #to_ids = batch_data['to_ids']
        #to_mask = batch_data['to_mask']
        #print(batch_data)
        #prompt = batch_data['prompt']

        text_mask = batch_data['text_mask']
        imgs_better = batch_data['imgs']
        text_ids = text_ids.view(text_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        #tq_ids = tq_ids.view(tq_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        #tq_mask = tq_mask.view(tq_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        #ti_ids = ti_ids.view(ti_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        #ti_mask = ti_mask.view(ti_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        
        #to_ids = to_ids.view(to_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        #to_mask = to_mask.view(to_mask.shape[0], -1).to(self.device)#prompt = prompt.view(prompt.shape[0], -1).to(self.device)
        # B,N,C,W,H = img_better.size()
        # img_better = img_better.view(-1,C,W,H)
        # print("Before mean operation: ", imgs_better.shape)
        # imgs_better = imgs_better.mean(2).to(self.device)  #[bs, views, C, H, W]-
        # print("After mean operation: ", imgs_better.shape)
        # with open("/DATA/DATA1/yangliu/code/output_log/test1.txt", "a") as file:
        #     # 将变量写入文件
        #     file.write(f"imgs_better.size(): \n")
        #     file.write(f"{imgs_better.size()}\n")
        # emb_better = emb_better[:, -1, :].float()
        emb_worse1 = batch_data['moz1']
        emb_worse1 = emb_worse1.to(self.device)
        emb_worse2 = batch_data['moz2']
        emb_worse2 = emb_worse2.to(self.device)
        emb_worse3 = batch_data['moz3']
        emb_worse3 = emb_worse3.to(self.device)
        emb_better1, emb_better2, emb_better3, quality_vector = self.blip2.forward(imgs_better, text_ids, text_mask)
        #loss, back_vector = self.blip2.forward_llm(image = img_better,vl_embeddings=quality_vector, tq_ids=tq_ids,tq_mask=tq_mask,ti_ids=ti_ids,ti_mask=ti_mask,to_ids=to_ids,to_mask=to_mask)
        #quality_vector = self.blip2.forward_3score(image = img_better, text_ids=text_ids, text_mask=text_mask)
        loss=0
        

        batch_data = {
            'img_name': img_name,
            'emb_better1': emb_better1,
            'emb_better2': emb_better2,
            'emb_better3': emb_better3,
            'emb_worse1': emb_worse1,
            'emb_worse2': emb_worse2,
            'emb_worse3': emb_worse3,
            'loss': loss,
        }
        print(img_name, '\n')
        print(emb_better1, '\n')
        print(emb_better2, '\n')
        print(emb_better3, '\n')

        return batch_data



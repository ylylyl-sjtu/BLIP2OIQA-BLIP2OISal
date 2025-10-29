import os
import json
import math
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('/DATA/DATA1/yangliu/code/config')
from utils import *
sys.path.append('/DATA/DATA1/yangliu/code/config')
from options import *
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import BertTokenizer
import clip
from transformers import LlamaTokenizer
from ERP_Process import Equirec2Perspec as E2P
import numpy as np
from torchvision import transforms

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
        # CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _transform_map(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        # CenterCrop(n_px),
        # _convert_image_to_rgb,
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_llm_tokenizer():
    #llm_model = "/home/wangjiarui/LAVIS/weight/vicuna-7b"
    llm_model = "/DATA/DATA1/yangliu/pretrained_models/vicuna-7b-v1.1"
    llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
    llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
    llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
    llm_tokenizer.padding_side = "right"
    return llm_tokenizer
    # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token


def init_tokenizer():
    
    #tokenizer = BertTokenizer.from_pretrained('/home/wangjiarui/ImageReward/ImageReward/models/BLIP/bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('/DATA/DATA1/yangliu/pretrained_models/bert-base-uncased')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def init_tokenizer2(truncation_side):
        #tokenizer = BertTokenizer.from_pretrained('/home/wangjiarui/ImageReward/ImageReward/models/BLIP/bert-base-uncased', truncation_side=truncation_side)
        tokenizer = BertTokenizer.from_pretrained('/DATA/DATA1/yangliu/pretrained_models/bert-base-uncased', truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

def custom_collate_fn(batch):
    pano_img = [item['pano'] for item in batch]
    # imgs = [item['imgs'] for item in batch]  # [batch_size, 6, C, H, W]
    maps = [item['maps'] for item in batch]
    text_ids = [item['text_ids'] for item in batch]
    text_mask = [item['text_mask'] for item in batch]
    img_name = [item['img_name'] for item in batch]
    # moz1 = [item['moz1'] for item in batch]
    # moz2 = [item['moz2'] for item in batch]
    # moz3 = [item['moz3'] for item in batch]
    # processing img
    # imgs = torch.stack([torch.stack(imgs_i, dim=0) for imgs_i in imgs], dim=0)
    
    # processing text_ids
    text_ids = torch.stack(text_ids, dim=0)
    text_mask = torch.stack(text_mask, dim=0)
    # moz1_tensors = [torch.tensor([m], dtype=torch.float32) for m in moz1]
    # moz2_tensors = [torch.tensor([m], dtype=torch.float32) for m in moz2]
    # moz3_tensors = [torch.tensor([m], dtype=torch.float32) for m in moz3]
    # moz1 = torch.stack(moz1_tensors, dim=0)
    # moz2 = torch.stack(moz2_tensors, dim=0)
    # moz3 = torch.stack(moz3_tensors, dim=0)
    # moz1 = torch.squeeze(moz1)
    # moz2 = torch.squeeze(moz2)
    # moz3 = torch.squeeze(moz3)
    pano_img = torch.stack(pano_img, dim = 0)
    maps = torch.stack(maps, dim = 0)
    # img_name = torch.stack(img_name, dim=0)
    # print("moz1", moz1)
    # print("moz2", moz2)
    # print("moz3", moz3)
    return {
        'pano': pano_img,
        'maps': maps,
        # 'imgs': imgs,  # [batch_size, 6, C, H, W]
        'text_ids': text_ids,
        'text_mask': text_mask,
        'img_name': img_name
        # 'moz1': moz1,
        # 'moz2': moz2,
        # 'moz3': moz3
    }


class datasetllm(Dataset):
    def __init__(self, dataset):
        self.preprocess = _transform((224, 224))
        self.preprocess_map = _transform_map((224, 224))
        self.tokenizer = init_tokenizer()
        self.tokenizer2 = init_tokenizer2(truncation_side="left")
        self.llm_tokenizer = init_llm_tokenizer()
        
        if dataset == "train":
            with open('/DATA/DATA1/yangliu/code/train_data_saliency_1.json', "r") as f:
                self.data = json.load(f)
        if dataset == "valid":
            with open('/DATA/DATA1/yangliu/code/test_data_saliency_1.json', "r") as f:
                self.data = json.load(f)
        if dataset == "test":
            with open('/DATA/DATA1/yangliu/code/test_data_saliency_1.json', "r") as f:
                self.data = json.load(f)


        #self.prompts, self.img_set, self.mos1, self.mos2, self.mos3, self.text_instruct, self.text_out = self.make_data()
        self.prompts, self.img_set, self.map_set, self.img_name = self.make_data()

        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))


    def __getitem__(self, index):
        item = {}
        '''only load pano_img, saliency_map and prompt'''
        prompt = self.prompts[index]
        imgpath = self.img_set[index]
        map_path = self.map_set[index]
        # print(map_path)
        #text_instruct = self.text_instruct[index]
        #text_out = self.text_out[index]
        #process images seperately
        item['pano'] = self._img_trans_pano(imgpath)
        # item['imgs'] = self._img_trans(imgpath)
        item['maps'] = self._img_trans_map(map_path)
        # print(item['maps'])
        item['text_ids'] , item['text_mask'] = self._txt_trans(prompt)
        item['img_name'] = self.img_name[index]
        # print(item['text_ids'])
        #item['tq_ids'], item['tq_mask'], item['ti_ids'], item['ti_mask']= self._texti_trans(text_instruct)
        #item['to_ids'], item['to_mask'] = self._texto_trans(text_out)
        # item['moz1'] = self.mos1[index]
        # item['moz2'] = self.mos2[index]
        # item['moz3'] = self.mos3[index]

        return item

    def __len__(self):
        return len(self.data)
    
    def store_dataset(self, dataset):
        makedir(config['pair_store_base'])
        torch.save(self.data, os.path.join(config['pair_store_base'], f"{dataset}.pth"))
    
    def text_io(self):
        with open('/home/wangjiarui/wangjiarui/ImageReward/train/src/text_i.txt','r') as f1:
            lines = f1.readlines()
        text_instruct = []
        for line in lines:
            text_instruct.extend(line.strip())

        with open('/home/wangjiarui/wangjiarui/ImageReward/train/src/text_o.txt','r') as f2:
            lines2 = f2.readlines()
        text_out = []
        for line in lines2:
            text_out.extend(line.strip())
        return text_instruct, text_out

    def _texti_trans(self, text_instruct):
        
        text_Qformer = self.tokenizer2(
                text_instruct,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
        
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            text_instruct,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=256,
        )
        return text_Qformer.input_ids, text_Qformer.attention_mask, text_input_tokens.input_ids, text_input_tokens.attention_mask

    def _texto_trans(self, text_out):  
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            text_out,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=256,
        )  
        return text_output_tokens.input_ids, text_output_tokens.attention_mask

    
    def _img_trans_pano(self, imgpath):
        
        pil_image = Image.open(imgpath)
        image = self.preprocess(pil_image)
        return image
    
    def _img_trans_map(self, imgpath):
        
        pil_image = Image.open(imgpath)
        # print(pil_image)
        image = self.preprocess_map(pil_image)
        # print(image.min())
        # print(image.max())
        return image
    

    def _img_trans(self, imgpath):

        if InterpolationMode:
            bicubic = InterpolationMode.BICUBIC
        else:
            bicubic = Image.BICUBIC

        equ = E2P.Equirectangular(imgpath)

        output_size = 512
        views = {
            'front': (110, 0, 0),
            'left': (110, 90, 0),
            'right': (110, -90, 0),
            'back': (110, 180, 0),
            'up': (110, 0, 90),
            'down': (110, 0, -90),
                }
        perspective_images_processed = []
        for view_name, (fov, theta, phi) in views.items():
            img = equ.GetPerspective(fov, theta, phi, output_size, output_size)
            img_pil = Image.fromarray(img)  # 假设img是一个numpy数组
            img_processed = self.preprocess(img_pil)
            perspective_images_processed.append(img_processed)

        return perspective_images_processed


    def _txt_trans(self, text):
        
        text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
        return text_input.input_ids, text_input.attention_mask
    


    def make_data(self): #read data
        img_set = []
        map_set = []
        prompts = []
        img_name = []
        # mos1 = []
        # mos2 = []
        # mos3 = []
        
        bar = tqdm(range(len(self.data)), desc=f'making dataset: ')
        for item in self.data:
            
            
            img_name.append(item['img'])
            #img_path = os.path.join('/home/wangjiarui/wangjiarui/ImageReward/allimg', item['img'])
            img_path = os.path.join('/DATA/DATA1/yangliu/data/pano600', item['img'])
            map_path = os.path.join('/DATA/DATA1/yangliu/data/fixation_map_0.4', item['map'])
            img_set.append(img_path)
            map_set.append(map_path)
            prompts.append(item["prompt"])
            # mos1.append(item['moz1'])
            # mos2.append(item['moz2'])
            # mos3.append(item['moz3'])
            #q1.append(item['q1'])
            #a1.append(item['a1'])
 
            bar.update(1)

        #return prompts, img_set, mos1, mos2, mos3, q1, a1
        return prompts, img_set, map_set, img_name

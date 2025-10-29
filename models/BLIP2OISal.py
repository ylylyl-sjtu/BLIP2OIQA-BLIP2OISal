"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from einops import rearrange
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
# from lavis.models.blip2_models.blip2 import (
#     Blip2Base,
#     compute_sim_matrix,
#     disabled_train,
# )
import sys
sys.path.append('/DATA/DATA1/yangliu/code/models')
from blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
sys.path.append('/DATA/DATA1/yangliu/code/ldm')
from attention import SpatialTransformer
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

class Upsample(nn.Module):
    '''
    steal from Restormer
    '''
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2)) # 通道*2/4=/2， 尺寸*2

    def forward(self, x):
        return self.body(x)
    
class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5


        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        '''
        :param x: [batch_size, seq_len, emb_dim]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        # b, c, h, w = x.shape

        # x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]
        # print("size of x")
        # print(x.shape)
        Q = self.Wq(x)  # [batch_size, seq_len, emb_dim]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim]
        V = self.Wv(context)

        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        # print("shape of att_weight")
        # print(att_weights.shape)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij,bjk->bik', att_weights, V)

        # print(out.shape)

        return out, att_weights
    
#@registry.register_model("blip2")
#@registry.register_model("blip2_feature_extractor")
class BLIP2OISal_qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        fix_rate = 0.7
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
       
        ## fix
        # fix_rate = 0.7

        self.image_layer_num = 38
        image_fix_num = "blocks.{}".format(int(self.image_layer_num * fix_rate))
    
        for name, parms in self.visual_encoder.named_parameters():
            parms.requires_grad_(False)
            print('name',name)
            if image_fix_num in name:
                break
        ##
        # if freeze_vit:
        #     for name, param in self.visual_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.visual_encoder = self.visual_encoder.eval()
        #     self.visual_encoder.train = disabled_train
        #     logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.quality1 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)
        self.quality2 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)
        self.quality3 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.itm_head2 = nn.Linear(self.Qformer.config.hidden_size, 1)
        self.itm_head3 = nn.Linear(self.Qformer.config.hidden_size, 3)
        print('self.Qformer.config.hidden_size',self.Qformer.config.hidden_size)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        '''decoder parameters'''
        self.cross_attn1 = SpatialTransformer(in_channels=1408,n_heads=8,d_head=16,context_dim=768, depth=1)
        self.cross_attn2 = SpatialTransformer(in_channels=512,n_heads=8,d_head=16,context_dim=768, depth=1)
        self.cross_attn3 = SpatialTransformer(in_channels=128,n_heads=8,d_head=16,context_dim=768, depth=1)

        self.encode3 = nn.Sequential(nn.Conv2d(1408, 1024, kernel_size=1, bias=False), Upsample(1024))
        self.encode2 = nn.Sequential(nn.Conv2d(1408, 1024, kernel_size=1, bias=False), Upsample(1024), Upsample(512), nn.Conv2d(256, 128, kernel_size=1, bias=False))
        self.encode1 = nn.Sequential(nn.Conv2d(1408, 1024, kernel_size=1, bias=False), Upsample(1024), Upsample(512), Upsample(256), nn.Conv2d(128, 32, kernel_size=1, bias=False))

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2816, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block
        
    def refine(self, emd):
        emd = rearrange(emd, 'b (h w) d -> b d h w', h=16)
        return emd
    
    def forward(self, image, map, text_ids, text_mask):
        # image = samples["image"]
        # text = samples["text_input"]
        visual_feature, x_list = self.visual_encoder(image)
        image_embeds = self.ln_vision(visual_feature)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )#(bs, 257, 1408)
        # print('---------image_feats')
        # print(image_feats.shape)

        text_output = self.Qformer.bert(
            text_ids,
            attention_mask=text_mask,
            return_dict=True,
        )
        # print('text_output')
        # print(text_output.shape)
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )#（bs, 256）
        
        query_tokens_itm = self.query_tokens.expand(text_ids.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_mask], dim=1)

        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        '''vl_embeddings: visual-text quality representation'''
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]#[bs, 32, 768]
        '''decoder'''
        x_1 = x_list[0]
        x_2 = x_list[1]
        x_3 = x_list[2]
        x_4 = x_list[3]
        x_5 = x_list[4]#【bs, 256, 1408】
        # x_5 = visual_feature[:, 1:, :]
        
        x_5 = self.refine(x_5) #[bs, 1408, 16, 16]
        # x = self.cross_attn1(x_5, vl_embeddings)#[bs, 1408, 16, 16] to restore spatial information

        x_4 = self.refine(x_4)
        x = torch.cat((x_5, x_4), dim=1) #[bs, 2816, 16, 16]
        x = self.deconv_layer1(x) #[bs, 512, 32, 32] 或者先deconv 再attn 再up？
        x = self.cross_attn2(x, vl_embeddings)#[bs, 512, 32, 32]
        # print(x.shape)

        x_3 = self.encode3(self.refine(x_3))#[bs, 512, 32, 32]
        # print(x_3.shape)
        x = torch.cat((x, x_3), dim=1)
        # print(x.shape)
        x = self.deconv_layer2(x)#[bs, 128, 64, 64]
        # x = self.cross_attn3(x, vl_embeddings)#[bs, 128, 64, 64]
        # print(x.shape)

        x_2 = self.encode2(self.refine(x_2))#[bs, 128, 64, 64]
        x = torch.cat((x, x_2), dim=1)
        x = self.deconv_layer3(x)#[bs, 32, 128, 128]
        # print(x.shape)

        x_1 = self.encode1(self.refine(x_1))#[bs, 32, 128, 128]
        x = torch.cat((x, x_1), dim=1)
        x = self.deconv_layer4(x)#[bs, 1, 256, 256]
        # print(x.shape)ß
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # x_resized = F.interpolate(x, size=(224, 224), mode='nearest')

        return x_resized


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)

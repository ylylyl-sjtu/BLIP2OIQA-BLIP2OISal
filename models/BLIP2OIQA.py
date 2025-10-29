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

import math
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from einops import rearrange

#@registry.register_model("blip2")
#@registry.register_model("blip2_feature_extractor")
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=7)
        self.fc1 = nn.Linear(100, 800)
        self.fc2 = nn.Linear(800, 800)
        # self.fc3 = nn.Linear(800, 1)

    def forward(self, input):
        x = input.view(-1, input.size(-3), input.size(-2), input.size(-1))

        x = self.conv(x)

        x1 = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        x2 = -F.max_pool2d(-x, (x.size(-2), x.size(-1)))

        h = torch.cat((x1, x2), 1)
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = F.dropout(h)
        h = F.relu(self.fc2(h))

        # q = self.fc3(h)

        return h
    
class CrossAttention(nn.Module):
    def __init__(self, emb_dim):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5


        # self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        # self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

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
    
class SelfAttention(nn.Module):
    def __init__(self, emb_dim):
        super(SelfAttention, self).__init__()
        self.emb_dim = emb_dim

        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)

        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, pad_mask=None):
        # [batch_szie, seq_len, emb_dim] = [3, 5, 512]

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        att_weights = torch.bmm(Q, K.transpose(1, 2))   # [batch_size, seq_len, emb_dim]
        att_weights = att_weights / math.sqrt(self.emb_dim)

        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        output = torch.bmm(att_weights, V)   # [batch_size, seq_len, emb_dim]
        output = self.fc(output)

        return output, att_weights

class BLIP2OIQA_qformer(Blip2Base):
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
        fix_rate = 0.6
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
        
        self.CrossAttention1 = CrossAttention(768)
        self.CrossAttention2 = CrossAttention(768)
        self.CrossAttention3 = CrossAttention(768)

        self.SelfAttention1 = SelfAttention(768)
        self.SelfAttention2 = SelfAttention(768)
        self.SelfAttention3 = SelfAttention(768)
        self.SelfAttention4 = SelfAttention(768)
        self.SelfAttention5 = SelfAttention(768)
        self.SelfAttention6 = SelfAttention(768)

        print('self.Qformer.config.hidden_size',self.Qformer.config.hidden_size)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block
    
    def forward(self, images, text_ids, text_mask):
        b,v,c,h,w = images.shape
        image_embeds = []
        image_atts = []
        query_tokens = []
        query_output = []
        image_feats = []
        image_feats_all = []
        sim_q2t = []
        sim_i2t = []
        sim_t2q = []
        sim_t2i = []
        text_ids = text_ids.repeat_interleave(6, dim=0)
        text_mask = text_mask.repeat_interleave(6, dim=0)
        images = rearrange(images, 'b v c h w -> (b v) c h w')
        # print("after rearrange: ", images.shape)
        images = images.to(self.device)
        image_embeds = self.ln_vision(self.visual_encoder(images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            images.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # print("shape of query tokens:", query_tokens.shape)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )
        # image_feats_perspective = rearrange(image_feats,"(b v) m n -> b v m n",b=b)
        # print("size of image_feats", image_feats.shape)
        # print("size of image_feats_perspective", image_feats_perspective.shape)
        text_output = self.Qformer.bert(
            text_ids,
            attention_mask=text_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(
            image_feats
        )
        # print("size of image_feats_all", image_feats_all.shape)
        # [batch_size*num_gpu, num_query_tokens, embed_dim]
        # print("shape of image_feats_all: ", image_feats_all.shape)
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]
        # print("shape of text_feats_all: ", image_feats_all.shape)
        # print("warning: the size of the last dimension of text_feat_all & shape of image_feats_all should be the same!")
        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]
        # print("shape of sim_q2t: ", sim_q2t.shape)
        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # print("shape of text_feat:", text_feat.shape)
        # print("shape of image_feats:", image_feats.shape)
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()
        # print("shape of sim_t2q: ", sim_t2q.shape)
        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        # print("shape of sim_t2i: ", sim_t2i.shape)
        # rank = dist.get_rank()
        rank = 0
        bs = images.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            images.device
        )
        # print("query_tokens_itm_before:", self.query_tokens1.shape) #[]
        # print("text_ids:", text_ids.shape)
        query_tokens_itm = self.query_tokens.expand(text_ids.shape[0], -1, -1)
        # print("query_tokens_itm:", query_tokens_itm.shape) #[]
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            images.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_mask], dim=1)
        # print("shape of attention_mask_all")
        # print(attention_mask_all.shape)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        # 'front': (90, 0, 0),
        # 'left': (90, 90, 0),
        # 'right': (90, -90, 0),
        # 'back': (90, 180, 0),
        # 'up': (90, 0, 90),
        # 'down': (90, 0, -90),
        adj_matrix = torch.tensor([
            [0, 1, 1, 0, 1, 1],  
            [1, 0, 0, 1, 1, 1],  
            [1, 0, 0, 1, 1, 1], 
            [0, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0] 
        ], dtype=torch.long).to(
            images.device
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] #[bs*views, 32, 768]
        vl_embeddings = rearrange(vl_embeddings, "(b v) m n -> b v m n", b=b)#[bs views 32 768]
        vl_embeddings_initial = vl_embeddings.mean(dim=1)
        #self-attention for every view first
        vl_embeddings_1, _ = self.SelfAttention1(vl_embeddings[:,0,:,:]) #[bs 32 768] view0
        vl_embeddings_2, _ = self.SelfAttention2(vl_embeddings[:,1,:,:])
        vl_embeddings_3, _ = self.SelfAttention3(vl_embeddings[:,2,:,:])
        vl_embeddings_4, _ = self.SelfAttention4(vl_embeddings[:,3,:,:])
        vl_embeddings_5, _ = self.SelfAttention5(vl_embeddings[:,4,:,:])
        vl_embeddings_6, _ = self.SelfAttention6(vl_embeddings[:,5,:,:])
        # print("vl_embeddings[:,0,:,:].shape")
        # print(vl_embeddings[:,0,:,:].shape)

        vl_embeddings = torch.stack([vl_embeddings_1, vl_embeddings_2, vl_embeddings_3, vl_embeddings_4, vl_embeddings_5, vl_embeddings_6], dim=1)
        # print("vl_embeddings.shape")
        # print(vl_embeddings.shape)
        #cross attention
        # vl_embeddings_1 = rearrange(vl_embeddings_1, "(b v) m n -> b v m n", b=b) #[bs, views, 32, 768]
        # vl_embeddings_2 = rearrange(vl_embeddings_2, "(b v) m n -> b v m n", b=b) #[bs, views, 32, 768]
        # vl_embeddings_3 = rearrange(vl_embeddings_3, "(b v) m n -> b v m n", b=b) #[bs, views, 32, 768]

        all_attention_results1 = torch.zeros(b, v, 4, vl_embeddings.shape[2], vl_embeddings.shape[3], device=images.device) # use an empty tensor to store the attention results
        all_attention_results2 = torch.zeros(b, v, 4, vl_embeddings.shape[2], vl_embeddings.shape[3], device=images.device) # use an empty tensor to store the attention results
        all_attention_results3 = torch.zeros(b, v, 4, vl_embeddings.shape[2], vl_embeddings.shape[3], device=images.device) # use an empty tensor to store the attention results
        for i in range(v):
            attn_idx = 0
            current_view = vl_embeddings[:, i, :, :]
            # current_view_2 = vl_embeddings_2[:, i, :, :]
            # current_view_3 = vl_embeddings_3[:, i, :, :]
            for j in range(v):
                if adj_matrix[i,j] == 1:
                    target_view = vl_embeddings[:, j, :, :]
                    # target_view_2 = vl_embeddings_2[:, j, :, :]
                    # target_view_3 = vl_embeddings_3[:, j, :, :]
                    # print("shape of target_view")
                    # print(target_view.shape)
                    attn_result1, _ = self.CrossAttention1(current_view, target_view)
                    attn_result2, _ = self.CrossAttention2(current_view, target_view)
                    attn_result3, _ = self.CrossAttention3(current_view, target_view)
                    all_attention_results1[:, i, attn_idx, :, :] = attn_result1
                    all_attention_results2[:, i, attn_idx, :, :] = attn_result2
                    all_attention_results3[:, i, attn_idx, :, :] = attn_result3
                    attn_idx += 1
                    # print("attn_idx")
                    # print(attn_idx)
        all_attn_result1 = rearrange(all_attention_results1, "b v o p q -> b (v o) p q")
        all_attn_result2 = rearrange(all_attention_results2, "b v o p q -> b (v o) p q")
        all_attn_result3 = rearrange(all_attention_results3, "b v o p q -> b (v o) p q")
        
        vl_embeddings_1 = all_attn_result1.mean(dim = 1)
        vl_embeddings_2 = all_attn_result2.mean(dim = 1)
        vl_embeddings_3 = all_attn_result3.mean(dim = 1)
        # print("shape of vl_embeddings after cross attention")
        # print(vl_embeddings_1.shape)

        # print("shape of vl_embeddings")
        # print(vl_embeddings.shape)
        itm_logit1 = self.quality1(vl_embeddings_1)
        itm_logit1 = itm_logit1[:, :, 1].mean(dim=1)
        # print("itm_logit1")
        # print(itm_logit1)
        # print(itm_logit1.shape)
        # print(itm_logit1)
        itm_logit2 = self.quality2(vl_embeddings_2)
        itm_logit2 = itm_logit2[:, :, 1].mean(dim=1)
        # itm_logit2 = rearrange(itm_logit2,"(b v) -> b v",b=b)
        # #itm_logit2 = itm_logit2.mean(dim=1)
        # itm_logit2 = self._attention_sum_2(image_feats_perspective, itm_logit2, adj_matrix, images.device)
        # itm_logit2 = torch.squeeze(itm_logit2)
        # print(itm_logit2)
        itm_logit3 = self.quality3(vl_embeddings_3)
        itm_logit3 = itm_logit3[:, :, 1].mean(dim=1)
        # itm_logit3 = rearrange(itm_logit3,"(b v) -> b v",b=b)
        # #itm_logit3 = itm_logit3.mean(dim=1)
        # itm_logit3 = self._attention_sum_3(image_feats_perspective, itm_logit3, adj_matrix, images.device)
        # itm_logit3 = torch.squeeze(itm_logit3)

        return itm_logit1, itm_logit2, itm_logit3, vl_embeddings_initial
        vl_output = self.itm_head2(vl_embeddings)
        logits = vl_output.mean(dim=1)
        #logits = vl_output
        print(logits.size())

        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(image.device)
        itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
        # print(bs)
        print(itm_labels.size())
        loss_itm = F.cross_entropy(logits, itm_labels)
        print(loss_itm.size())
        # for view in range(6):
        #     image_embeds.append(self.ln_vision(self.visual_encoder(images[view])))
        #     print('-------------------------------------')
        #     print('size of lists:')
        #     print(image_embeds.size())
        #     image_atts.append(torch.ones(image_embeds[view].size()[:-1], dtype=torch.long).to(
        #         image.device
        #     ))
        #     print(image_atts.size())
        #     query_tokens.append(self.query_tokens1.expand(image_embeds.shape[0], -1, -1))
        #     print(query_tokens.size())
        #     query_output.append(self.Qformer1.bert(
        #         query_embeds=query_tokens[view],
        #         encoder_hidden_states=image_embeds[view],
        #         encoder_attention_mask=image_atts[view],
        #         use_cache=True,
        #         return_dict=True,
        #     ))
        #     print(query_output.size())

        #     image_feats.append(F.normalize(
        #         self.vision_proj(query_output[i].last_hidden_state), dim=-1
        #     ))
        # #text_output same for every view
        # text_output = self.Qformer1.bert(
        #     text_ids,
        #     attention_mask=text_mask,
        #     return_dict=True,
        # )
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )

        # ###============== Image-text Contrastive ===================###
        # text_feat_all = concat_all_gather(text_feat)
        # # [batch_size*num_gpu, embed_dim]
        # for view in range(6):
        #     image_feats_all.append(concat_all_gather(
        #         image_feats[view]
        #     ))
        #     # [batch_size*num_gpu, num_query_tokens, embed_dim]

        #     sim_q2t.append(torch.matmul(
        #         image_feats[view].unsqueeze(1), text_feat_all.unsqueeze(-1)
        #     ).squeeze())

        #     sim_i2t_temp, _ = sim_q2t[view].max(-1)
        #     sim_i2t.append(sim_i2t_temp)
        #     sim_i2t[view] = sim_i2t[view] / self.temp
        # # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        #     sim_t2q.append(torch.matmul(
        #         text_feat.unsqueeze(1).unsqueeze(1), image_feats_all[view].permute(0, 2, 1)
        #     ).squeeze())

        # # text-image similarity: aggregate across all query tokens
        #     sim_t2i_temp, _ = sim_t2q[view].max(-1)
        #     sim_t2i.append(sim_t2i_temp)
        #     sim_t2i[view] = sim_t2i[view] / self.temp  # [batch_size, batch_size*num_gpu]

        # # rank = dist.get_rank()
        # rank = 0
        # bs = image.size(0)
        # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
        #     image.device
        # )


        # query_tokens_itm = self.query_tokens1.expand(text_ids.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_mask], dim=1)
        # output_itm = []
        # vl_embeddings = []
        # itm_logit1 = []
        # itm_logit2 = []
        # itm_logit3 = []
        # for view in range(6):
        #     output_itm.append(self.Qformer1.bert(
        #         text_ids,
        #         query_embeds=query_tokens_itm,
        #         attention_mask=attention_mask_all,
        #         encoder_hidden_states=image_embeds[view],
        #         encoder_attention_mask=image_atts[view],
        #         return_dict=True,
        #     ))
        #     vl_embeddings.append(output_itm[view].last_hidden_state[:, : query_tokens_itm.size(1), :])
        #     itm_logit1_temp = self.quality1(vl_embeddings[view])
        #     itm_logit1.append(itm_logit1_temp[:, :, 1].mean(dim=1))
        #     itm_logit1_final = itm_logit1.mean()
        #     print('----------------itm_logits_1')
        #     print(itm_logit1_temp)
        #     print(itm_logit1_final)
        #     itm_logit2_temp = self.quality2(vl_embeddings[view])
        #     itm_logit2.append(itm_logit2_temp[:, :, 1].mean(dim=1))
        #     itm_logit2_final = itm_logit2.mean()
        #     print('----------------itm_logits_2')
        #     print(itm_logit2_temp)
        #     print(itm_logit2_final)
        #     itm_logit3_temp = self.quality3(vl_embeddings[view])
        #     itm_logit3.append(itm_logit3_temp[:, :, 1].mean(dim=1))
        #     itm_logit3_final = itm_logit3.mean()
        #     print('----------------itm_logits_3')
        #     print(itm_logit3_temp)
        #     print(itm_logit3_final)
            
            
        # #itm_logit = self.itm_head(vl_embeddings)
        # #itm_logit = itm_logit[:, :, 1].mean(dim=1)
        # #print(itm_logit)
        # return itm_logit1_final, itm_logit2_final, itm_logit3_final, vl_embeddings

        # '''unreachable'''
        # vl_output = self.itm_head2(vl_embeddings)
        # logits = vl_output.mean(dim=1)
        # #logits = vl_output
        # print(logits.size())

        # itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
        # # print(bs)
        # print(itm_labels.size())
        # loss_itm = F.cross_entropy(logits, itm_labels)
        # print(loss_itm.size())

        # return loss_itm
        

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

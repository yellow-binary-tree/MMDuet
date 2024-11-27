#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
import copy
import random
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from ..modeling_live import build_live, LiveMixin
from ..configuration_live import VideoHeadLiveConfigMixin

from transformers.utils import logging
logger = logging.get_logger(__name__)


class VideoHeadLiveLlavaQwenConfig(Qwen2Config, VideoHeadLiveConfigMixin):
    def __init__(self, video_pooling_stride=4, video_head_stop_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.video_pooling_stride = video_pooling_stride
        self.video_head_stop_grad = video_head_stop_grad


@dataclass
class VideoHeadCausalLMOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    lm_loss: Optional[torch.FloatTensor] = None
    video_loss: Optional[torch.FloatTensor] = None
    informative_logits: Optional[torch.FloatTensor] = None
    relevance_logits: Optional[torch.FloatTensor] = None

class VideoHeadLlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = VideoHeadLiveLlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(VideoHeadLlavaQwenModel, self).__init__(config)


class VideoHeadLiveLlavaQwenForCausalLM(Qwen2ForCausalLM, LiveMixin):
    config_class = VideoHeadLiveLlavaQwenConfig

    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = VideoHeadLlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.informative_head = nn.Linear(config.hidden_size, 2, bias=False)
        self.relevance_head = nn.Linear(config.hidden_size, 2, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.vision_encoder = self.get_vision_tower()
        self.lm_loss_weight = 1
        self.video_loss_weight = 1
        print(f"using lm_loss_weight: {self.lm_loss_weight}, video_loss_weight: {self.video_loss_weight} for training")

    def get_model(self):
        return self.model

    def connector(self, frames):
        return self.get_model().mm_projector(frames)

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def vision_encode(self, vision_tower, frames):
        frame_features = vision_tower(frames)
        return frame_features

    def post_projector_pooling(self, image_feature):
        stride = self.config.video_pooling_stride
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim).contiguous()
        return image_feature

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        informative_labels: Optional[torch.LongTensor] = None,
        relevance_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        frames: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        model_outputs = copy.copy(outputs)
        hidden_states = outputs[0]
        outputs = outputs[1:]
        logits = self.lm_head(hidden_states).float()
        if self.config.video_head_stop_grad:
            hidden_states_no_grad = hidden_states.detach()
        else:
            hidden_states_no_grad = hidden_states
        informative_logits = self.informative_head(hidden_states_no_grad).float()
        relevance_logits = self.relevance_head(hidden_states_no_grad).float()

        # NOTE: all labels used here are already shifted in data collator
        loss_fct = CrossEntropyLoss()
        loss = 0.

        if labels is not None:
            if not(labels != -100).any():
                labels[:, 0] = input_ids[:, 1]      # make sure lm_loss is calculated for every example, or the deepspeed training process will hang
            lm_loss = loss_fct(logits.flatten(0, 1), labels.flatten())
            if not return_dict:
                outputs = (logits,) + outputs + (loss,)
        else:
            lm_loss = 0.

        # merge the 2 labels together, so this loss must be calculated as all training examples contains either informative_label or relevance_label.
        # otherwise the deepspeed training process will hang
        if informative_labels is not None and relevance_labels is not None:
            video_labels = torch.cat([informative_labels, relevance_labels], dim=0)
            video_logits = torch.cat([informative_logits, relevance_logits], dim=0)
            if not (video_labels != -100).any():
                video_labels[:, 0] = 0      # make sure video_loss is calculated for every example, or the deepspeed training process will hang
            video_loss = loss_fct(video_logits.flatten(0, 1), video_labels.flatten())
            if not return_dict:
                outputs = outputs + (video_loss,)
        else:
            video_loss = 0.

        loss = lm_loss * self.lm_loss_weight + video_loss * self.video_loss_weight

        if not return_dict:
            outputs = (loss,) + outputs
            return outputs

        return VideoHeadCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            lm_loss=lm_loss,
            video_loss=video_loss,
            informative_logits=informative_logits,
            relevance_logits=relevance_logits,
        )

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        frames: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        '''
        The original generate function of LLaVA.
        '''
        logger.warning('You are calling the generate function of LLaVA, which is deprecated for Live Video models. Please use a LiveInfer class for inference.')
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().generate(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        return outputs


def build_video_head_live_llava_qwen(**kwargs):
    model, tokenizer = build_live(config_class=VideoHeadLiveLlavaQwenConfig, model_class=VideoHeadLiveLlavaQwenForCausalLM, **kwargs)
    # freeze vit
    print('freezing ViT')
    for param in model.get_vision_tower().parameters():
        param.requires_grad = False
    return model, tokenizer

if __name__ == '__main__':
    from transformers import HfArgumentParser
    from models.arguments_live import LiveTrainingArguments
    args, = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
    args.llm_pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    print(args.to_dict())
    model, tokenizer = build_video_head_live_llava_qwen(is_training=True, **args.to_dict())
    print(model.config, tokenizer)

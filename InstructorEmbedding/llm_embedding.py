# This script is based on the modifications from https://github.com/UKPLab/sentence-transformers
import torch
import os
import json
import importlib
import numpy as np
from tqdm.autonotebook import trange
from torch import Tensor, device
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    SchedulerType, 
    default_data_collator, 
    get_scheduler, 
    LlamaPreTrainedModel,
    LlamaModel
)
from collections import OrderedDict
from torch import nn
from typing import List, Optional

from transformers.utils import get_full_repo_name
from peft import LoraConfig, TaskType, get_peft_model

def batch_to_device(batch, target_device: device):
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class LlamaForSequenceEmbedding(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.normalize = True

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # labels: Optional[torch.Tensor] = None, ## added
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # **kwargs
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("hello")
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            # labels=labels, ## added
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # **kwargs
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        embeddings = self.last_token_pool(last_hidden_state, attention_mask)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, queries: str) -> str:
    return [f'Instruct: {task_description}\nQuery: {query}' for query in queries]

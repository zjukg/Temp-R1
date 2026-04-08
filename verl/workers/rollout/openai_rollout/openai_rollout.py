# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
OpenAI API Rollout for evaluation purposes
"""
import os
import torch
from typing import List, Dict, Any
from omegaconf import DictConfig
from openai import OpenAI

from verl import DataProto
from verl.workers.rollout.base import BaseRollout


def _remove_padding(token_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """移除左侧padding"""
    non_pad_indices = torch.nonzero(token_ids != pad_token_id, as_tuple=True)[0]
    if len(non_pad_indices) == 0:
        return token_ids
    first_non_pad = non_pad_indices[0]
    return token_ids[first_non_pad:]


class OpenAIRollout(BaseRollout):
    """
    OpenAI API Rollout，模仿 vLLMRollout 的接口
    用于通过 OpenAI API 验证模型性能
    """
    
    def __init__(self, config: DictConfig, tokenizer, **kwargs):
        """
        Args:
            config: DictConfig，包含 API 配置
            tokenizer: HuggingFace tokenizer
            **kwargs: 其他参数
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # 从配置或环境变量读取 API 配置
        api_key = config.get('api_key', os.getenv('OPENAI_API_KEY'))
        if not api_key:
            raise ValueError("OpenAI API key not found. Set it in config or OPENAI_API_KEY env var.")
        
        base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.model_name = config.get('model_name', 'gpt-4')
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        print(f"[OpenAIRollout] Initialized with model: {self.model_name}")
        print(f"[OpenAIRollout] Base URL: {base_url}")
    
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        核心方法：生成序列
        
        Args:
            prompts: DataProto 对象，包含:
                - batch['input_ids']: tensor of prompt token ids (batch_size, seq_len)
                - meta_info: dict with generation config
        
        Returns:
            DataProto: 包含生成的 responses
        """
        batch_size = len(prompts)
        prompt_token_ids = prompts.batch['input_ids']  # ✅ 改为 'input_ids'
        
        # 1. 解码 token ids 为文本
        prompt_texts = []
        for i in range(batch_size):
            tokens = prompt_token_ids[i]
            # 移除 padding
            valid_tokens = _remove_padding(tokens, self.tokenizer.pad_token_id)
            text = self.tokenizer.decode(valid_tokens, skip_special_tokens=False)
            prompt_texts.append(text)
        
        # 2. 从 meta_info 读取生成参数
        meta_info = prompts.meta_info
        do_sample = meta_info.get('do_sample', False)
        max_new_tokens = meta_info.get('max_new_tokens', 1000)
        temperature = 0.0 if not do_sample else meta_info.get('temperature', 1.0)
        top_p = meta_info.get('top_p', 1.0) if do_sample else 1.0
        
        print(f"[OpenAIRollout] Generating {batch_size} sequences with temperature={temperature}")
        
        # 3. 调用 OpenAI API（批量处理）
        generated_texts = []
        for idx, prompt_text in enumerate(prompt_texts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=1,
                )
                generated_text = response.choices[0].message.content
                generated_texts.append(generated_text)
                
                if idx < 2:  # 打印前两个样本
                    print(f"\n[OpenAIRollout Sample {idx}]")
                    print(f"Prompt (first 200 chars): {prompt_text[:200]}...")
                    print(f"Response (first 200 chars): {generated_text[:200]}...")
                
            except Exception as e:
                print(f"[ERROR] OpenAI API call failed for sample {idx}: {e}")
                generated_texts.append("")  # 失败时返回空字符串
        
        # 4. 编码回 token ids
        response_token_ids_list = []
        for text in generated_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            response_token_ids_list.append(torch.tensor(tokens, dtype=torch.long))

        # 5. Padding 到相同长度
        if len(response_token_ids_list) == 0:
            max_len = 1
        else:
            max_len = max(len(tokens) for tokens in response_token_ids_list)

        # ✅ 获取输入 tensor 的设备
        device = prompt_token_ids.device

        padded_responses = []
        attention_masks = []

        for tokens in response_token_ids_list:
            # 右侧 padding - ✅ 在正确的设备上创建
            pad_length = max_len - len(tokens)
            padded = torch.cat([
                tokens.to(device),  # ✅ 移动到正确的设备
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long, device=device)  # ✅ 直接在设备上创建
            ])
            padded_responses.append(padded)
            
            # attention mask: 1 for valid tokens, 0 for padding - ✅ 在正确的设备上创建
            mask = torch.cat([
                torch.ones(len(tokens), dtype=torch.long, device=device),  # ✅ 直接在设备上创建
                torch.zeros(pad_length, dtype=torch.long, device=device)  # ✅ 直接在设备上创建
            ])
            attention_masks.append(mask)

        response_tensor = torch.stack(padded_responses)  # (batch_size, max_len)
        attention_mask_tensor = torch.stack(attention_masks)  # (batch_size, max_len)

        # 6. 构造返回的 DataProto（模仿 vLLM 格式）
        # 使用 'input_ids' 作为键名（与输入保持一致）
        output_batch = {
            'input_ids': prompt_token_ids,
            'responses': response_tensor,
            'attention_mask': torch.cat([
                prompts.batch.get('attention_mask', torch.ones_like(prompt_token_ids)),
                attention_mask_tensor
            ], dim=1),
        }
        
        # OpenAI API 不提供 log_probs，设为 None
        # 如果需要，可以设为 0
        if 'old_log_probs' in prompts.batch:
            output_batch['old_log_probs'] = None
        
        # 复制 non_tensor_batch
        output = DataProto(
            batch=output_batch,
            non_tensor_batch=prompts.non_tensor_batch.copy() if prompts.non_tensor_batch else {}
        )
        
        print(f"[OpenAIRollout] Generated responses with shape: {response_tensor.shape}")
        
        return output
    
    def update_weight(self, *args, **kwargs):
        """API 模型不需要更新权重"""
        print("[OpenAIRollout] update_weight called, but API models don't need weight updates")
        pass
    
    def generate_sequences_with_logprobs(self, prompts: DataProto) -> DataProto:
        """
        OpenAI API 不支持 logprobs（或需要额外配置）
        目前简化处理，直接调用 generate_sequences
        """
        print("[OpenAIRollout] generate_sequences_with_logprobs called, using standard generation")
        return self.generate_sequences(prompts)
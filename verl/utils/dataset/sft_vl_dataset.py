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

import copy
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    position_ids = torch.stack([x["position_ids"] for x in batch])
    loss_mask = torch.stack([x["loss_mask"] for x in batch])
    
    # pad multi_modal_inputs["pixel_values"]
    pixel_values_list = [x["multi_modal_inputs"]["pixel_values"] for x in batch]
    image_grid_thw_list = [x["multi_modal_inputs"]["image_grid_thw"][0] for x in batch]
    max_len = max([pv.shape[0] for pv in pixel_values_list])
    embedding_dim = pixel_values_list[0].shape[1]
    
    padded_pixel_values = torch.zeros(len(batch), max_len, embedding_dim)
    for i, pv in enumerate(pixel_values_list):
        padded_pixel_values[i, :pv.shape[0], :] = pv

    multi_modal_inputs = {
        "pixel_values": padded_pixel_values,
        "image_grid_thw": torch.stack(image_grid_thw_list)  
        }

    raw_input = [x["raw_input"] for x in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "multi_modal_inputs": multi_modal_inputs,
        "raw_input": raw_input,
    }



class SFTVLDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", False)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                if self.config.get("pure_text", False):
                    if "You are a helpful assistant." in content:
                        content = "You are a helpful assistant.\n"
                    if "Think first, call **image_zoom_in_tool** if needed" in content:
                        content = content.split("Think first")[0] + "Think first, then answer. Format strictly as:  <think>...</think>  <answer>...</answer> "
                if self.config.get("sft_train", False):
                    if "You are a helpful assistant." in content:
                        content = "You are a helpful assistant.\n"
                    if "Think first, call **image_zoom_in_tool** if needed" in content:
                        content = content.split("Think first")[0]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        response = row_dict["reward_model"]["ground_truth"]
        response_chat_str = response + self.tokenizer.eos_token
        response_ids_output = self.tokenizer([response_chat_str], return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        response_length = response_ids.shape[0]
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_raw_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}
            origin_multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                origin_images = [process_raw_image(image) for image in row_dict.get(self.image_key)]
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images
                origin_multi_modal_data["image"] = origin_images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict['origin_multi_modal_data'] = origin_multi_modal_data
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        # print('aaa', item, input_ids.shape, attention_mask.shape)

        prompt_length = input_ids[0].shape[0]
        sft_input_ids = torch.cat((input_ids[0], response_ids), dim=-1)
        sft_attention_mask = torch.cat((attention_mask[0], response_attention_mask), dim=-1)
        sft_raw_input = raw_prompt + response_chat_str
        # print('bbb', item, sft_input_ids.shape, sft_attention_mask.shape, prompt_length)

        sft_input_ids, sft_attention_mask = verl_F.postprocess_data(
            input_ids=sft_input_ids.unsqueeze(0),
            attention_mask=sft_attention_mask.unsqueeze(0),
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        sft_input_ids, sft_attention_mask = sft_input_ids[0], sft_attention_mask[0]
        # print('xxx', item, sft_input_ids.shape, sft_attention_mask.shape)

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            sft_position_ids = get_rope_index(
                    self.processor,
                    input_ids=sft_input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=sft_attention_mask,
                )  # (1, 3, seq_len)

        else:
            sft_position_ids = compute_position_id_with_mask(sft_attention_mask)

        sft_loss_mask = sft_attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            sft_loss_mask[: min(prompt_length, sft_loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        sft_loss_mask[min(prompt_length + response_length, sft_loss_mask.size(0)) - 1] = 0

        # print('yyy', item, sft_input_ids.shape, sft_attention_mask.shape, sft_position_ids.shape, sft_loss_mask.shape)

        # row_dict["input_ids"] = input_ids[0]
        # row_dict["attention_mask"] = attention_mask[0]
        # row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        # return row_dict
        # if row_dict["multi_modal_inputs"]["pixel_values"].shape[0]>= self.max_prompt_length:
        #     count = (input_ids == 151655).sum().item()
        #     print('aaaaaaaa', item, row_dict["multi_modal_inputs"], row_dict["multi_modal_inputs"]["pixel_values"].shape, count)
        
        # pixel_values = row_dict["multi_modal_inputs"]["pixel_values"]
        # max_len = self.max_prompt_length  # 可以固定，比如 6912
        # pad_len = max_len - pixel_values.shape[0]

        # if pad_len > 0:
        #     pad_tensor = torch.zeros(pad_len, pixel_values.shape[1], dtype=pixel_values.dtype)
        #     pixel_values = torch.cat([pixel_values, pad_tensor], dim=0)
        # row_dict["multi_modal_inputs"]["pixel_values"] = pixel_values
        
        # print('cccccccccc', item, row_dict["multi_modal_inputs"])
        # print('ddddddddddd', item, row_dict["multi_modal_inputs"]["pixel_values"].shape)
        return {
            "input_ids": sft_input_ids,
            "attention_mask": sft_attention_mask,
            "position_ids": sft_position_ids,
            "loss_mask": sft_loss_mask,
            "multi_modal_inputs": row_dict["multi_modal_inputs"],
            "raw_input": sft_raw_input,
        }

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

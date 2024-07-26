# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from unittest.mock import patch

from transformers import LlamaTokenizer

EXPECTED_RESULTS={
    "meta-llama/Llama-2-7b-hf":{
        "example_1": "[INST] Who made Berlin [/INST] dunno",
        "example_2": "[INST] Quiero preparar una pizza de pepperoni, puedes darme los pasos para hacerla? [/INST] Claro!",
    },
    "meta-llama/Meta-Llama-3-8B":{
        "example_1": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWho made Berlin<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ndunno<|eot_id|><|end_of_text|>",
        "example_2": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow to start learning guitar and become a master at it?",
    },
}

def check_padded_entry(batch, tokenizer):
    seq_len = sum(batch["attention_mask"][0])
    assert seq_len < len(batch["attention_mask"][0])

    if tokenizer.vocab_size >= 128000:
        END_OF_TEXT_ID = 128009
    else:
        END_OF_TEXT_ID = tokenizer.eos_token_id

    assert batch["labels"][0][0] == -100
    assert batch["labels"][0][seq_len-1] == END_OF_TEXT_ID
    assert batch["labels"][0][-1] == -100
    assert batch["input_ids"][0][0] == tokenizer.bos_token_id
    assert batch["input_ids"][0][-1] == tokenizer.eos_token_id


@pytest.mark.skip(reason="Flakey due to random dataset order @todo fix order")
@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.AutoTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_custom_dataset(step_lr, optimizer, get_model, tokenizer, train, mocker, setup_tokenizer, llama_version):
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)

    skip_special_tokens = llama_version == "meta-llama/Llama-2-7b-hf"
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]

    kwargs = {
        "dataset": "custom_dataset",
        "model_name": llama_version,
        "custom_dataset.file": "recipes/finetuning/datasets/custom_dataset.py",
        "custom_dataset.train_split": "validation",
        "batch_size_training": 2,
        "val_batch_size": 4,
        "use_peft": False,
        "batching_strategy": "padding"
        }

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    tokenizer = args[3]

    assert len(train_dataloader) == 1120
    assert len(eval_dataloader) == 1120 //2

    it = iter(eval_dataloader)
    batch = next(it)
    STRING = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=skip_special_tokens)
    assert STRING.startswith(EXPECTED_RESULTS[llama_version]["example_1"])

    assert batch["input_ids"].size(0) == 4
    assert set(("labels", "input_ids", "attention_mask")) == set(batch.keys())

    check_padded_entry(batch, tokenizer)

    it = iter(train_dataloader)
    next(it)

    batch = next(it)
    STRING = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=skip_special_tokens)
    assert STRING.startswith(EXPECTED_RESULTS[llama_version]["example_2"])

    assert batch["input_ids"].size(0) == 2
    assert set(("labels", "input_ids", "attention_mask")) == set(batch.keys())

    check_padded_entry(batch, tokenizer)



@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.AutoTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_unknown_dataset_error(step_lr, optimizer, tokenizer, get_model, train, mocker, llama_version):
    from llama_recipes.finetuning import main

    tokenizer.return_value = mocker.MagicMock(side_effect=lambda x: {"input_ids":[len(x)*[0,]], "attention_mask": [len(x)*[0,]]})
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]

    kwargs = {
        "dataset": "custom_dataset",
        "custom_dataset.file": "recipes/finetuning/datasets/custom_dataset.py:get_unknown_dataset",
        "batch_size_training": 1,
        "use_peft": False,
        }
    with pytest.raises(AttributeError):
        main(**kwargs)

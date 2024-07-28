# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from unittest.mock import patch


EXPECTED_RESULTS = {
    "meta-llama/Llama-2-7b-hf":{
        "label": 1152,
        "pos": 31,
    },
    "meta-llama/Meta-Llama-3-8B":{
        "label": 40,
        "pos": 26,
    },
}

@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.AutoTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_grammar_dataset(step_lr, optimizer, get_model, tokenizer, train, setup_tokenizer, llama_version):
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]

    BATCH_SIZE = 8
    kwargs = {
        "model_name": llama_version,
        "batch_size_training": BATCH_SIZE,
        "val_batch_size": 1,
        "use_peft": False,
        "dataset": "grammar_dataset",
        "batching_strategy": "padding",
        }

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]

    VAL_SAMPLES = 2988
    TRAIN_SAMPLES = 13016

    assert len(train_dataloader) == TRAIN_SAMPLES // BATCH_SIZE
    assert len(eval_dataloader) == VAL_SAMPLES

    batch = next(iter(train_dataloader))

    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert batch["labels"][0][EXPECTED_RESULTS[llama_version]["pos"]-1] == -100
    assert batch["labels"][0][EXPECTED_RESULTS[llama_version]["pos"]] == EXPECTED_RESULTS[llama_version]["label"]

    token = args[3]
    assert batch["input_ids"][0][0] == token.bos_token_id
    assert batch["labels"][0][-1] == token.eos_token_id
    assert batch["input_ids"][0][-1] == token.eos_token_id

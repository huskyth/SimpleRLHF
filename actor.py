import torch
import random

from util import TokenizerUtil
from datasets import load_dataset
from transformers import default_data_collator
from transformers import AutoModelForCausalLM
import lora
from transformers import get_scheduler
from accelerate import Accelerator


def tokenizer_tester():
    tokenizer = TokenizerUtil()

    input_ids, attention_mask = tokenizer.encode('how are you', max_length=4)

    input_ids, attention_mask, tokenizer.decode(input_ids)


dataset = load_dataset('json', data_files='dataset/train.json', split='train')

# 2,4,4切分,取第0部分
dataset = dataset.select(range(15000))


def f(data):
    # 随机生成两种回答
    if random.random() > 0.5:
        data['chosen'] = data['chosen'].swapcase()
    data = data['prompt'] + data['chosen']

    input_ids, attention_mask = tokenizer.encode(data)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()
    }


dataset = dataset.map(f, remove_columns=dataset.column_names)

loader = torch.utils.data.DataLoader(dataset,
                                     collate_fn=default_data_collator,
                                     batch_size=2,
                                     shuffle=True,
                                     drop_last=True)

len(loader), next(iter(loader))

model_actor = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b')

lora.insert(model_actor)
lora.count_params(model_actor)


def f():
    params = []
    params_lora = []
    for name, param in model_actor.named_parameters():
        if not param.requires_grad:
            continue

        if 'lora_A' in name or 'lora_B' in name:
            params_lora.append(param)
            continue

        params.append(param)

    return [{
        'params': params,
        'weight_decay': 0.0,
    }, {
        'params': params_lora,
        'weight_decay': 0.0,
        'lr': 5e-4
    }]


tokenizer = TokenizerUtil()
optimizer = torch.optim.Adam(f(), lr=1e-3, betas=(0.9, 0.95))

scheduler = get_scheduler(name='cosine',
                          optimizer=optimizer,
                          num_warmup_steps=0,
                          num_training_steps=100)

accelerator = Accelerator(gradient_accumulation_steps=64,
                          mixed_precision='fp16')

model_actor, loader, optimizer, scheduler = accelerator.prepare(
    model_actor, loader, optimizer, scheduler)

model_actor.train()

for i, data in enumerate(loader):
    with accelerator.accumulate(model_actor):
        out = model_actor(**data)
        accelerator.backward(out.loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(
                [i for i in model_actor.parameters() if i.requires_grad], 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if (i + 1) % 100 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(i, len(loader), out.loss.item(), lr)

        logits = out.logits[0].argmax(1)
        print(tokenizer.decode(logits))

    if i == 2000:
        break

lora.merge(model_actor)
model_actor.save_pretrained('model/actor')

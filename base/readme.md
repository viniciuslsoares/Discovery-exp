# Test Comands

## Pretrain

- python cli_pretrain.py --input_size 32 --dataset_name f3_N --batch_size 8 --num_epochs 5 --repetition 0


### Obs

``pretrain.py``
- mudar linha `        devices=[0]`para mais de 1 gpus
- descomentar linha `        # strategy=DDPStrategy(static_graph=True)` para mais de 1 gpu


## Finetune

- python cli_finetune.py --pretrain_data seam_ai_N --finetune_data seam_ai_N --num_epochs 5 --batch_size 8 --repetition 0 --cap 1.0

### Obs

- Por padrão ele busca o ckpt gerado pelo `pretrain.py`. Pode passar um path `.ckpt` diretamente pela cli
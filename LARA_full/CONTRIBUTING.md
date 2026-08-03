# Contributing

Before submitting changes:

```bash
python -m compileall -q lara_full moto scripts
bash -n run_tokenizer.sh run_lara_full_pretrain.sh run_libero_10.sh
git diff --check
```

Do not commit datasets, checkpoints, training outputs, private mount paths, or
credentials. New configuration files should use environment-variable
placeholders and document any required assets in `.env.example`.

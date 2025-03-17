# prime - decentralized training at scale
this is a fork of prime that remove the distributed part just to debug the core stuff

```
curl -sSL https://raw.githubusercontent.com/samsja/prime/main/install.sh | bash
source $HOME/.local/bin/env

```


run debug

```bash
uv  run torchrun --nproc_per_node=2 train_fsdp.py  @ configs/debug/normal.toml
```


run 1b 

```bash
uv  run torchrun --nproc_per_node=8 train_fsdp.py @ configs/1B/H100.toml
```






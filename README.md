# prime - decentralized training at scale
this is a fork of prime that remove the distributed part just to debug the core stuff

```
curl -sSL https://raw.githubusercontent.com/samsja/prime/main/install.sh | bash
```


run debug

```bash
uv  run torchrun --nproc_per_node=2 train_fsdp.py  @ configs/debug/normal.toml
```
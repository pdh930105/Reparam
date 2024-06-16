### Timm Reparameterize model latency test


```
# 1 setting docker (x86_64 server => gpu, raspberrypi 5 => raspberry5)
cd docker/gpu
sh compose.sh # start docker
# 2 run code
cd /workspace/Reparam # in docker CLI
python rep_latency_test.py --model {fastvit, repvgg, repghost, mobileone, repvit}
```

| 모델 | 구분 | condition_id | 배치 | 입력/생성 또는 해상도 | 기존 peak reserved (MiB) | 기존 실행 시간 (s) | 선정 이유 |
|---|---|---|---|---|---|---|---|
| opt-125m | light | `opt-125m__b1__in128__new128` | 1 | 128/128 | 330.0 | 14.1 | batch 1, smallest peak reserved among batch-1 ok conditions |
| opt-125m | medium | `opt-125m__b32__in256__new128` | 32 | 256/128 | 1,512.0 | 16.0 | peak reserved closest to geometric mean of light/heavy (1848 MiB); no retry; wall <= 600s |
| opt-125m | heavy | `opt-125m__b128__in512__new256` | 128 | 512/256 | 10,352.0 | 131.8 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| TinyLlama-1.1B-Chat-v1.0 | light | `tinyllama-1.1b-chat__b1__in128__new128` | 1 | 128/128 | 2,264.0 | 43.3 | batch 1, smallest peak reserved among batch-1 ok conditions |
| TinyLlama-1.1B-Chat-v1.0 | medium | `tinyllama-1.1b-chat__b32__in256__new512` | 32 | 256/512 | 3,782.0 | 243.5 | peak reserved closest to geometric mean of light/heavy (4245 MiB); no retry; wall <= 600s |
| TinyLlama-1.1B-Chat-v1.0 | heavy | `tinyllama-1.1b-chat__b128__in512__new128` | 128 | 512/128 | 7,958.0 | 274.3 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| Qwen2.5-1.5B-Instruct | light | `qwen2.5-1.5b-instruct__b1__in128__new128` | 1 | 128/128 | 3,452.0 | 57.8 | batch 1, smallest peak reserved among batch-1 ok conditions |
| Qwen2.5-1.5B-Instruct | medium | `qwen2.5-1.5b-instruct__b128__in128__new256` | 128 | 128/256 | 5,492.0 | 244.3 | peak reserved closest to geometric mean of light/heavy (5936 MiB); no retry; wall <= 600s |
| Qwen2.5-1.5B-Instruct | heavy | `qwen2.5-1.5b-instruct__b128__in512__new256` | 128 | 512/256 | 10,206.0 | 521.1 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| gpt2 | light | `gpt2__b1__in128__new128` | 1 | 128/128 | 488.0 | 17.1 | batch 1, smallest peak reserved among batch-1 ok conditions |
| gpt2 | medium | `gpt2__b32__in128__new128` | 32 | 128/128 | 1,086.0 | 15.6 | peak reserved closest to geometric mean of light/heavy (1319 MiB); no retry; wall <= 600s |
| gpt2 | heavy | `gpt2__b128__in128__new128` | 128 | 128/128 | 3,564.0 | 32.2 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| gpt2-medium | light | `gpt2-medium__b1__in512__new128` | 1 | 512/128 | 1,010.0 | 21.6 | batch 1, smallest peak reserved among batch-1 ok conditions |
| gpt2-medium | medium | `gpt2-medium__b16__in128__new512` | 16 | 128/512 | 3,106.0 | 84.6 | peak reserved closest to geometric mean of light/heavy (3116 MiB); no retry; wall <= 600s |
| gpt2-medium | heavy | `gpt2-medium__b32__in128__new512` | 32 | 128/512 | 9,612.0 | 132.7 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| gpt2-xl | light | `gpt2-xl__b1__in512__new128` | 1 | 512/128 | 3,488.0 | 38.5 | batch 1, smallest peak reserved among batch-1 ok conditions |
| gpt2-xl | medium | `gpt2-xl__b8__in128__new512` | 8 | 128/512 | 5,814.0 | 158.9 | peak reserved closest to geometric mean of light/heavy (5657 MiB); no retry; wall <= 600s |
| gpt2-xl | heavy | `gpt2-xl__b16__in512__new128` | 16 | 512/128 | 9,174.0 | 94.7 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| resnet-18 | light | `resnet-18__b1__r128` | 1 | r128 | 50.0 | 6.2 | batch 1, smallest peak reserved among batch-1 ok conditions |
| resnet-18 | medium | `resnet-18__b16__r512` | 16 | r512 | 364.0 | 5.9 | peak reserved closest to geometric mean of light/heavy (357 MiB); no retry; wall <= 600s |
| resnet-18 | heavy | `resnet-18__b128__r512` | 128 | r512 | 2,548.0 | 9.1 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| resnet-50 | light | `resnet-50__b1__r128` | 1 | r128 | 84.0 | 5.9 | batch 1, smallest peak reserved among batch-1 ok conditions |
| resnet-50 | medium | `resnet-50__b16__r512` | 16 | r512 | 846.0 | 6.4 | peak reserved closest to geometric mean of light/heavy (720 MiB); no retry; wall <= 600s |
| resnet-50 | heavy | `resnet-50__b128__r512` | 128 | r512 | 6,168.0 | 15.3 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| mobilenet_v2_1.0_224 | light | `mobilenet-v2-1.0-224__b1__r128` | 1 | r128 | 28.0 | 5.7 | batch 1, smallest peak reserved among batch-1 ok conditions |
| mobilenet_v2_1.0_224 | medium | `mobilenet-v2-1.0-224__b32__r256` | 32 | r256 | 430.0 | 5.9 | peak reserved closest to geometric mean of light/heavy (429 MiB); no retry; wall <= 600s |
| mobilenet_v2_1.0_224 | heavy | `mobilenet-v2-1.0-224__b128__r512` | 128 | r512 | 6,580.0 | 10.4 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| vit-large-patch16-224 | light | `vit-large-patch16-224__b1__r224` | 1 | r224 | 624.0 | 14.2 | batch 1, smallest peak reserved among batch-1 ok conditions |
| vit-large-patch16-224 | medium | `vit-large-patch16-224__b32__r224` | 32 | r224 | 844.0 | 8.1 | peak reserved closest to geometric mean of light/heavy (982 MiB); no retry; wall <= 600s |
| vit-large-patch16-224 | heavy | `vit-large-patch16-224__b128__r224` | 128 | r224 | 1,544.0 | 13.3 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| t5-small | light | `t5-small__b1__in128__new128` | 1 | 128/128 | 276.0 | 18.2 | batch 1, smallest peak reserved among batch-1 ok conditions |
| t5-small | medium | `t5-small__b128__in256__new128` | 128 | 256/128 | 1,444.0 | 22.2 | peak reserved closest to geometric mean of light/heavy (1759 MiB); no retry; wall <= 600s |
| t5-small | heavy | `t5-small__b128__in128__new512` | 128 | 128/512 | 11,214.0 | 71.7 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| t5-base | light | `t5-base__b1__in128__new128` | 1 | 128/128 | 688.0 | 32.8 | batch 1, smallest peak reserved among batch-1 ok conditions |
| t5-base | medium | `t5-base__b128__in128__new128` | 128 | 128/128 | 2,256.0 | 34.4 | peak reserved closest to geometric mean of light/heavy (2275 MiB); no retry; wall <= 600s |
| t5-base | heavy | `t5-base__b128__in512__new128` | 128 | 512/128 | 7,524.0 | 90.1 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |
| t5-3b | light | `t5-3b__b1__in256__new128` | 1 | 256/128 | 7,222.0 | 51.9 | batch 1, smallest peak reserved among batch-1 ok conditions |
| t5-3b | medium | `t5-3b__b32__in128__new128` | 32 | 128/128 | 9,154.0 | 85.3 | peak reserved closest to geometric mean of light/heavy (9032 MiB); no retry; wall <= 600s |
| t5-3b | heavy | `t5-3b__b16__in128__new256` | 16 | 128/256 | 11,296.0 | 113.7 | largest peak reserved with no allocator retry, <= 85% of 14913 MiB, wall <= 600s |

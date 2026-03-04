# nanochat training report

Generated: 2026-03-04 08:00:58

## Environment

### Git Information
- Branch: master
- Commit: 8b6ad0b (dirty)
- Message: Add combined runner for 002 vs 001-baseline on same node

### Hardware
- Platform: Linux
- CPUs: 64 cores (128 logical)
- Memory: 1510.9 GB
- GPUs: 8x NVIDIA H200
- GPU Memory: 1118.5 GB total
- CUDA Version: 12.8
- Hourly Rate: $16.00/hour

### Software
- Python: 3.10.19
- PyTorch: 2.9.1+cu128


### Bloat
- Characters: 144,696
- Lines: 3,039
- Files: 9
- Tokens (approx): 36,174
- Dependencies (uv.lock lines): 0

Run started: 2026-03-04 08:01:03

---

## Tokenizer training
timestamp: 2026-03-04 08:02:22

- max_chars: 2,000,000,000
- doc_cap: 10,000
- vocab_size: 32,768
- train_time: 73.1676
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 19
- token_bytes_mean: 6.6029
- token_bytes_std: 2.8250


## Tokenizer evaluation
timestamp: 2026-03-04 08:02:28

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 403 | 4.51 | +0.2% |
| korean | 893 | 745 | 1.20 | 797 | 1.12 | -7.0% |
| code | 1259 | 576 | 2.19 | 620 | 2.03 | -7.6% |
| math | 1834 | 936 | 1.96 | 1025 | 1.79 | -9.5% |
| science | 1112 | 260 | 4.28 | 258 | 4.31 | +0.8% |
| fwe-train | 4208518 | 900364 | 4.67 | 892476 | 4.72 | +0.9% |
| fwe-val | 4768657 | 1027270 | 4.64 | 1023546 | 4.66 | +0.4% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 403 | 4.51 | -4.1% |
| korean | 893 | 364 | 2.45 | 797 | 1.12 | -119.0% |
| code | 1259 | 309 | 4.07 | 620 | 2.03 | -100.6% |
| math | 1834 | 832 | 2.20 | 1025 | 1.79 | -23.2% |
| science | 1112 | 249 | 4.47 | 258 | 4.31 | -3.6% |
| fwe-train | 4208518 | 874799 | 4.81 | 892476 | 4.72 | -2.0% |
| fwe-val | 4768657 | 1001442 | 4.76 | 1023546 | 4.66 | -2.2% |


## Base model training
timestamp: 2026-03-04 11:40:33

- run: dummy
- device_type: 
- fp8: True
- fp8_recipe: tensorwise
- compile_mode: default
- depth: 26
- aspect_ratio: 64
- head_dim: 128
- max_seq_len: 2048
- window_pattern: SSSL
- n_kv_head: -1
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 8.2500
- device_batch_size: 16
- total_batch_size: -1
- embedding_lr: 0.3000
- unembedding_lr: 0.0040
- weight_decay: 0.2000
- matrix_lr: 0.0200
- scalar_lr: 0.5000
- adam_beta1: 0.8000
- adam_beta2: 0.9500
- warmup_ratio: 0.0000
- warmdown_ratio: 0.5000
- final_lr_frac: 0.0000
- resume_from_step: -1
- matrix_warmdown_frac: 1.0000
- adamw_warmdown_frac: 0.3000
- use_hyperball: False
- use_teon: True
- adam_every_n: 2
- mtp_schedule: 
- tie_embed_until: 0.0000
- batch_schedule: 
- dynamic_window: False
- eval_every: 1000
- eval_tokens: 20,971,520
- core_metric_every: 999,999
- core_metric_max_per_task: 500
- sample_every: -1
- save_every: -1
- model_tag: None
- Number of parameters: 1,845,458,812
- FLOPs per token: 6.185864e+09
- Iterations: 7226
- Training tokens: 7,577,010,176
- Token:Param ratio: 8.2492
- DDP world size: 8
- Min val bpb: 0.9193
- Final val bpb: 0.9193
- CORE metric: 0.1229
- MFU: 46.54%
- Total flops: 4.687036e+19
- Training time: 212.11m
- Peak memory: 83951.64MiB


## Base model evaluation
timestamp: 2026-03-04 11:47:29

- model: base_model (step 7226)
- CORE metric: 0.1192
- train bpb: 0.9187
- val bpb: 0.9193
- hellaswag_zeroshot: 0.0986
- jeopardy: 0.0033
- bigbench_qa_wikidata: 0.2221
- arc_easy: 0.3176
- arc_challenge: -0.0159
- copa: 0.0600
- commonsense_qa: 0.1667
- piqa: 0.2731
- openbook_qa: 0.0880
- lambada_openai: 0.2633
- hellaswag: 0.0975
- winograd: 0.1429
- winogrande: 0.0387
- bigbench_dyck_languages: 0.1090
- agi_eval_lsat_ar: 0.1196
- bigbench_cs_algorithms: 0.4068
- bigbench_operators: 0.0857
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.0543
- coqa: 0.0894
- boolq: -0.1742
- bigbench_language_identification: 0.1750
- sample 0: <|bos|>The capital of France is the most important city of the world’s’s’s’s’s’s’s’s’s
- sample 1: <|bos|>The chemical symbol of gold is the symbol of the sun god..........
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be the day of the year 1111111111
- sample 3: <|bos|>The opposite of hot is the the the the the the the the the the the the the the the the
- sample 4: <|bos|>The planets of the solar system are:::::::::::::::::
- sample 5: <|bos|>My favorite color is the the the the the the the the the the the the the the the the
- sample 6: <|bos|>If 5*x + 3 = 13, then x is the number of the two-digit number of the first and last digit of the number
- unconditioned 0: <|bos|>The announcement in America that begins Monday, but one day is for for for more for for longer for for for for for and for for for for for in for and for for for the child.
This is a very ch ch ch ch ch ch.
YoungstersCh ch ch ch ch ch ch ch-ch chill ch ch ch ch ch ch ch Ch ch ch chch chill apart ch ch ch ch chillchat chill chillch ch chillCh chat Ch Ch Ch Ch ch ch chch Ch chillsch chill chill ch ch Ch ch ChCh ChCh ChrysChCh chCh Ch Ch-ch Ch chaChch ch ch chill Ch chCh
- unconditioned 1: <|bos|>New York Roof the World Heritage Languages Corner stands now a generation have inherited the River of the Knight Knight after thebreak of his life's an a a a lot is cut weld, which which which he almost entirely a a a a  a a aa a a a aa a a a a a a a a a a a a a b a a a a a a a a a a a a an a aa a a a a an a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a
- unconditioned 2: <|bos|>Ininking cities of execution speedier inspiredrewrewrew17

A NEW Culture Ecology Ecology Biology Life Support

- Perm A A R Card Unit Unit Unit Unit Unit Unit Unit Planning Spruce El C C C C T N H Communication Communicating Environmental Environmental Ecology Ecology Conservation Education Centre Comment Comment Comment Comment Commentaryaryaryaryaryaryaryaryaryaryaryaryaryaryaryaryaryary BacteriaBBECECECECECECECECAC EECECECECECEOEOEOEOEOEO E R M A BAS BAS Legislation E E R R R AB A B A C C C D D D
- unconditioned 3: <|bos|>Civil War 100
The Revolutionary War never to to to to to to toto to to to the oldenendendendendendendendendendendendendendendendendendendendendendendendendendendendendendendendendendendendendendruendendendendendendendendendendendendendendendendendendendendendendendendendend end endear endendend-end endendendendend-end end-end end justify endendendendendendendendsendend Endendendendendendendendendendendendend
- unconditioned 4: <|bos|>Dialk means missing Bos oros and thought to exterminate the Republic of England. An adherence to devoutoutououououououououououououou Toupspspspspspspspspspspspspspspspspsps
Hououououououousououlousousousousousousououououdousousou�ighighousou DouchemoouOUoustakiakiaki```````
Dobobobobobobsobsobsobsobsobsidididsidsidsithithithithithylylyl
- unconditioned 5: <|bos|>Freedom in the family consists of the whom the decision was conducted on whether or whether individuals or from societies, memorialize their reproductive rights. The political situation situation in England during which which which figures figures figures figures figures figures figure figure figures figures figures figure figures It It’s’s’s’s’s’s’s’s’s’s’s’s’s’s’s’s’s’’s own’s his his him him her her her her her her her she her her her she she herself she her she herHer him she she herself she sheshe sheshe she hershe her she herself herself she she sheshe she herselfshe she shesheshe she sheshe she
- unconditioned 6: <|bos|>SASUMUMUMUMUMUMORORORORORMEDEDED LEADLLLE LEADEFERHOHOHOHOHOHIGHTMULULUMULLONANANENEELETUUIDIDIDIME FLOUROURTHTHYYY-F Cam Cam Cam CamCamcam CamCamcam CamcamcamcucamCamCamCamcamcamcamcam cam Cam CameraCamcam Cam Camcam microphone camera Cam cameracam camera CamCamcam Cotton camera cameracamCamarCamcam cameracam lens incam cam CR CR ESURUUUUUniversity Cam camcam CameraCamcam
- unconditioned 7: <|bos|>The bladder is located in in in in the Dutch Caribbean Province,, during a year...with with the the radicionive Wik Wik Wik Wik Wikediaediaediaediaediaediaediaediaediaediaedia is the Digital Maj Maj Maj Maj Maj Maj Maj Maj Maj Maj Royal Royal Royal Royal Royal Royal Royal Royal Royal Royal Royal Institution Institution institution Institution Institution Campus Astron Institution Asia Pacific Pacific Pacificacificacific Pacific Tropical Pacificacific Pacific Pacific Pacificacific Pacific Pacific Tu Tu Stanford Stanford Stanford Stanford UC Stanford Medicine Medicine M M M M Muslims Muslims Muslims Ramadan Islam Islamisations Western European European European Islam Islam Pakistan Spain Sudan China Philippines Muslim Pakistan colonized colonized


## Chat evaluation sft
timestamp: 2026-03-04 12:32:13

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.2706
- ARC-Challenge: 0.2440
- MMLU: 0.2567
- GSM8K: 0.0000
- HumanEval: 0.0000
- SpellingBee: 0.0000
- ChatCORE metric: 0.0048


## Summary

- Characters: 144,696
- Lines: 3,039
- Files: 9
- Tokens (approx): 36,174
- Dependencies (uv.lock lines): 0

| Metric          | BASE     | SFT      | RL       |
|-----------------|----------|----------|----------|
| CORE            | 0.1192   | -        | -        |
| ARC-Challenge   | -        | 0.2440   | -        |
| ARC-Easy        | -        | 0.2706   | -        |
| GSM8K           | -        | 0.0000   | -        |
| HumanEval       | -        | 0.0000   | -        |
| MMLU            | -        | 0.2567   | -        |
| ChatCORE        | -        | 0.0048   | -        |

Total wall clock time: 4h31m

summarize the gsm8k_cot performance of these models obtained by 
" ls | grep checkpoints-llama32-1b-mathpros1sfeos_intra_from_"

Let me first find those checkpoint directories and then look for their GSM8K CoT results.

**Command Executed:**
```bash
ls /home/aiops/zhuty/nanotron | grep "checkpoints-llama32-1b-mathpros1sfeos_intra_from_"
```

**Output:**
```
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k_old
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_2000_hf_converted-4k
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_4000_hf_converted-4k
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_6000_hf_converted-4k
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_8000_hf_converted-4k
```


```bash
for dir in /home/aiops/zhuty/nanotron/checkpoints-llama32-1b-mathpros1sfeos_intra_from_*; do echo "=== $(basename $dir) ==="; find "$dir" -path "*gsm8k_cot*" -name "*.json" 2>/dev/null | head -5; done
```

**Current Output:**
```
=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k ===
=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k ===
<|cursor|>
```

Subagent: 88e87a64-ed83-4474-a557-8c123fd58844

No JSON files found inside those directories. Let me search more broadly for where GSM8K results might be stored.

**Command Executed:**
```bash
for dir in /home/aiops/zhuty/nanotron/checkpoints-llama32-1b-mathpros1sfeos_intra_from_*; do echo "=== $(basename $dir) ==="; ls "$dir"/ 2>/dev/null | head -10; echo ""; done
```

**Output:**
```
=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k ===
100_hf
200_hf
300_hf
400
400_hf
500
500_hf
latest.txt

=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k ===
100_hf
200_hf
300_hf
400
400_hf
500
500_hf
latest.txt

=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k_old ===
1000
500
500_hf
latest.txt

=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_2000_hf_converted-4k ===
100_hf
200_hf
300_hf
400
400_hf
500
500_hf
latest.txt

=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_4000_hf_converted-4k ===
100_hf
200_hf
300_hf
400
400_hf
500
500_hf
latest.txt

=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_6000_hf_converted-4k ===
100_hf
200_hf
300_hf
400
400_hf
500
500_hf
latest.txt

=== checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_8000_hf_converted-4k ===
100_hf
200_hf
300_hf
400
400_hf
500
500_hf
latest.txt

```


**Command Executed:**
```bash
ls /home/aiops/zhuty/nanotron/checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k/500_hf/
```

**Output:**
```
config.json             harness_eval_8shot       tokenizer.json
generation_config.json  model.safetensors        tokenizer_config.json
harness_eval_0shot      special_tokens_map.json
```


**Command Executed:**
```bash
ls /home/aiops/zhuty/nanotron/checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k/500_hf/harness_eval_8shot/
```

**Output:**
```
__home__aiops__zhuty__nanotron__checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k__500_hf
```


**Command Executed:**
```bash
find /home/aiops/zhuty/nanotron/checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k/500_hf/harness_eval_8shot/ -type f -name "*.json" | head -10
```

**Output:**
```
/home/aiops/zhuty/nanotron/checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k/500_hf/harness_eval_8shot/__home__aiops__zhuty__nanotron__checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k__500_hf/results_2026-03-05T08-57-36.877244.json
```


**Command Executed:**
```bash
python3 -c "
import json, sys
f = '/home/aiops/zhuty/nanotron/checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k/500_hf/harness_eval_8shot/__home__aiops__zhuty__nanotron__checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k__500_hf/results_2026-03-05T08-57-36.877244.json'
d = json.load(open(f))
# print top-level keys
print('Keys:', list(d.keys())[:10])
if 'results' in d:
    for k,v in d['results'].items():
        if 'gsm8k' in k.lower():
            print(k, v)
"

```

**Output:**
```
Keys: ['results', 'group_subtasks', 'configs', 'versions', 'n-shot', 'higher_is_better', 'n-samples', 'config', 'git_hash', 'date']
gsm8k {'alias': 'gsm8k', 'exact_match,strict-match': 0.052312357846853674, 'exact_match_stderr,strict-match': 0.006133057708959232, 'exact_match,flexible-extract': 0.05534495830174375, 'exact_match_stderr,flexible-extract': 0.006298221796179556}
gsm8k_cot {'alias': 'gsm8k_cot', 'exact_match,strict-match': 0.04245640636846096, 'exact_match_stderr,strict-match': 0.005553837749990046, 'exact_match,flexible-extract': 0.052312357846853674, 'exact_match_stderr,flexible-extract': 0.006133057708959229}
```

Good, now I understand the structure. Let me extract gsm8k_cot results across all models and checkpoints.

**Command Executed:**
```bash
python3 -c "
import json, glob, os, re

base = '/home/aiops/zhuty/nanotron'
dirs = sorted(glob.glob(os.path.join(base, 'checkpoints-llama32-1b-mathpros1sfeos_intra_from_*')))

for d in dirs:
    dirname = os.path.basename(d)
    # find all harness_eval dirs (0shot and 8shot)
    result_files = glob.glob(os.path.join(d, '*_hf', 'harness_eval_*', '*', 'results_*.json'))
    if not result_files:
        # also check 0shot
        result_files = glob.glob(os.path.join(d, '*_hf', 'harness_eval_0shot', '*', 'results_*.json'))
    for rf in sorted(result_files):
        # extract step from path
        parts = rf.split('/')
        step_part = [p for p in parts if p.endswith('_hf')][0] if any(p.endswith('_hf') for p in parts) else '?'
        eval_type = [p for p in parts if 'harness_eval' in p][0] if any('harness_eval' in p for p in parts) else '?'
        try:
            data = json.load(open(rf))
            results = data.get('results', {})
            gsm8k_cot = results.get('gsm8k_cot', {})
            if gsm8k_cot:
                strict = gsm8k_cot.get('exact_match,strict-match', 'N/A')
                flex = gsm8k_cot.get('exact_match,flexible-extract', 'N/A')
                print(f'{dirname} | step={step_part} | {eval_type} | strict={strict:.4f} | flex={flex:.4f}')
        except Exception as e:
            print(f'{dirname} | step={step_part} | {eval_type} | ERROR: {e}')
"

```

**Output:**
```
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k | step=100_hf | harness_eval_8shot | strict=0.0281 | flex=0.0425
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k | step=200_hf | harness_eval_8shot | strict=0.0371 | flex=0.0561
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k | step=300_hf | harness_eval_8shot | strict=0.0409 | flex=0.0667
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k | step=400_hf | harness_eval_8shot | strict=0.0447 | flex=0.0576
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_10000_hf_converted-4k | step=500_hf | harness_eval_8shot | strict=0.0425 | flex=0.0523
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k | step=100_hf | harness_eval_8shot | strict=0.0243 | flex=0.0356
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k | step=200_hf | harness_eval_8shot | strict=0.0576 | flex=0.0697
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k | step=300_hf | harness_eval_8shot | strict=0.0341 | flex=0.0576
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k | step=400_hf | harness_eval_8shot | strict=0.0311 | flex=0.0523
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k | step=500_hf | harness_eval_8shot | strict=0.0265 | flex=0.0508
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_2000_hf_converted-4k | step=100_hf | harness_eval_8shot | strict=0.0326 | flex=0.0561
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_2000_hf_converted-4k | step=200_hf | harness_eval_8shot | strict=0.0705 | flex=0.0963
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_2000_hf_converted-4k | step=300_hf | harness_eval_8shot | strict=0.0561 | flex=0.0788
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_2000_hf_converted-4k | step=400_hf | harness_eval_8shot | strict=0.0607 | flex=0.0819
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_2000_hf_converted-4k | step=500_hf | harness_eval_8shot | strict=0.0607 | flex=0.0773
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_4000_hf_converted-4k | step=100_hf | harness_eval_8shot | strict=0.0409 | flex=0.0599
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_4000_hf_converted-4k | step=200_hf | harness_eval_8shot | strict=0.0462 | flex=0.0758
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_4000_hf_converted-4k | step=300_hf | harness_eval_8shot | strict=0.0417 | flex=0.0781
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_4000_hf_converted-4k | step=400_hf | harness_eval_8shot | strict=0.0432 | flex=0.0644
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_4000_hf_converted-4k | step=500_hf | harness_eval_8shot | strict=0.0371 | flex=0.0569
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_6000_hf_converted-4k | step=100_hf | harness_eval_8shot | strict=0.0334 | flex=0.0417
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_6000_hf_converted-4k | step=200_hf | harness_eval_8shot | strict=0.0637 | flex=0.0773
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_6000_hf_converted-4k | step=300_hf | harness_eval_8shot | strict=0.0531 | flex=0.0735
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_6000_hf_converted-4k | step=400_hf | harness_eval_8shot | strict=0.0561 | flex=0.0690
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_6000_hf_converted-4k | step=500_hf | harness_eval_8shot | strict=0.0447 | flex=0.0667
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_8000_hf_converted-4k | step=100_hf | harness_eval_8shot | strict=0.0296 | flex=0.0387
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_8000_hf_converted-4k | step=200_hf | harness_eval_8shot | strict=0.0644 | flex=0.0766
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_8000_hf_converted-4k | step=300_hf | harness_eval_8shot | strict=0.0500 | flex=0.0675
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_8000_hf_converted-4k | step=400_hf | harness_eval_8shot | strict=0.0349 | flex=0.0516
checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_8000_hf_converted-4k | step=500_hf | harness_eval_8shot | strict=0.0531 | flex=0.0629
```

Now let me also check for `_old` and `0shot` results, and look for any without `gsm8k_cot`:

**Command Executed:**
```bash
find /home/aiops/zhuty/nanotron/checkpoints-llama32-1b-mathpros1sfeos_intra_from_llama32-1b-mathpros1sfeos_diff_intra-4k_12500-4k_old/ -name "results_*.json" 2>/dev/null
```

**Output:**
```

```

The `_old` directory has no eval results. Here's the summary of **gsm8k_cot** (8-shot) performance across all models. The model names differ by the `diff_intra` checkpoint step they were initialized from (2k, 4k, 6k, 8k, 10k, 12.5k), and each was further fine-tuned for 100–500 steps:

**`exact_match,flexible-extract` (the more lenient metric):**

| Init from (diff_intra step) | step 100 | step 200  | step 300  | step 400 | step 500 | Best           |
| --------------------------- | -------- | --------- | --------- | -------- | -------- | -------------- |
| **2000**                    | 5.61%    | **9.63%** | 7.88%     | 8.19%    | 7.73%    | **9.63% @200** |
| **4000**                    | 5.99%    | 7.58%     | **7.81%** | 6.44%    | 5.69%    | 7.81% @300     |
| **6000**                    | 4.17%    | **7.73%** | 7.35%     | 6.90%    | 6.67%    | 7.73% @200     |
| **8000**                    | 3.87%    | **7.66%** | 6.75%     | 5.16%    | 6.29%    | 7.66% @200     |
| **10000**                   | 4.25%    | 5.61%     | **6.67%** | 5.76%    | 5.23%    | 6.67% @300     |
| **12500**                   | 3.56%    | **6.97%** | 5.76%     | 5.23%    | 5.08%    | 6.97% @200     |
**Key observations:**
- **Best overall model**: init from **diff_intra-2000**, at step 200, achieving **9.63%** flexible-extract (7.05% strict).
- **Earlier diff_intra checkpoints tend to perform better**: 2000 > 4000 ≈ 6000 > 8000 > 10000 ≈ 12500.
- **Step 200 is typically the sweet spot** — most models peak at step 200, with performance degrading after that (possible overfitting).
- All scores are quite low overall (sub-10%), suggesting these intra-trained models still struggle significantly on GSM8K CoT.
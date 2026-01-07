  sailctl job create nanollama31bmathprosf -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py   --config-file examples/config_llama32_1b_continual.yaml " 

  sailctl job create nanollama33bmathprosf -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py   --config-file examples/config_llama32_3b_continual.yaml " 

  sailctl job create nanopencoder484opptsfp1 -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py   --config-file examples/config_opencoder484_continual.yaml " 

  sailctl job create nanollama31bmathpros10sf -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; DATASET_NAME=mathpros10sf torchrun --nproc_per_node=8 run_train.py --config-file examples/config_llama32_1b_continual_template.yaml " 


  sailctl job create nanollama31bmathpros10sf -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; DATASET_NAME=mathpros1sf torchrun --nproc_per_node=8 run_train.py --config-file examples/config_llama32_1b_continual_template.yaml " 

    sailctl job create nanollama31bmathpros10sf -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; DATASET_NAME=mathpromaxsf torchrun --nproc_per_node=8 run_train.py --config-file examples/config_llama32_1b_continual_template.yaml " 


    python3 tools/preprocess_data.py --tokenizer-name-or-path tyzhu/opencoder484 --output-folder /home/aiops/zhuty/cont_data/opcansf/tokenized/data --n-tasks 32 jsonl --dataset /home/aiops/zhuty/cont_data/opcansf/train

    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Llama-3.2-1B --output-folder /home/aiops/zhuty/cont_data/mathpros10sf/tokenized --n-tasks 32 jsonl --dataset /home/aiops/zhuty/cont_data/mathpros10sf/train

    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Llama-3.2-1B --output-folder /home/aiops/zhuty/cont_data/opcansf/llama_tokenized --n-tasks 32 jsonl --dataset /home/aiops/zhuty/cont_data/opcansf/train

    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Llama-3.2-1B --output-folder /home/aiops/zhuty/cont_data/opptsfp1/llama_tokenized --n-tasks 32 jsonl --dataset /home/aiops/zhuty/cont_data/opptsfp1/train

    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Llama-3.2-1B --output-folder /home/aiops/zhuty/cont_data/mathpromaxsf/llama_tokenized --n-tasks 32 jsonl --dataset /home/aiops/zhuty/cont_data/mathpromaxsf/train

sailctl job create preprocessmathpromaxsf -g 1 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ;     python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Llama-3.2-1B --output-folder /home/aiops/zhuty/cont_data/mathpromaxsf/llama_tokenized --n-tasks 32 jsonl --dataset /home/aiops/zhuty/cont_data/mathpromaxsf/train " -f ~/Downloads/configs/saildata.yml" 



  sailctl job create nano2llama3b -g 8 --replicas 4 --debug -p high --args --image asia-docker.pkg.dev/sail-tpu-02/images/common/golden-image:12.3 --args 
source /home/aiops/zhuty/nano_start.sh ; cd /home/aiops/zhuty/nanotron ; bash train_multi.sh


📖 - READER: 🐿 Jsonl
    Runtime: (11.29%) 12.60 seconds±0.93 seconds/task, min=10.70 seconds, max=14.43 seconds [0.04 milliseconds±1.55 milliseconds/doc]
    Stats: {input_files: 100, doc_len: 15059809779 [min=88, max=2587001, 1421.07±4939/doc], documents: 10597527 [min=105903, max=105976, 105975.27±7/input_file]}
🔢 - TOKENIZER: ✍️ Writer
    Runtime: (88.71%) 1 minute and 39.02 seconds±8.03 seconds/task, min=1 minute and 25.78 seconds, max=1 minute and 54.92 seconds [2 seconds and 966.97 milliseconds±1 second and 523.08 milliseconds/batch]
    Stats: {tokens: 4266491643 [min=44, max=1743663, 402.59±1600/doc]}



  # Dec 30
    sailctl job create nanollama31bopcansf -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; DATASET_NAME=opcansf torchrun --nproc_per_node=8 run_train.py --config-file examples/config_llama32_1b_continual_template.yaml " 

    sailctl job create nanollama31bopptsfp1 -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; DATASET_NAME=opptsfp1 torchrun --nproc_per_node=8 run_train.py --config-file examples/config_llama32_1b_continual_template.yaml " 

# dec 31
    sailctl job create nanollama31bmathpromaxsf -g 8 --debug -p high --command-line-args "source nano_start.sh ; cd /home/aiops/zhuty/nanotron ; DATASET_NAME=mathpromaxsf torchrun --nproc_per_node=8 run_train.py --config-file examples/config_llama32_1b_continual_template.yaml " 
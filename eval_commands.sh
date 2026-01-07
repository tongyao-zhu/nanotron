    # for step in 00250 00500 00750 01000 01250 01500 01750 02000 02250 02500 02750    ; do 
    for step in $(seq 500 500 12500); do
    # for model_name in llama3-1b-mathprosf-12.5ksteps-8k-const-lr5e-5-bsz512 ; do
    for model_name in checkpoints ; do
    model=/home/aiops/zhuty/nanotron/$model_name/$step\_hf
    # model=/home/aiops/zhuty/litgpt_out/pretrain/$model_name/step-000$step\_hf
    # model=/home/aiops/zhuty/litgpt_out/pretrain/llama3-1b-finecode-25ksteps-4nodes-2k/step-000$step\_hf
    echo $model $task
    # sailctl job create basictest$step --image ghcr.io/bigcode-project/evaluation-harness-multiple  --debug -g 1 -p low   --command-line-args " source /home/aiops/zhuty/lit_start.sh ; cd /home/aiops/zhuty/litgpt ;  lm_eval --model hf --model_args pretrained=$model  --tasks hellaswag,arc_easy,arc_challenge --num_fewshot 0 --batch_size 32 --output_path $model/harness_eval_0shot/ --log_samples" --track  & 

    # sailctl job create math4test$step --image ghcr.io/bigcode-project/evaluation-harness-multiple  --debug -g 1 -p low   --command-line-args " source /home/aiops/zhuty/lit_start.sh ; cd /home/aiops/zhuty/litgpt ; export HF_ALLOW_CODE_EVAL=1 ;  lm_eval --model hf --model_args pretrained=$model --tasks hendrycks_math --num_fewshot 4 --batch_size 64 --output_path $model/harness_eval_4shot/ --confirm_run_unsafe_code --trust_remote_code --log_samples" --track  &

    sailctl job create gsm8test$step --image ghcr.io/bigcode-project/evaluation-harness-multiple  --debug -g 1 -p low   --command-line-args " source /home/aiops/zhuty/lit_start.sh ; cd /home/aiops/zhuty/litgpt ; export HF_ALLOW_CODE_EVAL=1 ;  lm_eval --model hf --model_args pretrained=$model --tasks gsm8k,gsm8k_cot --num_fewshot 8 --batch_size 64 --output_path $model/harness_eval_8shot/ --confirm_run_unsafe_code --trust_remote_code --log_samples" --track  &

    done
    done 
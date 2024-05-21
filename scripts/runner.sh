pdataset_pattern_name="pattern3"

torchrun --nproc_per_node=8 ${HOME}/mergoo/scripts/train.py \
--tokenizer hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160 \
--repo_id ./merged_tanuki_dentaku \
--wandb_project router_tanuki_dentaku \
--wandb_entity weblab-geniac3 \
--dataset_pattern_name ${dataset_pattern_name} \
--output_dir /storage6/aa_fujimoto/router/${dataset_pattern_name}

python  ${HOME}/mergoo/scripts/infarence.py \
--tokenizer hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160  \
--repo_id /storage6/aa_fujimoto/router/{dataset_pattern_name} \
--prompt_file ./default_eval.csv > ./eval/{dataset_pattern_name}.txt

dataset_pattern_name="pattern4"

torchrun --nproc_per_node=8 ${HOME}/mergoo/scripts/train.py \
--tokenizer hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160 \
--repo_id ./merged_tanuki_dentaku \
--wandb_project router_tanuki_dentaku \
--wandb_entity weblab-geniac3 \
--dataset_pattern_name ${dataset_pattern_name} \
--output_dir /storage6/aa_fujimoto/router/${dataset_pattern_name}

python  ${HOME}/mergoo/scripts/infarence.py \
--tokenizer hatakeyama-llm-team/Tanuki_pretrained_stage6_step62160  \
--repo_id /storage6/aa_fujimoto/router/{dataset_pattern_name} \
--prompt_file ./default_eval.csv > ./eval/{dataset_pattern_name}.txt



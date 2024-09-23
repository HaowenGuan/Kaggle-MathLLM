# Kaggle-MathLLM

This is my repository for the Kaggle competition [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics).

## 1st Week Progress

* Initialize the repo and preparing environment and data.
* Getting familiar with the data and the competition objectives [visualize_data.ipynb](visualize_data.ipynb).
* Review the current leading approaches in the forum, and working on reproducing some result.
* Currently, I am following the most prevailing approach, which is finetune a BGE Embedding Model.
  * baai-general-embedding (BGE) model is the current state-of-the-art embedding Model which map a paragraph of text into vector representation.
  * The general idea of this approach is to use BGE to map the question description and each answer choice into vector representation, then calculate the similarity between the question and each answer choice, and then rank the answer choices based on the similarity score.

How to run the training code:

Firstly, configure the environment properly.

Then run the following command to finetune the BGE Embedding Model:

```bash
python eedi-train-finetune-bge-embedding-model.py
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine --model_name_or_path BAAI/bge-large-en-v1.5 --input_file finetune_data.jsonl --candidate_pool pretrain_data.jsonl --output_file finetune_data_minedHN.jsonl --range_for_sampling 1-100 --negative_number 15 --use_gpu_for_searching
torchrun --nproc_per_node 2 -m FlagEmbedding.baai_general_embedding.finetune.run --output_dir eedi_model --model_name_or_path BAAI/bge-large-en-v1.5 --train_data finetune_data_minedHN.jsonl --learning_rate 1e-5 --fp16 --temperature 0.03 --num_train_epochs 3 --per_device_train_batch_size 8 --query_max_len 256 --passage_max_len 64 --logging_steps 100 --query_instruction_for_retrieval "" --report_to none --save_steps 250
```

Inference code will be added soon.



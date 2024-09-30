# Kaggle-MathLLM

This is my repository for the Kaggle competition [Eedi - Mining Misconceptions in Mathematics](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics).

## 1st Week Progress

* Initialize the GitHub repo and prepare the environment and data.
* Getting familiar with the data and the competition objectives [visualize_data.ipynb](visualize_data.ipynb).
* Review the current leading approaches in the forum, and work on reproducing some results.
* Currently, I am review the prevailing approaches from forum, one of it is fine-tuning a BGE Embedding Model.
  * baai-general-embedding (BGE) model is the current state-of-the-art embedding Model which maps a paragraph of text into vector representation.
  * The general idea of this approach is using the BGE model to map the question description and each answer choice into vector representation, then calculate the similarity between the question and each answer choice, and then rank the answer choices based on the similarity score.
* My first attempt is to fine-tune the BGE Embedding Model on the training data, the code is as following:

How to run the training code:

Firstly, configure the environment properly.

Then run the following command to finetune the BGE Embedding Model:

```bash
python eedi-train-finetune-bge-embedding-model.py
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine --model_name_or_path BAAI/bge-large-en-v1.5 --input_file finetune_data.jsonl --candidate_pool pretrain_data.jsonl --output_file finetune_data_minedHN.jsonl --range_for_sampling 1-100 --negative_number 15 --use_gpu_for_searching
torchrun --nproc_per_node 2 -m FlagEmbedding.baai_general_embedding.finetune.run --output_dir eedi_model --model_name_or_path BAAI/bge-large-en-v1.5 --train_data finetune_data_minedHN.jsonl --learning_rate 1e-5 --fp16 --temperature 0.03 --num_train_epochs 3 --per_device_train_batch_size 8 --query_max_len 256 --passage_max_len 64 --logging_steps 100 --query_instruction_for_retrieval "" --report_to none --save_steps 250
```

## 2nd Week Progress

* Since last week, I made my first attempt at fine-tuning a BGE Embedding Model using the training data. However, the training did not go as well as expected due to the presence of many unfamiliar hyperparameters. As a result, I decided to slow down and start with a basic baseline approach to gradually improve performance.
  
* **Preliminaries about this competition:**
  * The competition involves a multiple-choice math question answering task.
  * Each incorrect question-answer pair is associated with a misconception label, which indicates general categories of mistakes that students make.
  * For example, for the question-answer pair `What is 2 + 2 * 3? Result: 12`, the misconception label would be `Unawareness of the correct Order of Operations`.

* I began with a simple baseline approach by using a pretrained, unmodified BGE model `baai-bge-large-en` to encode each question-answer pair and all provided misconception choices. I then calculated the cosine similarity between the question-answer pair and each misconception choice, ranked the misconceptions based on the similarity scores, and submitted the top 25 misconceptions for each question-answer pair according to the competition rules.

* Currently, I have tested two different context architectures and inference strategies:
  * **Strategy 1:** Formulated the context and question as `df["ConstructName"] + " ### " + df["QuestionText"] + " ### " + Answer[A-D]Text`. This approach achieved a public test set score of 0.141.
  * **Strategy 2:** Formulated the context and question as `"Category: " + df["ConstructName"] + ". Question: " + df["QuestionText"] + " Given Answer: " + Answer[A-D]Text`. This approach achieved a public test set score of 0.144.

* For the next step, I plan to fine-tune the BGE model on the training data using the aforementioned context architectures to see if performance can be improved.



















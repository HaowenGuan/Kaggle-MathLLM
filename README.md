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

## 3rd Week Progress
* Since last week, I started experimenting with 2 baseline approaches, where they test scores of 0.141 and 0.144.
* This week, I have been spending a lot of time importing baai-BGE model training code and trying to get it to work.
* I am mostly following the tutorial in Flag Embedding Open repository: https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune
* I am still working on debugging the training code. There are some dependent issues with the anaconda environment, a package is unable to be installed. I am trying to switch to a native python environment. This is costing a lot of time.
* Hopefully, I will successfully fine-tune my first model next week.

## 4th Week Progress
* Since last week, I have been working on debugging the training code for the BGE model. I have successfully resolved the dependency issues and have been able to run the training code.
  * Reference: [Training-finetune-bge-embedding-model.ipynb](Training-finetune-bge-embedding-model.ipynb)
  * I have successfully fine-tuned the BGE model on the training data using the context architecture `test["SubjectName"] + " ### " + test["ConstructName"] + " ### " + test["QuestionText"] + " ### " + Answer[A-D]Text`. The model was trained for 3 epochs, loss decreased from 1.48 to 0.45.
* I have my inference code at [Inference-finetune-bge-embedding-model.ipynb](Inference-finetune-bge-embedding-model.ipynb).
* My current public test set score is 2.1 right now. According to the leaderboard, the finetune can go up to at least 0.27-0.4. Since it is just a preliminary attempt, I am happy with the results.

## 5th Week Progress
* Last week, I had the first successful try on fine-tuning a model. The model was trained for 3 epochs, loss decreased from 1.48 to 0.45. The public test score is 2.1.
* This week, I have been doing extensive training with various trick.
  * I retrained a new model with larger about of hard negative samples (50s, previous is 10s).
    * Explanation: In this research, we are predicting the correct misconception label for each question-answer pair. However, in the choices of misconception labels, there are in total 2587 total misconception labels.  Though this is a multi-class classification problem, the number of classes is too large, and we can't compute the embedding of all misconception labels to calculate the full cross-entropy loss. Therefore, we need to sample a subset of misconception labels as negative samples to calculate the loss. The number of negative samples is a hyperparameter that needs to be tuned. The more negative samples we use, the more robust the model will be, but the slower the training will be.
  * This time, I was able to train the model with 50 negative samples. The model was trained for 3 epochs, loss decreased from 1.48 to 0.3. The public test score is 2.45. [0.245-infer-finetune-bge-embedding-model.ipynb](0.245-infer-finetune-bge-embedding-model.ipynb).
  * I took the above model and retrained it for an additional 3 epochs. The model was trained for 6 epochs, loss decreased from 0.3 to 0.2. The public test score is 2.31 (Dropped). Seems like the model is overfitting.
  
## 6th Week Progress
* Last week, I achieved a public test score of 2.45 by training a model with 50 negative samples for 3 epochs. However, the model overfitted when trained for 6 epochs.
* This week, we need a better learning strategy.
* Previously, we dumped all correct question-answer pairs because this is a misconception prediction task, and there is no misconception label for correct question-answer pairs. However, it is possible to use the correct question-answer pairs to train the model.
* I manually created a new misconception label called `Correct Answer.` and assigned it to all correct question-answer pairs. By making this simple change, I was able to increase the total number of training samples from 4800 to 6300.
* Furthermore, letting model learn from correct question-answer pairs can help the model to learn the correct answer patterns, so it can better predict the misconception labels (presumably).
* This experiment did not end up well. The public test score is 2.25. I think it is related to the fact that I used the above model to continue the training. The model is already overfitting, and adding more data to the training set will make the model overfit more.
* Next week, I will do more experiments with different learning strategies.













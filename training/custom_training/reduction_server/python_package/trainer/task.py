# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

import torch
import torch.distributed as dist
torch.cuda.empty_cache()

import datasets
from datasets import ClassLabel, Sequence, load_dataset

import transformers
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    EvalPrediction, 
    Trainer, 
    TrainingArguments,
    default_data_collator)

from google.cloud import storage


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", default=2)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=32)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else "")
    argv = parser.parse_args()

    model_name_or_path = "bert-large-uncased"
    padding = "max_length"
    max_seq_length = 128

    datasets = load_dataset("imdb", verification_mode='no_checks')
    label_list = datasets["train"].unique("label")
    label_to_id = {1: 1, 0: 0, -1: 0}

    tokenizer = AutoTokenizer.from_pretrained(
      model_name_or_path,
      use_fast=True,
    )

    def preprocess_function(examples):
        """
        Tokenize the input example texts
        """
        args = (examples["text"],)
        result = tokenizer(
          *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
          result["label"] = [label_to_id[example] for example in examples["label"]]

        return result

    # apply preprocessing function to input examples
    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)

    model = AutoModelForSequenceClassification.from_pretrained(
      model_name_or_path, 
      num_labels=len(label_list)
    )

    ngpus_per_node = torch.cuda.device_count()
    world_size = int(os.environ["WORLD_SIZE"])

    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    world_size =  world_size * ngpus_per_node

    start = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(f'Starting distributed training: {start}') 

    # Use torch.multiprocessing.spawn to launch distributed processes
    torch.multiprocessing.spawn(main_worker,
    args = (ngpus_per_node, world_size, datasets, model, tokenizer, argv),
    nprocs = ngpus_per_node,
    join = True)

    end = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(f'Distributed training complete: {end}')

def main_worker(local_rank, ngpus_per_node, world_size, datasets, model, tokenizer, argv):

    # This is the (global) rank of the current process
    rank = int(os.environ["RANK"])

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    rank = rank * ngpus_per_node + local_rank
    print (f"Distributed and Multi-processing. Setting rank for each worker. rank={rank}")

    dist.init_process_group(
      backend="nccl", 
      init_method="env://",
      world_size=world_size, 
      rank=rank)

    per_device_batch_size = int(argv.batch_size / ngpus_per_node)

    training_args = TrainingArguments(
      output_dir="/tmp/output/",
      num_train_epochs=argv.epochs, 
      per_device_train_batch_size=per_device_batch_size,
      per_device_eval_batch_size=per_device_batch_size,
      local_rank=local_rank,
    )

    def compute_metrics(p: EvalPrediction):
        
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Save the trained model locally
    model_filename = "pytorch-bert-model"
    local_path = os.path.join("/tmp", model_filename)
    trainer.save_model(local_path)

    if (os.path.exists(local_path)):
        # Upload the trained model to Cloud storage
        model_directory = argv.model_dir
        storage_path = os.path.join(model_directory, model_filename)
        blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())

        files = [f for f in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, f))]

        for file in files:
            local_file = os.path.join(local_path, file)
            blob.upload_from_filename(local_file)

        print(f"Saved model files in {model_directory}/{model_filename}")


if __name__ == "__main__":
    main()

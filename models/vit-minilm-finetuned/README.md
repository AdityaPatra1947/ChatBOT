---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:2029
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: vit cutoff
  sentences:
  - How are placements at VIT?
  - Can I get an education loan?
  - What is the cutoff for VIT?
- source_sentence: study lon
  sentences:
  - Can I get an education loan?
  - Can I get an education loan?
  - How are placements at VIT?
- source_sentence: stdent loan
  sentences:
  - What is the cutoff for VIT?
  - What are the college timings?
  - Can I get an education loan?
- source_sentence: top recuiters
  sentences:
  - How are placements at VIT?
  - How are placements at VIT?
  - How are placements at VIT?
- source_sentence: cuotff for it
  sentences:
  - What is the cutoff for VIT?
  - Can I change my branch later?
  - How are placements at VIT?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'cuotff for it',
    'What is the cutoff for VIT?',
    'How are placements at VIT?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.4700, -0.0282],
#         [ 0.4700,  1.0000,  0.1177],
#         [-0.0282,  0.1177,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2,029 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                         | positive                                                                           |
  |:---------|:-------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type     | string                                                                         | string                                                                             |
  | modality | text                                                                           | text                                                                               |
  | details  | <ul><li>min: 3 tokens</li><li>mean: 5.1 tokens</li><li>max: 9 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 40.78 tokens</li><li>max: 67 tokens</li></ul> |
* Samples:
  | anchor             | positive                                                                                                                                                                                        |
  |:-------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>hi</code>    | <code>Hello! 👋 I'm the VIT Mumbai enquiry assistant.<br><br>You can ask me about **admissions, fees, documents, courses, placements, internships, hostel, scholarships** or campus life.</code> |
  | <code>hello</code> | <code>Hello! 👋 I'm the VIT Mumbai enquiry assistant.<br><br>You can ask me about **admissions, fees, documents, courses, placements, internships, hostel, scholarships** or campus life.</code> |
  | <code>hey</code>   | <code>Hello! 👋 I'm the VIT Mumbai enquiry assistant.<br><br>You can ask me about **admissions, fees, documents, courses, placements, internships, hostel, scholarships** or campus life.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 5
- `warmup_steps`: 0.1
- `use_cpu`: True
- `dataloader_pin_memory`: False
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 5
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: True
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: False
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.1562 | 10   | 1.6719        |
| 0.3125 | 20   | 1.3715        |
| 0.4688 | 30   | 1.1363        |
| 0.625  | 40   | 1.0630        |
| 0.7812 | 50   | 0.9123        |
| 0.9375 | 60   | 0.8407        |
| 1.0938 | 70   | 0.6096        |
| 1.25   | 80   | 0.5688        |
| 1.4062 | 90   | 0.4410        |
| 1.5625 | 100  | 0.5651        |
| 1.7188 | 110  | 0.4545        |
| 1.875  | 120  | 0.4762        |
| 2.0312 | 130  | 0.4332        |
| 2.1875 | 140  | 0.2871        |
| 2.3438 | 150  | 0.2528        |
| 2.5    | 160  | 0.2560        |
| 2.6562 | 170  | 0.2602        |
| 2.8125 | 180  | 0.2491        |
| 2.9688 | 190  | 0.2902        |
| 3.125  | 200  | 0.1622        |
| 3.2812 | 210  | 0.1791        |
| 3.4375 | 220  | 0.1671        |
| 3.5938 | 230  | 0.1387        |
| 3.75   | 240  | 0.1505        |
| 3.9062 | 250  | 0.1846        |
| 4.0625 | 260  | 0.1204        |
| 4.2188 | 270  | 0.1060        |
| 4.375  | 280  | 0.0897        |
| 4.5312 | 290  | 0.1679        |
| 4.6875 | 300  | 0.1089        |
| 4.8438 | 310  | 0.1191        |
| 5.0    | 320  | 0.1209        |


### Training Time
- **Training**: 8.4 minutes

### Framework Versions
- Python: 3.13.6
- Sentence Transformers: 6.1.0
- Transformers: 5.19.0
- PyTorch: 2.14.1+cpu
- Accelerate: 1.15.0
- Datasets: 5.1.0
- Tokenizers: 0.23.3

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
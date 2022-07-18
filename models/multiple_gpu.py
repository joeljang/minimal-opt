import minimal_opt
import torch
import transformers  # Just for the tokenizer!
import time
model = minimal_opt.PPOPTModel(minimal_opt.OPT_66B_CONFIG, use_cache=True)
start = time.time()
tokenizer = transformers.GPT2Tokenizer.from_pretrained(
    "facebook/opt-125m"
)
from datasets import load_dataset
from promptsource.promptsource.templates import DatasetTemplates

#Loading IMDB dataset
type_path = 'test'
dataset_name = 'imdb'
dataset_config_name = dataset_name.split('/')[1] if len(dataset_name.split('/'))!=1 else None
imdb_dataset = load_dataset(dataset_name, dataset_config_name,ignore_verifications=True)[type_path]

prompt = DatasetTemplates(
            f"{dataset_name}"
            if dataset_config_name is None
            else f"{dataset_name}/{dataset_config_name}"
        )
prompt_elem = prompt['Sentiment with choices ']
query = imdb_dataset[0]
print(query)
prompt_temp = prompt_elem.my_apply(query)[1]
input_ = prompt_elem.apply(query)[0]
output_ = prompt_elem.get_answer_choices_list(query)
print(prompt_temp)
print(input_)
print(output_)
exit()
# Takes a while? I should add a status bar. Also although it is loading shard by
# shard (not all at once), it still takes a good amount of RAM.
minimal_opt.load_sharded_weights(model, [
    "66b/reshard-model_part-0-shard0.pt",
    "66b/reshard-model_part-1-shard0.pt",
    "66b/reshard-model_part-2-shard0.pt",
    "66b/reshard-model_part-3-shard0.pt",
    "66b/reshard-model_part-4-shard0.pt",
    "66b/reshard-model_part-5-shard0.pt",
    "66b/reshard-model_part-6-shard0.pt",
    "66b/reshard-model_part-7-shard0.pt",
])
print(f'language model loaded! required time: {time.time()-start}')
with torch.inference_mode():
    text = minimal_opt.greedy_generate_text(
        model, tokenizer,
        "Large language models, which are often trained for hundreds of thousands"
        " of compute days, have shown remarkable capabilities for zero- and"
        " few-shot learning. Given their computational cost, these models are"
        " difficult to replicate without significant capital. For the few that"
        " are available through APIs, no access is granted to the full model"
        " weights, making them difficult to study. We present Open Pre-trained"
        " Transformers (OPT)",
        max_seq_len=128,
    )
    print(text)



end = time.time()
print(f'total time needed (seconds): {end-start}')
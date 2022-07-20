import minimal_opt
import torch
import transformers  # Just for the tokenizer!
model = minimal_opt.PPOPTModel(minimal_opt.OPT_66B_CONFIG, use_cache=True)
tokenizer = transformers.GPT2Tokenizer.from_pretrained(
    "facebook/opt-125m"
)
# Takes a while? I should add a status bar. Also although it is loading shard by
# shard (not all at once), it still takes a good amount of RAM.
minimal_opt.load_sharded_weights(model, [
    "opt_downloads/66b/reshard-model_part-0-shard0.pt",
    "opt_downloads/66b/reshard-model_part-1-shard0.pt",
    "opt_downloads/66b/reshard-model_part-2-shard0.pt",
    "opt_downloads/66b/reshard-model_part-3-shard0.pt",
    "opt_downloads/66b/reshard-model_part-4-shard0.pt",
    "opt_downloads/66b/reshard-model_part-5-shard0.pt",
    "opt_downloads/66b/reshard-model_part-6-shard0.pt",
    "opt_downloads/66b/reshard-model_part-7-shard0.pt",
])
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
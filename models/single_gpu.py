import minimal_opt
import torch
import transformers  # Just for the tokenizer!
model = minimal_opt.OPTModel(minimal_opt.OPT_2_7B_CONFIG, device="cuda:0", use_cache=True)
tokenizer = transformers.GPT2Tokenizer.from_pretrained(
    "facebook/opt-125m"
)
# Takes a while? I should add a status bar
minimal_opt.load_sharded_weights(model, [
    "2.7b/reshard-model_part-0.pt",
    "2.7b/reshard-model_part-1.pt",
    "2.7b/reshard-model_part-2.pt",
    "2.7b/reshard-model_part-3.pt",
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
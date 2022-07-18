import torch.nn as nn
import torch
from tqdm import auto as tqdm_lib


def greedy_generate(model: nn.Module, input_ids: torch.Tensor, max_seq_len: int,
                    verbose=True):
    """Generate greedily from OPT.

    :param model: OPTModel
    :param input_ids: token IDs [batch_size, seq_len]
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    initial_input_length = input_ids.shape[1]
    current_input_ids = input_ids
    layer_past = None
    layer_past_length = 0
    all_token_ids = input_ids.tolist()
    batch_size = len(all_token_ids)

    if verbose:
        trange = tqdm_lib.trange(initial_input_length, max_seq_len)
    else:
        trange = range(initial_input_length, max_seq_len)

    for _ in trange:
        input_length = current_input_ids.shape[1]
        model_out, layer_past = model(
            current_input_ids,
            layer_past=layer_past,
        )
        greedy_predicted_token_ids = model_out[:, -1].argmax(-1)
        current_input_ids = greedy_predicted_token_ids[:, None]
        for i in range(batch_size):
            all_token_ids[i].append(greedy_predicted_token_ids[i])
        layer_past_length += input_length
    return all_token_ids


def greedy_generate_text(model: nn.Module,
                         tokenizer,
                         initial_str: str,
                         max_seq_len: int,
                         device=torch.device("cuda:0"),
                         verbose=True):
    """Generate greedily from OPT.

    :param model: OPTModel
    :param tokenizer: OPT tokenizer (i.e. GPT-2, non-fast tokenizer)
    :param initial_str: initial string to start generation from
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param device: device to use
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    tokenized = tokenizer.encode(initial_str)
    input_ids = torch.LongTensor([tokenized]).to(device)
    all_token_ids = greedy_generate(model=model, input_ids=input_ids, max_seq_len=max_seq_len, verbose=verbose)
    return tokenizer.decode(all_token_ids[0])


def greedy_generate_classify(model: nn.Module,
                         tokenizer,
                         initial_str: str,
                         max_seq_len: int,
                         options: list,
                         device=torch.device("cuda:0"),
                         verbose=True):
    """Generate greedily from OPT.

    :param model: OPTModel
    :param tokenizer: OPT tokenizer (i.e. GPT-2, non-fast tokenizer)
    :param initial_str: initial string to start generation from
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param device: device to use
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    tokenized = tokenizer.encode(initial_str)
    input_ids = torch.LongTensor([tokenized]).to(device)
    options_tok = []
    for o in options:
        tokenized_o = tokenizer.encode(o)
        ids = torch.LongTensor([tokenized_o]).to(device)
        options_tok.append(ids)
    result_index = greedy_classify(model=model, input_ids=input_ids, max_seq_len=max_seq_len, options=options_tok, verbose=verbose)
    return result_index


def greedy_classify(model: nn.Module, input_ids: torch.Tensor, max_seq_len: int, options: list,
                    verbose=True):
    """Generate greedily from OPT.

    :param model: OPTModel
    :param input_ids: token IDs [batch_size, seq_len]
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    initial_input_length = input_ids.shape[1]
    probs = []
    for o in options:
        concat = torch.cat([input_ids,o], axis=1)
        total_length = concat.shape[1]
        outputs, _ = model(concat)
        prob = 1
        start = initial_input_length
        for i in range(total_length-initial_input_length):
            prob = prob * outputs[0][start][o[0][i]]
        probs.append(prob.item())
    return probs.index(min(probs))
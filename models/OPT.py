import minimal_opt
import torch
import transformers 
import time
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

from datasets import load_dataset
from promptsource.promptsource.templates import DatasetTemplates
from Datasets import Custom_Dataset
from torch.utils.data import DataLoader
import string
import re
import numpy as np
import random

class OPT:
    def __init__(self, configs):     
        self.configs = configs
        start = time.time()
        model_size = configs.model_size
        if model_size == '125m':
            self.model = minimal_opt.OPTModel(minimal_opt.OPT_125M_CONFIG, device = f"cuda:{configs.CUDA_VISIBLE_DEVICES}" ,use_cache=True)
        elif model_size == '1.3b':
            #self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_1_3B_CONFIG,use_cache=True)
            self.model = minimal_opt.OPTModel(minimal_opt.OPT_1_3B_CONFIG, device = f"cuda:{configs.CUDA_VISIBLE_DEVICES}" ,use_cache=True)
        elif model_size == '2.7b':
            #self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_2_7B_CONFIG ,use_cache=True)
            self.model = minimal_opt.OPTModel(minimal_opt.OPT_2_7B_CONFIG, device = f"cuda:{configs.CUDA_VISIBLE_DEVICES}" ,use_cache=True)
        elif model_size == '6.7b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_6_7B_CONFIG ,use_cache=True)
        elif model_size == '13b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_13B_CONFIG, use_cache=True)
        elif model_size == '30b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_30B_CONFIG, use_cache=True)
        elif model_size == '66b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_66B_CONFIG, use_cache=True)
        elif model_size == '175b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_175B_CONFIG, use_cache=True)
        else:
            raise Exception('Gave a model size of OPT that is not one of the options. Please select again.')
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("facebook/opt-125m")

        opt_dir = 'opt_downloads'
        shards = {
            "125m" : [f"{opt_dir}/125m/reshard-model_part-0.pt", f"{opt_dir}/125m/reshard-model_part-1.pt"],
            "1.3b": [f"{opt_dir}/1.3b/reshard-model_part-0.pt", f"{opt_dir}/1.3b/reshard-model_part-1.pt"],
            "2.7b": [f"{opt_dir}/2.7b/reshard-model_part-0.pt", f"{opt_dir}/2.7b/reshard-model_part-1.pt", f"{opt_dir}/2.7b/reshard-model_part-2.pt", f"{opt_dir}/2.7b/reshard-model_part-3.pt"],
            "6.7b": [f"{opt_dir}/6.7b/reshard-model_part-0.pt", f"{opt_dir}/6.7b/reshard-model_part-1.pt"],
            "13b": [f"{opt_dir}/13b/reshard-model_part-0.pt", f"{opt_dir}/13b/reshard-model_part-1.pt"],
            "30b": [f"{opt_dir}/30b/reshard-model_part-0.pt", f"{opt_dir}/30b/reshard-model_part-1.pt"],
            "66b" : [f"{opt_dir}/66b/reshard-model_part-0-shard0.pt",f"{opt_dir}66b/reshard-model_part-1-shard0.pt",f"{opt_dir}66b/reshard-model_part-2-shard0.pt",f"{opt_dir}66b/reshard-model_part-3-shard0.pt",f"{opt_dir}66b/reshard-model_part-4-shard0.pt",f"{opt_dir}66b/reshard-model_part-5-shard0.pt",f"{opt_dir}66b/reshard-model_part-6-shard0.pt",f"{opt_dir}66b/reshard-model_part-7-shard0.pt"
            ]
        }

        # Takes a while? I should add a status bar. Also although it is loading shard by
        # shard (not all at once), it still takes a good amount of RAM.
        minimal_opt.load_sharded_weights(self.model, shards[model_size])
        print(f'language model loaded! required time: {time.time()-start}')

    def ids_to_clean_text(self, tokenizer, generated_ids):
        gen_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            return text
        
        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def accuracy_match_score_normalize(self, prediction, ground_truth):
        if self.normalize_answer(prediction)== '' or self.normalize_answer(ground_truth)== '':
            return 0
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def get_dataset(self):
        #Loading IMDB dataset
        self.dataset = Custom_Dataset(self.configs)
    
    def evaluate_single_batch(self, prompt_name=None):
        if '/' in self.configs.dataset:
            dataset_name = self.configs.dataset.split('/')[0]
            dataset_config_name = self.configs.dataset.split('/')[1]
        else:
            dataset_name = self.configs.dataset
            dataset_config_name = dataset_name.split('/')[1] if len(dataset_name.split('/'))!=1 else None
        type_path = 'test'
        #type_path = 'validation'   
        #dataset_config_name = 'qqp'
        if 'story_cloze' in dataset_name:
            eval_dataset = load_dataset('story_cloze','2016', data_dir='data/story_cloze')[type_path]
        else:
            eval_dataset = load_dataset(dataset_name, dataset_config_name, ignore_verifications=True)[type_path]
        if self.configs.dataset != 'lambada':
            eval_dataset = eval_dataset.shuffle(seed=42)

        self.prompt = DatasetTemplates(
            f"{dataset_name}"
            if dataset_config_name is None
            else f"{dataset_name}/{dataset_config_name}"
        )
        #Selecting a single prompt
        if prompt_name!=None:
            self.prompt_elem = self.prompt[prompt_name]
        else:
            self.prompt_elem = self.prompt[self.configs.prompt_name]

        limit=100
        total_cnt=0
        correct_cnt=0
        wront_cnt=0
        total_length = len(eval_dataset)
        gap = total_length // limit
    
        for query in eval_dataset:
            result = self.prompt_elem.apply(query)
            input_ = self.tokenizer.encode(result[0], return_tensors='pt', add_special_tokens = False)
            target_ = self.tokenizer.encode(result[1], return_tensors='pt', add_special_tokens = False)
            option_lst = self.prompt_elem.get_answer_choices_list(query)
            input_length = len(input_[0])
            #print(f'input : {result[0]}')
            #print(f'ground truth: {result[1]}')
            loss_lst = []
            #print(option_lst)
            for option in option_lst:
                options_ = self.tokenizer.encode(" " + option, return_tensors='pt', add_special_tokens = False)
                #options_ = self.tokenizer.encode(option, return_tensors='pt')
                
                lm_logits, _ = minimal_opt.inference(
                        self.model,
                        input_ids=(torch.concat([input_,options_], dim=1)).cuda()
                    )  
                labels = torch.concat([input_,options_], dim=1)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().cuda()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = torch.exp(loss)
                loss_lst.append(loss.detach().cpu())
            min_index = np.argmin(loss_lst)
            #print(f'prediction {min_index}: {option_lst[min_index]}')
            #if min_index==0:
            if str(option_lst[min_index]) == str(result[1]):
                #print(f'correct!')
                correct_cnt+=1
            else:
                #print('wrong!')
                wront_cnt+=1
            total_cnt+=1
            if total_cnt==limit:
                print(f'total: {total_cnt}, correct: {correct_cnt}, wrong: {wront_cnt}')
                print(f'Percentage correct: {correct_cnt / total_cnt}')
                break
     

    def evaluate(self):
        validation_dataloader = DataLoader(self.dataset, batch_size=self.configs.batch_size, shuffle=True, num_workers=self.configs.num_workers)
        prediction_dict = {}
        answer_dict = {}
        total_cnt = 0
        accuracy_correct_num = 0

        for i, batch in enumerate(validation_dataloader):
            prob_list = []
            with torch.inference_mode():
                for index in range(len(batch["option_list"])):
                    option = batch["option_list"]
                    option_ = self.tokenizer.batch_encode_plus(option[index], max_length=self.configs.output_length,
                                                    padding=True, truncation=True, return_tensors="pt")
                    input_ = torch.concat([batch["source_ids"], option_['input_ids']], dim=1)
                    attention_masks = torch.concat([batch["source_mask"], option_['attention_mask']], dim=1)
                    attention_masks = attention_masks.bool()
                    lm_labels = option_["input_ids"].expand(len(batch['source_ids']), -1)
                    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                    #outputs, _ = minimal_opt.greedy_classify(
                    #    self.model,
                    #    input_ids=batch["source_ids"].cuda(),
                    #    attention_mask=batch["source_mask"].cuda(),
                    #    labels=lm_labels.cuda(),
                    #    decoder_attention_mask=option_["attention_mask"].cuda()
                    #)
                    outputs, _ = minimal_opt.greedy_classify(
                        self.model,
                        input_ids=input_.cuda(),
                        attention_mask=attention_masks.cuda()
                    )
                    outputs_ = torch.log_softmax(outputs[:,self.configs.input_length:], dim=-1)
                    options_ = option_["attention_mask"].cuda().unsqueeze(-1)
                    options_ = torch.tensor(options_)
                    #print(outputs_.shape)
                    #print(options_.shape)
                    logits = outputs_ * options_
                    #print(options_)
                    #logits = torch.matmul(outputs_, options_)
                    #logits = outputs_
                    lm_labels=lm_labels.cuda().unsqueeze(-1)
                    seq_token_log_prob=torch.zeros(lm_labels.shape)
                    #print(seq_token_log_prob.shape, logits.shape, lm_labels.shape)
                    for i in range(lm_labels.shape[0]):
                        for j in range(lm_labels.shape[1]):
                            seq_token_log_prob[i][j][0] = logits[i][j][lm_labels[i][j][0]]
                    seq_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
                    prob_list.append(seq_log_prob)
                concat = torch.cat(prob_list).view(-1,len(batch['source_ids']))
                predictions = concat.argmax(dim=0)
                dec = [batch["option_list"][i.item()][elem_num] for elem_num, i in enumerate(predictions)]
                
            texts = [self.tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = self.ids_to_clean_text(self.tokenizer, batch['target_ids']) 
            for i in range(len(batch['source_ids'])):
                total_cnt+=1
                ground_truth = targets[i]
                predicted = dec[i]
                print("prediction:",total_cnt,predicted)
                accuracy = self.accuracy_match_score_normalize(predicted, ground_truth)
                if accuracy == 1:
                    accuracy_correct_num+=1
                print("ground_truth", ground_truth)

                print("acc",accuracy_correct_num)
                if predicted not in prediction_dict:
                    prediction_dict[predicted] = 1
                else:
                    prediction_dict[predicted] += 1
                if ground_truth not in answer_dict:
                    answer_dict[ground_truth] = 1
                else:
                    answer_dict[ground_truth] += 1
        print(f'Number of total validation data: {total_cnt}')
        print(f'Number of correct predictions: {accuracy_correct_num}. Percentage : {accuracy_correct_num / total_cnt}')
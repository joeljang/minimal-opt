import minimal_opt
import torch
import transformers 
import time
from datasets import load_dataset
from promptsource.promptsource.templates import DatasetTemplates
from torch.utils.data import Dataset, DataLoader

class OPT:
    def __init__(self, configs):     
        start = time.time()
        model_size = configs.model_size
        if model_size == '125m':
            self.model = minimal_opt.OPTModel(minimal_opt.OPT_125M_CONFIG, device = f"cuda:{configs.CUDA_VISIBLE_DEVICES}" ,use_cache=True)
        elif model_size == '1.3b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_1_3B_CONFIG, device = configs.CUDA_VISIBLE_DEVICES ,use_cache=True)
        elif model_size == '2.7b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_2_7B_CONFIG, device = configs.CUDA_VISIBLE_DEVICES ,use_cache=True)
        elif model_size == '6.7b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_6_7B_CONFIG, device = configs.CUDA_VISIBLE_DEVICES ,use_cache=True)
        elif model_size == '13b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_13B_CONFIG, device = configs.CUDA_VISIBLE_DEVICES ,use_cache=True)
        elif model_size == '30b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_30B_CONFIG, device = configs.CUDA_VISIBLE_DEVICES ,use_cache=True)
        elif model_size == '66b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_66B_CONFIG, device = configs.CUDA_VISIBLE_DEVICES ,use_cache=True)
        elif model_size == '175b':
            self.model = minimal_opt.PPOPTModel(minimal_opt.OPT_175B_CONFIG, device = configs.CUDA_VISIBLE_DEVICES ,use_cache=True)
        else:
            raise Exception('Gave a model size of OPT that is not one of the options. Please select again.')
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("facebook/opt-125m")

        opt_dir = 'opt_downloads'
        shards = {
            "125m" : [f"{opt_dir}/125m/reshard-model_part-0.pt", f"{opt_dir}/125m/reshard-model_part-1.pt"],
            "66b" : [f"{opt_dir}/66b/reshard-model_part-0-shard0.pt",f"{opt_dir}66b/reshard-model_part-1-shard0.pt",f"{opt_dir}66b/reshard-model_part-2-shard0.pt",f"{opt_dir}66b/reshard-model_part-3-shard0.pt",f"{opt_dir}66b/reshard-model_part-4-shard0.pt",f"{opt_dir}66b/reshard-model_part-5-shard0.pt",f"{opt_dir}66b/reshard-model_part-6-shard0.pt",f"{opt_dir}66b/reshard-model_part-7-shard0.pt"
            ]
        }

        # Takes a while? I should add a status bar. Also although it is loading shard by
        # shard (not all at once), it still takes a good amount of RAM.
        minimal_opt.load_sharded_weights(self.model, shards[model_size])
        print(f'language model loaded! required time: {time.time()-start}')

    def get_dataset(self, dataset_name):
        #Loading IMDB dataset
        type_path = 'test'
        dataset_config_name = dataset_name.split('/')[1] if len(dataset_name.split('/'))!=1 else None
        self.dataset = load_dataset(dataset_name, dataset_config_name,ignore_verifications=True)[type_path]

        self.prompt = DatasetTemplates(
            f"{dataset_name}"
            if dataset_config_name is None
            else f"{dataset_name}/{dataset_config_name}"
        )

        #Selecting a single prompt
        self.prompt_elem = self.prompt['Sentiment with choices ']
    
    def inference(self):
        total = 0
        correct = 0
        with torch.inference_mode():
            for query in self.dataset:
                label = query['label']
                input_ = self.prompt_elem.apply(query)[0]
                output_list = self.prompt_elem.get_answer_choices_list(query)
                text = minimal_opt.greedy_generate_text(
                    self.model, self.tokenizer,
                    input_,
                    max_seq_len=128
                )
                result = minimal_opt.greedy_generate_classify(
                    self.model, self.tokenizer,
                    input_,
                    max_seq_len=128,
                    options=output_list
                )            
                if result==label:
                    correct+=1
                total+=1
                print(f'total :{total}, correct: {correct}, accuracy: {correct/total}')

    def sample_inference(self):
        inference_start = time.time()
        with torch.inference_mode():
            text = minimal_opt.greedy_generate_text(
                self.model, self.tokenizer,
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
        print(f'inference time: {time.time()-inference_start}')
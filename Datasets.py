from torch.utils.data import Dataset
from datasets import load_dataset
import transformers 
from promptsource.promptsource.templates import DatasetTemplates

class Custom_Dataset(Dataset):
    def __init__(self, configs):
        self.configs = configs
        dataset_name = configs.dataset
        type_path = 'test'
        dataset_config_name = dataset_name.split('/')[1] if len(dataset_name.split('/'))!=1 else None
        self.dataset = load_dataset(dataset_name, dataset_config_name,ignore_verifications=True)[type_path]
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("facebook/opt-125m")
        self.tokenizer.padding_side='left'
        self.prompt = DatasetTemplates(
            f"{dataset_name}"
            if dataset_config_name is None
            else f"{dataset_name}/{dataset_config_name}"
        )
        #Selecting a single prompt
        #self.prompt_elem = self.prompt['complete_first_then']

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        query = self.dataset[idx]
        result = self.prompt_elem.apply(query)
        input_ = result[0]
        target_ = result[1]
        option_list = self.prompt_elem.get_answer_choices_list(query)
        for i in range(len(option_list)):
            option_list[i] = " " + option_list[i]
        
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.configs.input_length,
                                                            padding='max_length', truncation=True, return_tensors="pt", add_special_tokens=True)
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.configs.output_length, 
                                                            padding='max_length', truncation=True, return_tensors="pt")
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "option_list": option_list }
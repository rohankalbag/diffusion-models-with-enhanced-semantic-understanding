## CODE ADDED BY TEAM SAROVAR (CS726) STARTS

LLAMA_model = "meta-llama/Llama-2-7b-chat-hf"
DIFF_model_name = "runwayml/stable-diffusion-v1-5"

from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pickle
import transformers
import torch
import time

import warnings
warnings.filterwarnings("ignore")

DEBUG = True

device = torch.device('cuda:4')

llama_tokenizer=AutoTokenizer.from_pretrained(LLAMA_model)

pipeline=transformers.pipeline(
    "text-generation",
    model=LLAMA_model,
    tokenizer=llama_tokenizer,
    device_map="auto",
    num_return_sequences=1,
    eos_token_id=llama_tokenizer.eos_token_id,
    return_full_text=False
)

CHECKPOINT_FREQ = 5


def get_caption_detailing_prompt(caption):
    base_prompt = '''
    Please generate the long prompt version of the short one according to the given examples. Long prompt version should consist of 3 to 5 sentences. Long prompt version must specify the color, shape, texture or spatial relation of the included objects. DO NOT generate sentences that describe the surroundings of the object!!!
    
        Short: A calico cat with eyes closed is perched upon a Mercedes.
        Long: a multicolored cat perched atop a shiny black car. the car is parked in front of a building with wooden walls and a green fence. the reflection of the car and the surrounding environment can be seen on the car's glossy surface.
    
        Short: A boys sitting on a chair holding a video game remote.
        Long: a young boy sitting on a chair, wearing a blue shirt and a baseball cap with the letter 'm'. he has a red medal around his neck and is holding a white game controller. behind him, there are two other individuals, one of whom is wearing a backpack. to the right of the boy, there's a blue trash bin with a sign that reads 'automatic party'.
    
        Short: A man is on the bank of the water fishing.
        Long: a serene waterscape where a person, dressed in a blue jacket and a red beanie, stands in shallow waters, fishing with a long rod. the calm waters are dotted with several sailboats anchored at a distance, and a mountain range can be seen in the background under a cloudy sky.
    
        Short: A kitchen with a cluttered counter and wooden cabinets.
        Long: a well-lit kitchen with wooden cabinets, a black and white checkered floor, and a refrigerator adorned with a floral decal on its side. the kitchen countertop holds various items, including a coffee maker, jars, and fruits.
    
        Short: 
        %s
    '''
    return base_prompt % caption
    
def get_detailed_caption_with_llama(caption):
    t1 = time.perf_counter()
    sequences = pipeline(get_caption_detailing_prompt(caption))
    if DEBUG: print("llama inference took:", time.perf_counter() - t1)
    return sequences[0]['generated_text'].strip(" \nLong:").split("\n")[0]


dataset_name = 'diffusers/pokemon-llava-captions'

if dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(dataset_name)
else:
    data_files = {}
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

column_names = dataset["train"].column_names

image_column = 'image'
caption_column = 'text'
dataset_size = len(dataset["train"])

try:
    with open('llama_prompts.pickle', 'rb') as handle:
        prompts = pickle.load(handle)
except:
    prompts =  {}
        
try:
    curr_prompt = 0
    for x in dataset["train"]:
        prompt = x['text']
        if prompt not in prompts.keys():
            print("--"*50)
            print(f"{curr_prompt}/{dataset_size}: small prompt:", prompt)
            print("--"*50)
            c_prompt = get_detailed_caption_with_llama(prompt)
            print("--"*50)
            print(f"{curr_prompt}/{dataset_size}: big prompt:", c_prompt)
            prompts[prompt] = c_prompt
        if curr_prompt % CHECKPOINT_FREQ == 0: 
            with open('llama_prompts.pickle', 'wb') as handle:
                pickle.dump(prompts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Saved checkpoint")
        curr_prompt += 1
            

except KeyboardInterrupt:
    with open('llama_prompts.pickle', 'wb') as handle:
        pickle.dump(prompts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved checkpoint")

## CODE ADDED BY TEAM SAROVAR (CS726) ENDS
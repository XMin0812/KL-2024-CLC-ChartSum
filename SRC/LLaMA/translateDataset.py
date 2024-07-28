from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def generate(num_batches, batch_size, model, tokenizer, dataset):
    result = []
    for idx in range(num_batches):
        start_id = idx * batch_size
        end_id = min((idx + 1) * batch_size, len(dataset))
        bach_data = dataset[start_id:end_id]
        temp = model.generate(tokenizer(bach_data, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
        result += tokenizer.batch_decode(temp, skip_special_tokens=True)
    return result

def main(batch_size, file_path_input, file_path_output):
    with open(file_path_input, "r") as f:
        dataset_en = json.load(f)
    
    instructions = []
    inputs = []
    outputs = []

    for item in dataset_en:
        instructions.append("en: " + item["instruction"])
        inputs.append("en: " + item["input"])
        outputs.append("en: " + item["output"])
        
        
    model_name = "VietAI/envit5-translation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.cuda()
        
    num_batches = (len(dataset_en) + batch_size + 1) // batch_size
    print("Number batches : ", num_batches)
    vi_instructions = generate(num_batches, batch_size, model, tokenizer, instructions)
    vi_inputs = generate(num_batches, batch_size, model, tokenizer, inputs)
    vi_outputs = generate(num_batches, batch_size, model, tokenizer, outputs)

    dataset_vn = []
    for i in range(len(vi_outputs)):
        each_json = {'instruction': vi_instructions[i].replace("vi: ", ""),
                    'input': vi_inputs[i].replace("vi: ", ""),
                    'output': vi_outputs[i].repacce("vi : ", "")}
        dataset_vn.append(each_json)

    with open(file_path_output, "w", encoding="utf-8") as file:
        json.dump(dataset_vn, file, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A script to process input file and write output to another file')
    parser.add_argument('--batch_size', type=int, help='Bach Size')
    parser.add_argument('--file_path_input', type=str, help='Path to the input JSON file')
    parser.add_argument('--file_path_output', type=str, help='Path to the output JSON file')
    args = parser.parse_args()
    
    main(args.batch_size, args.file_path_input, args.file_path_output)
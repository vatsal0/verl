import os
import re
from collections import defaultdict

import datasets

SINGLE_COT_SYSTEM_PROMPT = '''/no_think
Answer the provided question after providing a few sentences of reasoning. Your answer must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
Respond in the following format:
<reasoning>
...
</reasoning>

<answer>
...
</answer>
'''

DOUBLE_COT_SYSTEM_PROMPT = '''/no_think
Answer the provided questions after providing a few sentences of reasoning for each question. Your answer must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
Respond in the following format:
<reasoning1>
...
</reasoning1>

<answer1>
...
</answer1>

<reasoning2>
...
</reasoning2>

<answer2>
...
</answer2>
'''

ENCODED_SYSTEM_PROMPT = '''/no_think
You will be given two questions and must answer both question. You can provide a few sentences of reasoning but only pertaining to the first question. You are not allowed to discuss the second question in your reasoning. 

Your answer for each question must only contain a single integer with no additional symbols or phrases. Do not output anything other than a single integer between the <answer> and </answer> tags.
<reasoning>
...
</reasoning>

<answer1>
...
</answer1>

<answer2>
...
</answer2>
'''

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

if __name__ == "__main__":
    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    for method in ["encoded", "length_penalty", "weak_model_interp"]:

        data_dict = {
            "train": defaultdict(list),
            "test": defaultdict(list)
        }

        for split in ["train", "test"]:
            data_split = dataset[split]
            N = len(data_split)

            encoded_incorrect_ids = []
            
            for _i in range(len(encoded_incorrect_ids) if method == "encoded" and split == "train" else N):
                if method == "encoded" and split == "train":
                    i = (encoded_incorrect_ids[_i] - 1) * pow(2, -1, N) % N # want (i * 2 + 1) % N to equal ids[_i]
                else:
                    i = _i

                data_dict[split]["data_source"].append(f"{data_source}/{method}")
                data_dict[split]["ability"].append("math")

                if method == "encoded" or method == "length_penalty":
                    question_1 = data_split[(i * 2) % N]["question"]
                    question_2 = data_split[(i * 2 + 1) % N]["question"]

                    answer_1 = data_split[(i * 2) % N]["answer"]
                    answer_2 = data_split[(i * 2 + 1) % N]["answer"]

                    solution_1 = extract_solution(answer_1)
                    solution_2 = extract_solution(answer_2)

                    data_dict[split]["prompt"].append([
                        {
                            "role": "system",
                            "content": ENCODED_SYSTEM_PROMPT if method == "encoded" else DOUBLE_COT_SYSTEM_PROMPT if method == "length_penalty" else None,
                        },
                        {
                            "role": "user",
                            "content": f"Question 1:\n{question_1}\n\nQuestion 2:\n{question_2}",
                        }
                    ])
                    data_dict[split]["reward_model"].append({"style": "rule", "ground_truth": f"{solution_1}\n{solution_2}"})
                    data_dict[split]["extra_info"].append({
                        "split": split,
                        "index": _i,
                        "answer_1": answer_1,
                        "question_1": question_1,
                        "answer_2": answer_2,
                        "question_2": question_2,
                    })
                elif method == "weak_model_interp":
                    question = data_split[i]["question"]

                    answer = data_split[i]["answer"]

                    solution = extract_solution(answer)

                    data_dict[split]["prompt"].append([
                        {
                            "role": "system",
                            "content": SINGLE_COT_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": question,
                        }
                    ])
                    data_dict[split]["reward_model"].append({"style": "rule", "ground_truth": solution})
                    data_dict[split]["extra_info"].append({
                        "split": split,
                        "index": i,
                        "answer": answer,
                        "question": question,
                        "prompt": SINGLE_COT_SYSTEM_PROMPT,
                    })
                else:
                    raise NotImplementedError

        train_dataset = datasets.Dataset.from_dict(data_dict["train"])
        test_dataset = datasets.Dataset.from_dict(data_dict["test"])

        local_dir = f"~/data/{method}"

        train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

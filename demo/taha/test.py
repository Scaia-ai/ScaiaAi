import torch
from transformers import (
 BertForQuestionAnswering,
 BertTokenizer,
)


model = BertForQuestionAnswering.from_pretrained('out2')
tokenizer = BertTokenizer.from_pretrained('out2')
print('Enter the Context:')
context = input()
print('Enter your Question:')
question = input()
input_ids = tokenizer.encode(question, context)
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

print(answer)
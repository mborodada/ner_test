import re

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

examples = [
    'My name is Wolfgang and I live in Berlin',
    '1234 Mozart street',
    'I am rn at 03067 Baker street, apt.78',
    'Go to 221B Baker Street',
    '66777 Odesa str, apt. 67'
    '221B Baker Street, 111222, 34444 & 888o98. I am here! Find meh!'
]

# 1: using bert
for i in examples:
    print(f'sentence ==> {i}', f'answer ==> {nlp(i)}')

# 2: using regex
re_list = [
    # regex to catch streets like: 123 Blabla steet
    '\b\d+\s*(?:\w+\s*)+(?:street|Street|str|st|avenue|ave|road|rd|boulevard|blvd|lane|ln)\b',
    # regex to catch just regular streets:
    '\b\w+\s+(?:str(?:eet)?|avenue)\b',
    # regex to catch apartments additionaly
    '\b(?:apt(?:.)?|apartment|room)\s*\d+\b',
    # regex pattern to catch any other additional info after apartment number
    '\b\d+[A-Za-z]?\d+\b'
]

matches = []
for r in re_list:
   matches += re.findall(r, '1234 Mozart street')

print(matches)

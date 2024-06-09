# pip install transformers
# pip install tf-keras
from transformers import pipeline
classifier=pipeline("sentiment-analysis")
result=classifier("This is a good movie.")
print(result)





from transformers import pipeline
question_answerer=pipeline("question-answering")
result=question_answerer({
    "question":"What is the name of the company?",
    "context":"We created Biox Systems LTD company back in the year of 2000."
    })
print(result)




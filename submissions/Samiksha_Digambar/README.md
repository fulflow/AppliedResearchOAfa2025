Emotion > Haiku (Hugging Face)

What it does
Given a short confession, this app:
1) Detects its **emotion** using a Hugging Face model, and  
2) Generates a **haiku (5-7-5)** that matches that emotion’s tone using GPT-2.  

Hugging Face models used:
- Emotion classification: `cardiffnlp/twitter-roberta-base-emotion`
- Text generation: `gpt2` (tokenizer + causal LM)

Type in the following exanmple when prompted:
Enter your confession: I failed my exam.


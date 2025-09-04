from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


def confession_to_poem(confession: str) -> str:
 
    result = sentiment(confession)[0]
    mood = result["label"].lower()


    prompt = (
        "Write a short free-verse poem in EXACTLY 3 lines.\n"
        "Each line should be under 8 words.\n"
        f"The mood is {mood}.\n"
        f"Confession: {confession}\n"
        "Poem:\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 60,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,  
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    
    poem = text.split("Poem:", 1)[-1].strip()
 
    lines = [ln.strip() for ln in poem.splitlines() if ln.strip()]
    if len(lines) > 3:
        lines = lines[:3]
    elif len(lines) < 3:
        lines += ["..."] * (3 - len(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    print("🤖 AI Confessions — Sentiment-to-Poetry Generator")
    user_text = input("Enter your confession: ").strip()
    poem = confession_to_poem(user_text)
    print("\n✨ Generated Poem ✨\n")
    print(poem)

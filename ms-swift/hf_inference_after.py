from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch, readline

model_path = "/home/developer/workspace/ms-swift/converted_model/qwen3-8b-base-mcore-20250614-2254"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
class EndOfAssistant(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == im_end_id
stop_list = StoppingCriteriaList([EndOfAssistant()])

system_prompt = "You are a helpful assistant."

print("モデル読み込み完了")

while True:
    q = input("\n プロンプト > ").strip()
    if q.lower() in {"exit", "quit"}:
        break

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": q}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.5,
            top_k=30,
            top_p=0.8,
            max_new_tokens=256,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end_id,
            stopping_criteria=stop_list
        )

    gen_tokens = output[0, inputs.input_ids.shape[-1]:]
    reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).split("<|im_end|>")[0].strip()

    print("\n 応答:\n" + reply)


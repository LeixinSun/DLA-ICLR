import argparse
import json
import os
import re  
from openai import OpenAI
from prompts.rubric_prompt_template import build_rubric_prompt

def load_api_config(config_path, model_name):
    
    with open(config_path) as f:
        api_config = json.load(f)
    if model_name not in api_config:
        raise ValueError(f"Model '{model_name}' not found in {config_path}")
    return api_config[model_name]["model_list"][0]

def call_llm(client, prompt, model_name, temperature, max_tokens, timeout):
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Rubric Generator for CRAS")
    
    parser.add_argument("--model_name", type=str, default="ds",
                        help="Name of the model key from model_api_config.")
    parser.add_argument("--model_api_config", type=str, default="configs/api_config.json",
                        help="Path to the JSON file containing API configs.")
    parser.add_argument("--model_temperature", type=float, default=0.3,
                        help="Sampling temperature for the model.")
    parser.add_argument("--model_max_tokens", type=int, default=3054,
                        help="Maximum tokens returned by the model.")
    parser.add_argument("--model_timeout", type=int, default=1000,
                        help="Request timeout in seconds.")

    args = parser.parse_args()

    model_cfg = load_api_config(args.model_api_config, args.model_name)

    client = OpenAI(api_key=model_cfg["api_key"], base_url=model_cfg["model_url"])

    roles = ["Logician","Physicist"]
    output_dir = "Rubric_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for role in roles:
        print(f"[+] Generating rubric for: {role}")
        prompt = build_rubric_prompt(role)
        response_str = call_llm(client,
                                prompt,
                                model_name=model_cfg["model_name"],
                                temperature=args.model_temperature,
                                max_tokens=args.model_max_tokens,
                                timeout=args.model_timeout)
        
        
        clean_json_str = None
       
        match = re.search(r"```json\s*\n(.*?)\n\s*```", response_str, re.DOTALL)
        if match:
            clean_json_str = match.group(1).strip()
        else:
           
            start_index = response_str.find('{')
            end_index = response_str.rfind('}')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                 clean_json_str = response_str[start_index:end_index+1]

       
        rubric = None
        if clean_json_str:
            try:
                rubric = json.loads(clean_json_str)
            except json.JSONDecodeError:
                print(f"[!] Failed to parse extracted JSON for role {role}. Saving raw output.")
                rubric = {"role": role, "raw_output": response_str}
        
        if rubric is None: 
            print(f"[!] Could not find or parse JSON block for role {role}. Saving raw output.")
            rubric = {"role": role, "raw_output": response_str}

        with open(os.path.join(output_dir, f"rubric_{role.lower()}.json"), "w", encoding='utf-8') as f:
            json.dump(rubric, f, indent=2, ensure_ascii=False)

    print("[✓] Rubric generation complete.")

if __name__ == "__main__":
    main()
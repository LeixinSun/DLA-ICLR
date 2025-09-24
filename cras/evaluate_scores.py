import os
import json
import argparse
import re
from openai import OpenAI
from prompts.scoring_prompt_template import build_multi_dim_scoring_prompt
import time
from collections import defaultdict
from maas.utils.cost_manager import CostManager
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple

def setup_cost_logger(log_file_path):
    """
    Configure a logger specifically for recording costs.
    It only writes logs to file, not to console.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logger = logging.getLogger("maas") 
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def load_api_config(config_path, model_name):
    with open(config_path) as f:
        api_config = json.load(f)
    if model_name not in api_config:
        raise ValueError(f"Model '{model_name}' not found in {config_path}")
    return api_config[model_name]["model_list"][0]

def load_rubric(role, rubric_dir):
    if len(role) > 15:
        print(f"      ! Role name too long ({len(role)} chars), using general rubric for role: '{role[:50]}...'")
        general_path = os.path.join(rubric_dir, "rubric_general.json")
        try:
            with open(general_path, "r", encoding="utf-8") as f:
                return json.load(f)["rubric"]
        except FileNotFoundError:
            raise FileNotFoundError(f"General rubric not found in {rubric_dir}")
    
    specific_path = os.path.join(rubric_dir, f"rubric_{role.lower()}.json")
    general_path = os.path.join(rubric_dir, "rubric_general.json")
    
    try:
        with open(specific_path, "r", encoding="utf-8") as f:
            return json.load(f)["rubric"]
    except FileNotFoundError:
        try:
            with open(general_path, "r", encoding="utf-8") as f:
                print(f"      ! Using general rubric for role '{role}' (specific rubric not found)")
                return json.load(f)["rubric"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Neither specific rubric for '{role}' nor general rubric found in {rubric_dir}")

def try_extract_json(text):
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_str_match:
                json_str = json_str_match.group()
            else:
                return None
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return None

def extract_scoring_data(record: Dict[str, Any], method: str, original_query: str) -> Tuple[str, str, str]:
    """
    Extract scoring data based on different methods
    
    Args:
        record: A record from history
        method: Method name (AutoGen_Main, DyLAN_Main, DyLAN_MMLU, MacNet_Main, SelfConsistency)
        original_query: Original query
        
    Returns:
        Tuple[role, question, parsed_answer]
    """
    if method == "AutoGen_Main":
        role = record.get("role", "")
        question = original_query
        parsed_answer = record.get("content", "")
        
    elif method in ["DyLAN_Main", "DyLAN_MMLU"]:
        role = record.get("role", "")
        question = original_query
        parsed_answer = record.get("reply", "")
        
    elif method == "MacNet_Main":
        system_message = record.get("system_message", "")
        actor_prompt = record.get("actor_prompt", "")
        role = f"{system_message} {actor_prompt}".strip()
        question = original_query
        parsed_answer = record.get("actor_response", "")
        
    elif method == "SelfConsistency":
        role = record.get("role", "")
        question = original_query
        parsed_answer = record.get("content", "")
        
    else:
        role = record.get("role", "")
        question = record.get("question", original_query)
        parsed_answer = record.get("parsed_answer", "")
    
    return role, question, parsed_answer

def should_skip_round(record: Dict[str, Any], method: str) -> bool:
    """
    Determine whether this round should be skipped
    
    Args:
        record: A record from history
        method: Method name
        
    Returns:
        bool: Whether to skip
    """
    if method in ["SelfConsistency", "AutoGen_Main"]:
        round_num = record.get("round", -1)
        return round_num == 0
    
    return False

def score_single_record(client: OpenAI, model_cfg: Dict[str, Any], record: Dict[str, Any], 
                            method: str, original_query: str, rubric_dir: str, 
                            temperature: float, max_tokens: int, cost_manager: CostManager) -> Tuple[str, Any]:
    """
    Score a single record
    
    Args:
        client: OpenAI client
        model_cfg: Model configuration
        record: Single record
        method: Method name
        original_query: Original query
        rubric_dir: Rubric directory
        temperature: Temperature parameter
        max_tokens: Maximum token count
        cost_manager: Cost manager
        
    Returns:
        Tuple[role, score_result]
    """
    if should_skip_round(record, method):
        return None, None
    
    role, question, parsed_answer = extract_scoring_data(record, method, original_query)
    
    if not role or not parsed_answer:
        return None, None
    
    try:
        rubric = load_rubric(role, rubric_dir)
    except FileNotFoundError:
        print(f"      ! Rubric for role '{role}' not found. Skipping.")
        return None, None
    
    prompt = build_multi_dim_scoring_prompt(role, rubric, question, parsed_answer)
    
    try:
        response = client.chat.completions.create(
            model=model_cfg["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if response.usage:
            cost_manager.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens, model_cfg["model_name"])
            
            if cost_manager.get_total_cost() > cost_manager.max_budget:
                print("\n" + "!"*60)
                print("! CRITICAL: BUDGET EXCEEDED !")
                print(f"  - Current Total Cost: ${cost_manager.get_total_cost():.6f}")
                print(f"  - Max Budget Set:     ${cost_manager.max_budget:.6f}")
                print("! Halting the script to prevent further charges. !")
                print("!"*60 + "\n")
                sys.exit(1)
        
        content = response.choices[0].message.content.strip()
        score_dict = try_extract_json(content)
        
        if score_dict:
            score_result = score_dict
        else:
            score_result = content
            print(f"      ! Failed to parse JSON for role '{role}'. Storing raw response.")
        
        return role, score_result
        
    except Exception as e:
        score_result = {"error": f"API call failed: {e}"}
        print(f"      ! API Error for role '{role}': {e}")
        return role, score_result

def main():
    parser = argparse.ArgumentParser(description="CRAS Multi-Dimension Scorer (Incremental Write, Handles Duplicates)")
    parser.add_argument("--model_name", type=str, default="USD-guiji/deepseek-r1", help="Model key from api_config.json")
    parser.add_argument("--api_config", type=str, default="./configs/api_config.json")
    parser.add_argument("--agent_output_root", type=str, default="./agent_outputs")
    parser.add_argument("--rubric_dir", type=str, default="./Rubric_outputs")
    parser.add_argument("--output_root", type=str, default="./scores")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=10240)
    parser.add_argument("--max_budget", type=float, default=20.0, help="Maximum budget in USD to spend before halting the script.")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent API requests")
    args = parser.parse_args()

    log_file_path = os.path.join("maas", "scoring_run.log")
    setup_cost_logger(log_file_path)

    model_cfg = load_api_config(args.api_config, args.model_name)
    client = OpenAI(api_key=model_cfg["api_key"], base_url=model_cfg["model_url"])

    cost_manager = CostManager(max_budget=args.max_budget)
    print(f"CostManager initialized with a max budget of ${cost_manager.max_budget:.2f}. Starting scoring process...")
    print(f"Using concurrency level: {args.concurrency}")
    
    try:
        for dataset in os.listdir(args.agent_output_root):
            dataset_path = os.path.join(args.agent_output_root, dataset)
            if not os.path.isdir(dataset_path):
                continue
                
            for model in os.listdir(dataset_path):
                model_path = os.path.join(dataset_path, model)
                if not os.path.isdir(model_path):
                    continue
                
                for subdir in os.listdir(model_path):
                    subdir_path = os.path.join(model_path, subdir)
                    if not os.path.isdir(subdir_path):
                        continue
                    
                    json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
                    if not json_files:
                        continue
                    
                    input_dir = subdir_path
                    output_dir = os.path.join(args.output_root, dataset, model, subdir)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    scores_file_path = os.path.join(output_dir, "scores.json")
                    print(f"\nProcessing {dataset}/{model}/{subdir} -> {scores_file_path}")
                    print(f"  - Found {len(json_files)} JSON files to process")

                    for fname in sorted(json_files):
                        query_id = os.path.splitext(fname)[0]
                        
                        input_file_path = os.path.join(input_dir, fname)
                        with open(input_file_path, "r", encoding="utf-8") as f:
                            query_data = json.load(f)
                        
                        method = query_data.get("method", "")
                        original_query = query_data.get("query", "")
                        
                        roles_to_score = query_data.get("history", [])
                        num_total_roles = len(roles_to_score)

                        try:
                            with open(scores_file_path, "r", encoding="utf-8") as f:
                                scores_on_disk = json.load(f)
                            num_scores_on_disk = sum(len(v) for v in scores_on_disk.get(query_id, {}).values())
                            if num_scores_on_disk >= num_total_roles:
                                print(f"  - Skipping fully scored query: {query_id}")
                                continue
                            print(f"  + Resuming/Scoring query: {query_id} ({num_scores_on_disk}/{num_total_roles} roles scored)")
                        except (FileNotFoundError, json.JSONDecodeError):
                            scores_on_disk = {}
                            print(f"  + Scoring new query: {query_id}")
                        
                        valid_records = []
                        for record in roles_to_score:
                            if not should_skip_round(record, method):
                                valid_records.append(record)
                        
                        print(f"    - Processing {len(valid_records)} valid records for query '{query_id}'")
                        
                        if args.concurrency > 1:
                            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                                future_to_record = {
                                    executor.submit(
                                        score_single_record, 
                                        client, 
                                        model_cfg, 
                                        record, 
                                        method, 
                                        original_query, 
                                        args.rubric_dir, 
                                        args.temperature, 
                                        args.max_tokens, 
                                        cost_manager
                                    ): record for record in valid_records
                                }
                                
                                for future in as_completed(future_to_record):
                                    try:
                                        role, score_result = future.result()
                                        if role and score_result is not None:
                                            try:
                                                with open(scores_file_path, "r", encoding="utf-8") as f:
                                                    all_scores_data = json.load(f)
                                            except (FileNotFoundError, json.JSONDecodeError):
                                                all_scores_data = {}
                                            
                                            query_scores = defaultdict(list, all_scores_data.get(query_id, {}))
                                            query_scores[role].append(score_result)
                                            all_scores_data[query_id] = dict(query_scores)

                                            with open(scores_file_path, "w", encoding="utf-8") as f:
                                                json.dump(all_scores_data, f, indent=2, ensure_ascii=False)
                                            
                                            print(f"      ✓ Scored role: '{role}' for query '{query_id}'")
                                            
                                    except Exception as e:
                                        print(f"      ! Error processing record: {e}")
                        else:
                            for i, record in enumerate(valid_records):
                                role, score_result = score_single_record(
                                    client, model_cfg, record, method, original_query, 
                                    args.rubric_dir, args.temperature, args.max_tokens, cost_manager
                                )
                                
                                if role and score_result is not None:
                                    print(f"      - Scoring role: '{role}' for query '{query_id}' ({i+1}/{len(valid_records)})")
                                    
                                    try:
                                        with open(scores_file_path, "r", encoding="utf-8") as f:
                                            all_scores_data = json.load(f)
                                    except (FileNotFoundError, json.JSONDecodeError):
                                        all_scores_data = {}
                                    
                                    query_scores = defaultdict(list, all_scores_data.get(query_id, {}))
                                    query_scores[role].append(score_result)
                                    all_scores_data[query_id] = dict(query_scores)

                                    with open(scores_file_path, "w", encoding="utf-8") as f:
                                        json.dump(all_scores_data, f, indent=2, ensure_ascii=False)
                                    
                                    scores_on_disk = all_scores_data
                
                print(f"[✓] Scoring complete for {dataset}/{model}")

    finally:
        final_costs = cost_manager.get_costs()
        summary_report = (
            f"\n{'='*60}\n"
            f"Script finished or was halted. Final Cost Report:\n"
            f"  - Total Prompt Tokens:   {final_costs.total_prompt_tokens}\n"
            f"  - Total Completion Tokens: {final_costs.total_completion_tokens}\n"
            f"  - Total Estimated Cost:    ${final_costs.total_cost:.6f}\n"
            f"  - Budget Set:            ${cost_manager.max_budget:.2f}\n"
            f"{'='*60}\n"
        )
        
        print(summary_report)
        
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(summary_report)
        except Exception as e:
            print(f"\n[Warning] Could not write final summary to log file: {e}")

if __name__ == "__main__":
    main()
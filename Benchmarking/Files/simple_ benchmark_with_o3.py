import os
import json
import re
import multiprocessing
from datetime import datetime
import signal
import openai

def signal_handler(signal, frame):
    raise TimeoutError("Timeout occured!")

def run_benchmark():
    MODEL_NAME = "o3-mini"
    PROMPT_TEMPLATE = """
    Solve the following task: {prompt}
    Wrap your final code in [PYTHON]...[/PYTHON] tags. No additional commentary, no test code, just the function.
    """
    DATASET_FILE = "MbppPlus.jsonl"
    PROMPT_LIMIT = 378
    
    with open(DATASET_FILE, "r", encoding="utf-8") as dataset:
        prompts = [json.loads(line) for line in dataset if line.strip()][:PROMPT_LIMIT]
    
    API_KEY = "Nu-Uh"
    total_prompts = len(prompts)
    execution_dir = os.path.join("/home/marc/Documents/BA/MBPP", MODEL_NAME)
    timestamp = datetime.now().strftime("execution_%Y%m%d_%H%M%S")
    run_dir = os.path.join(execution_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    failed_results = []
    jsonl_file = os.path.join(run_dir, "failed_results.jsonl")
    txt_file = os.path.join(run_dir, "failed_results.txt")
    
    for index, prompt_data in enumerate(prompts, start=1):
        prompt_text = prompt_data["prompt"]
        assertion_code = prompt_data.get("assertion", "")
        print(f"Processing prompt {index}...")

        code_match = openai.ChatCompletion.create(
                model="o3-mini", # important to dinstinguish
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=prompt_text)}],
                api_key=API_KEY
            )
        code_match = code_match["choices"][0]["message"]["content"]

        code_match = re.search(r'\[PYTHON\](.*?)\[/PYTHON\]', code_match, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```python(.*?)```", code_match, re.DOTALL)
        if not code_match:
                code_match = re.search(r"```(.*?)```", code_match, re.DOTALL)
        if not code_match:
            failed_results.append({"index": index, "prompt": prompt_text, "generated_code": None, "results": "No code generated", "test_results": [], "assertion": assertion_code})
            continue

        code = code_match.group(1)
        global_context = {}
        results = []

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(5)

        try:
            exec(code, global_context, global_context)
        except Exception as e:
            failed_results.append({"index": index, "prompt": prompt_text, "generated_code": code, "results": f"Error executing code: {e}", "test_results": [], "assertion": assertion_code})
            continue
        finally:
            signal.alarm(0)

        for test in assertion_code.split("\n"):
            test = test.strip()
            if not test:
                continue

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(5)

            try:
                result = run_test(test, global_context)
                results.append(result)
            except Exception:
                results.append(f"{test} => Timeout")
            finally:
                signal.alarm(0)
                
        if "Failed" in " ".join(results) or "Error" in " ".join(results):
            failed_results.append({"index": index, "prompt": prompt_text, "generated_code": code, "results": "Executed", "test_results": results, "assertion": assertion_code})
    
    with open(jsonl_file, "w", encoding="utf-8") as jsonl_f, open(txt_file, "w", encoding="utf-8") as txt_f:
        for res in failed_results:
            json.dump(res, jsonl_f)
            jsonl_f.write("\n")
            txt_f.write(f"=== Prompt {res['index']} ===\n{res['prompt']}\n")
            txt_f.write(f"=== Generated Code ===\n{res['generated_code']}\n")
            txt_f.write("=== Test Results ===\n")
            for r in res["test_results"]:
                txt_f.write(f"{r}\n")
            txt_f.write("\n" + "=" * 80 + "\n\n")
    print(f"Benchmark completed. Failed results saved in {jsonl_file} and {txt_file}")

def run_test(test, global_context):
    try:
        match = re.match(r"assert\s+(.+?)\s*==\s*(.+)", test)
        if match:
            function_call = match.group(1).strip()
            expected_result = eval(match.group(2).strip(), global_context, global_context)
            actual_result = eval(function_call, global_context, global_context)
            return f"{test} => Passed" if actual_result == expected_result else f"{test} => Failed (Expected: {expected_result}, Got: {actual_result})"
        else:
            exec(test, global_context, global_context)
            return f"{test} => Passed"
    except Exception as e:
        return f"{test} => Error: {e}"

if __name__ == "__main__":
    run_benchmark()

import os
import json
import re
import multiprocessing
from datetime import datetime
import numpy as np
from langchain_ollama import OllamaLLM
import signal

# Unbaised pass@k formula taken from https://arxiv.org/pdf/2107.03374
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def signal_handler(signal, frame):
    raise TimeoutError("Timeout occured!")

def run_benchmark():
    MODEL_NAME = "llama2:13b"
    PROMPT_TEMPLATE = """
    Solve the following task: {prompt}
    Wrap your final code in [PYTHON]...[/PYTHON] tags. No additional commentary, no test code, just the function.
    """
    DATASET_FILE = "MbppPlus.jsonl"
    PROMPT_LIMIT = 378
    NUM_SAMPLES = 10

    with open(DATASET_FILE, "r", encoding="utf-8") as dataset:
        prompts = [json.loads(line) for line in dataset if line.strip()][:PROMPT_LIMIT]

    llm = OllamaLLM(model=MODEL_NAME)
    execution_dir = os.path.join("/home/marc/Documents/BA/MBPP", MODEL_NAME)
    timestamp = datetime.today().strftime("execution_%d-%m-%Y_%H:%M:%S")
    run_dir = os.path.join(execution_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    logging = os.path.join(run_dir, "Logging.txt")

    results_log = []
    pass_at_1_arr = []
    pass_at_3_arr = []
    pass_at_5_arr = []

    print(f"Model: {MODEL_NAME}")

    for index, prompt_data in enumerate(prompts, start=1):
        print(f"Processing prompt {index}/{PROMPT_LIMIT}")
        prompt_text = prompt_data["prompt"]
        assertion_code = prompt_data.get("assertion", "")

        correct_samples = 0

        for sample_index in range(NUM_SAMPLES):
            print(f"Generating sample {sample_index + 1}/{NUM_SAMPLES}")
            llm_output = llm.invoke(PROMPT_TEMPLATE.format(prompt=prompt_text))

            print("Prompt retrieved!")
            
            with open(logging, "a") as log_data:
                log_data.write("\n")
                log_data.write(f"Prompt {index}, Sample Number: {sample_index + 1}:\n")
                log_data.write(f"Model input: {prompt_text}")
                log_data.write(f"Response:\n {llm_output}\n")
                log_data.write("="*80)
                log_data.write("\n")
                log_data.flush()

            code_match = re.search(r'\[PYTHON\](.*?)\[/PYTHON\]', llm_output, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```python(.*?)```", llm_output, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```(.*?)```", llm_output, re.DOTALL)
            if not code_match:
                continue

            code = code_match.group(1)
            global_context = {}
            test_results = []

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(5)

            try:
                exec(code, global_context, global_context)
            except Exception:
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
                    test_results.append(result)
                except Exception:
                    test_results.append(f"{test} => Timeout")
                finally:
                    signal.alarm(0)

            if all("Passed" in result for result in test_results):
                correct_samples += 1

        pass_at_1 = pass_at_k(NUM_SAMPLES, correct_samples, 1)
        pass_at_3 = pass_at_k(NUM_SAMPLES, correct_samples, 3)
        pass_at_5 = pass_at_k(NUM_SAMPLES, correct_samples, 5)

        pass_at_1_arr.append(pass_at_1)
        pass_at_3_arr.append(pass_at_3)
        pass_at_5_arr.append(pass_at_5)

        print(f"Completed prompt {index}: pass@1={pass_at_1:.4f}, pass@3={pass_at_3:.4f}, pass@5={pass_at_5:.4f}")

        results_log.append({
            "index": index,
            "prompt": prompt_text,
            "correct_samples": correct_samples,
            "pass@1": pass_at_1,
            "pass@3": pass_at_3,
            "pass@5": pass_at_5
        })

    total_samples = len(prompts) * NUM_SAMPLES
    total_correct = sum(result["correct_samples"] for result in results_log)

    global_pass_at_1 = np.mean(pass_at_1_arr)
    global_pass_at_3 = np.mean(pass_at_3_arr)
    global_pass_at_5 = np.mean(pass_at_5_arr)

    results_file = os.path.join(run_dir, "pass_k_results.jsonl")
    global_results_file = os.path.join(run_dir, "global_pass_k_results.json")

    with open(results_file, "w", encoding="utf-8") as f:
        for result in results_log:
            json.dump(result, f)
            f.write("\n")

    global_results = {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "global_pass@1": global_pass_at_1,
        "global_pass@3": global_pass_at_3,
        "global_pass@5": global_pass_at_5
    }

    with open(global_results_file, "w", encoding="utf-8") as f:
        json.dump(global_results, f, indent=4)

    print("\n=== Final Global pass@k Results ===")
    print(f"Global pass@1: {global_pass_at_1:.4f}")
    print(f"Global pass@3: {global_pass_at_3:.4f}")
    print(f"Global pass@5: {global_pass_at_5:.4f}")

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


"""Contains the job definitions"""

import os
import re
from typing import Any, Dict, Optional, Tuple

import google.generativeai as genai

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 1.5


PROMPT = """
Create a Python function called 'dynamic_function' that implements the following logic:

{user_prompt}

The function receives the following arguments: {kwargs}

Only respond with the Python code implementing the function without including the usual if __name__ == '__main__': clause.
Do not include markdown syntax.
"""


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def clean_code(code):
    """Clean code"""
    match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
    return match.group(1) if match else code


def evaluate_code(code, **kwargs):
    """Dynamically evaluates a function defined as a string"""

    print("--------------------------------------------")
    print(f"Evaluating the following function:\n{code}\n")
    print(f"kwargs = {kwargs}")

    try:
        local_scope = {}
        exec(code, {}, local_scope)
        result = local_scope.get("dynamic_function")(**kwargs)
        print(f"Result = {result}")
        return result
    except Exception as e:
        print(f"An exception occured while evaluating the code: {e}")
        return None
    finally:
        print("--------------------------------------------")


def is_gemini_api_key_valid(gemini_api_key: str):
    """Validates whether an API key is valid"""
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(DEFAULT_MODEL)
        model.generate_content("Hello!")
        return True

    except Exception:
        return False


def dynamic_tool(
    user_prompt: str,
    gemini_api_key: Optional[str],
    temperature: float = DEFAULT_TEMPERATURE,
    **kwargs,
):
    """
    A tool that dynamically creates and evaluates LLM-generated code.

    user_prompt: a description of a function to be dynamically implemented by the LLM
    gemini_api_key: API key for Gemini
    temperature: the LLM model's temperature
    kwargs: the keyword argument the generated function is expected to take
    """

    # Model has to be temporarily fixed as the agent keeps trying to use it paid models
    model_name = DEFAULT_MODEL

    if gemini_api_key is None or not is_gemini_api_key_valid(gemini_api_key):
        gemini_api_key = os.getenv("GEMINI_API_KEY", None)

    genai.configure(api_key=gemini_api_key)

    model = genai.GenerativeModel(model_name)
    generation_config_kwargs = {"temperature": temperature}

    try:
        response = model.generate_content(
            PROMPT.format(user_prompt=user_prompt, kwargs=tuple(kwargs.keys())),
            generation_config=genai.types.GenerationConfig(
                **generation_config_kwargs,
            ),
        )
    except Exception as e:
        print(f"Gemini request failed: {e}")
        return None

    code = clean_code(response.text)
    return evaluate_code(code, **kwargs)


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    api_keys = kwargs.pop("api_keys", {})

    gemini_api_key = api_keys.get("gemini", os.environ.get("GEMINI_API_KEY"))
    if not gemini_api_key:
        return error_response("gemini_api_key was not provided")

    model_name = kwargs.pop("model", DEFAULT_MODEL)
    temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
    user_prompt = kwargs.pop("prompt", None)
    if not user_prompt:
        return error_response("Prompt was not provided")

    result = dynamic_tool(user_prompt, gemini_api_key, temperature, **kwargs)

    if result is None:
        return error_response("Code evaluation produced an exception")

    return result, None, None, None

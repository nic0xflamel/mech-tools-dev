"""Contains the job definitions"""

import functools
import importlib
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import google.generativeai as genai
import yaml
from google.api_core.exceptions import InternalServerError, ResourceExhausted

DEFAULT_TEMPERATURE = 1.5
DEFAULT_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """
Your target is the following:
{goal}

You have a selection of tools you can use to achieve your goal.
Decide what is the next tool to use and only respond with the next function call.
"""


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def rate_limit(interval: int = 5):
    """Rate limit a function call"""

    def decorator(func):
        # Needs to be a mutable to store state
        # We initialize it so we do not wait for some time before the first call
        last_called = [time.time() - interval]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            elapsed = now - last_called[0]

            if elapsed < interval:
                time.sleep(interval - elapsed)

            last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def finalize_tool():
    """This function signals the end of the execution"""


def get_local_tools():
    """Get all the local mech tools"""

    tools = [finalize_tool]

    # Get tool paths, excluding this tool
    repo_root = Path(__file__).parent.parent.parent.parent.parent

    for root, dirs, files in os.walk(os.path.join(repo_root, "packages")):
        if "component.yaml" in files and "orchestrator_tool" not in root:
            # Load component.yaml
            with open(
                os.path.join(root, "component.yaml"), "r", encoding="utf-8"
            ) as file:
                config = yaml.safe_load(file)
                script_path = os.path.join(root, config["entry_point"])
                script_path_relative = os.path.relpath(script_path, repo_root)
                module_name = os.path.splitext(script_path_relative)[0].replace(
                    os.sep, "."
                )

                # Import module, add it to tools and to the global scope
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if callable(attr) and attr_name.endswith("_tool"):
                        globals()[attr_name] = attr
                        tools.append(attr)
    return tools


@rate_limit(interval=10)
def send_message(chat, message):
    """Send a message to the chat"""
    while True:
        try:
            result = chat.send_message(message)
            return result
        except ResourceExhausted:
            print("Hit rate limit. Retrying...")
            time.sleep(30)
            pass


def orchestrate(model_name: str, goal: str, gemini_api_key: str):
    """Orchestrate all the available tools through Gemini"""

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name=model_name, tools=get_local_tools())
    chat = model.start_chat()
    response_parts = None
    result = None

    while True:
        # Receive a call request
        try:
            call_request = send_message(
                chat, response_parts or SYSTEM_PROMPT.format(goal=goal)
            )
        except InternalServerError:
            print("Exception")
            continue

        # Get the function call request
        fn = None
        for part in call_request.parts:
            fn = part.function_call
            if not fn:
                continue
            break

        if not fn:
            print("No function to call")
            break

        if fn.name == "finalize_tool":
            print(f"Execution has finalized. Result = {result}")
            break

        # Make the call
        try:
            print(f"Calling {fn.name}({dict(fn.args)})")
            method = globals().get(fn.name)
            result = method(**dict(fn.args))
        except Exception as e:
            print(f"Exception while calling the function: {e}")
            continue

        print(f"Result: {result}\n")

        # Build the response
        function_calls = {fn.name: result}

        response_parts = [
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=fn, response={"result": val}
                )
            )
            for fn, val in function_calls.items()
        ]

    return result


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""

    gemini_api_key = kwargs.get("api_keys", {}).get(
        "gemini", os.environ.get("GEMINI_API_KEY")
    )
    if not gemini_api_key:
        return error_response("GEMINI_API_KEY was not provided")

    goal = kwargs.get("goal", None)
    if not goal:
        return error_response("Goal was not provided")

    model_name = kwargs.get("model", DEFAULT_MODEL)

    result = orchestrate(model_name, goal, gemini_api_key)

    return result, None, None, None

import functools
from dune_client.client import DuneClient
import os
import re
import openai
from openai import OpenAI
from typing import Dict, Any, Optional, TypedDict, Tuple, Callable, List
from datetime import datetime
import json

# Type Definitions
class QueryDetails(TypedDict):
    query_id: int
    description: str
    row_count: int
    column_names: List[str]
    execution_time: str
    last_refresh_time: str

class APIClients:
    def __init__(self, api_keys: Any):
        self.dune_api_key = api_keys["dune"]
        self.openai_api_key = api_keys["openai"]
        
        if not all([self.dune_api_key, self.openai_api_key]):
            raise ValueError("Missing required API keys")
            
        self.openai_client = OpenAI()
        self.dune_client = DuneClient(self.dune_api_key)

def get_dune_results(clients: APIClients, query_id: int) -> Optional[Dict[Any, Any]]:
    """Fetch the latest results from a Dune query"""
    try:
        result = clients.dune_client.get_latest_result(query_id)
        
        # Convert ResultsResponse to dictionary
        if hasattr(result, 'result'):
            # Limit to 100 rows and convert to dictionary
            rows = result.result.rows[:100] if len(result.result.rows) > 100 else result.result.rows
            return {
                'result': rows,
                'metadata': {
                    'row_count': len(rows),
                    'column_names': list(rows[0].keys()) if rows else [],
                    'execution_time': result.execution_time if hasattr(result, 'execution_time') else None,
                    'last_refresh_time': result.last_refresh_time if hasattr(result, 'last_refresh_time') else None
                }
            }
        else:
            print("Invalid or empty response from Dune API")
            return None
            
    except Exception as e:
        print(f"Error fetching Dune results: {e}")
        return None

def generate_analysis(clients: APIClients, data: Dict[Any, Any], query_description: str) -> str:
    """Generate an opinionated analysis of the Dune query results using GPT"""
    try:
        result_data = data.get('result', [])
        metadata = data.get('metadata', {})
        
        prompt = f"""As a senior blockchain data analyst, analyze these on-chain metrics and provide actionable insights:

QUERY CONTEXT:
This query {query_description}

DATA SUMMARY:
- Rows Analyzed: {metadata.get('row_count', 0)}
- Metrics Available: {', '.join(metadata.get('column_names', []))}
- Last Updated: {metadata.get('last_refresh_time', 'Unknown')}

RAW DATA (Limited to 100 rows):
{json.dumps(result_data, indent=2)}

Based on this data, provide:
1. Key findings and their significance
2. Notable trends or patterns
3. Actionable insights or recommendations
4. Potential risks or limitations in the data

Focus on insights that would be valuable for investment or strategic decisions. Support your analysis with specific numbers from the data."""

        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior blockchain data analyst known for extracting actionable insights from on-chain data. Focus on patterns and implications that matter for decision-making. Be specific and cite numbers from the data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating analysis: {e}")
        return None

def extract_query_id(prompt: str) -> Optional[int]:
    """Extract Dune query ID from the prompt"""
    try:
        # Look for numbers after common patterns
        patterns = [
            r"query (\d+)",
            r"query id (\d+)",
            r"dune (\d+)",
            r"dune query (\d+)",
            r"#(\d+)",
            r"id: (\d+)",
            r"id (\d+)",
            r"(\d+)"  # fallback to any number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                return int(match.group(1))
        
        return None
        
    except Exception as e:
        print(f"Error extracting query ID: {e}")
        return None

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except openai.RateLimitError as e:
                if retries_left["openai"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                api_keys.rotate("openai")
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper

@with_key_rotation
def run(
    prompt: str,
    api_keys: Any,
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run Dune query analysis and return structured response."""
    # default response
    response = "Invalid response"
    metadata_dict = None

    try:
        clients = APIClients(api_keys)
        
        print(f"\nAnalyzing prompt: {prompt}")
        
        # Extract query ID from prompt
        query_id = extract_query_id(prompt)
        if not query_id:
            return "Could not determine Dune query ID. Please provide a valid query ID.", "", None, None
        
        # Get query results
        results = get_dune_results(clients, query_id)
        if not results:
            return f"Could not fetch results for query {query_id}. Please verify the query exists and has recent results.", "", None, None
        
        # Generate analysis
        analysis = generate_analysis(clients, results, prompt)
        if not analysis:
            return "Error generating analysis. Please try again.", "", None, None
        
        # Store all context in metadata
        metadata_dict = {
            "query_details": {
                "query_id": query_id,
                "description": prompt,
                "row_count": results['metadata']['row_count'],
                "column_names": results['metadata']['column_names'],
                "execution_time": results['metadata']['execution_time'],
                "last_refresh_time": results['metadata']['last_refresh_time']
            },
            "raw_data": {
                "results": results['result']
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Return just the analysis text as the response
        response = analysis

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e), "", None, None

    return (
        response,
        "",
        metadata_dict,
        None,
    )
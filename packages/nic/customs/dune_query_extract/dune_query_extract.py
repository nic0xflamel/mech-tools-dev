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
    question: str
    total_row_count: int
    returned_row_count: int
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
            # Store total row count before limiting
            total_rows = len(result.result.rows)
            
            # Get column names if available
            column_names = list(result.result.rows[0].keys()) if result.result.rows else []
            
            # Limit to 100 rows by default
            rows = result.result.rows[:100]
            
            return {
                'result': rows,
                'metadata': {
                    'total_row_count': total_rows,
                    'returned_row_count': len(rows),
                    'column_names': column_names,
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

def extract_specific_info(clients: APIClients, data: Dict[Any, Any], question: str) -> str:
    """Extract specific information from query results based on the question"""
    try:
        result_data = data.get('result', [])
        metadata = data.get('metadata', {})
        
        # Add warning about data limitation if necessary
        data_limitation_note = ""
        if metadata.get('total_row_count', 0) > metadata.get('returned_row_count', 0):
            data_limitation_note = f"""Note: This query contains {metadata.get('total_row_count')} rows in total, 
but only the first {metadata.get('returned_row_count')} rows are shown for analysis. 
If you need specific information that might be in the remaining rows, please refine your question."""
        
        prompt = f"""As a data analyst, extract specific information from this query result to answer the following question:

QUESTION:
{question}

DATA STRUCTURE:
Available columns: {', '.join(metadata.get('column_names', []))}
Total rows in query: {metadata.get('total_row_count', 0)}
Rows available for analysis: {metadata.get('returned_row_count', 0)}
Last updated: {metadata.get('last_refresh_time', 'Unknown')}

{data_limitation_note}

RAW DATA (Limited to {metadata.get('returned_row_count')} rows):
{json.dumps(result_data, indent=2)}

Please provide:
1. A direct answer to the question using specific numbers/values from the data
2. Only include relevant information that was asked for
3. Format numbers clearly (e.g., percentages, dollar amounts)
4. If the exact information isn't available or might be in the hidden rows, clearly state this limitation

Keep the response focused and concise. Only answer what was asked."""

        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data analyst who extracts specific information from query results. Provide direct, focused answers using only the data available. Format numbers clearly and consistently. If data is limited, acknowledge this limitation in your response."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3  # Lower temperature for more focused responses
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error extracting information: {e}")
        return None

def extract_query_details(prompt: str) -> Tuple[Optional[int], str]:
    """Extract query ID and the specific question from the prompt"""
    try:
        # Extract query ID
        id_patterns = [
            r"query (\d+)",
            r"query id (\d+)",
            r"dune (\d+)",
            r"dune query (\d+)",
            r"#(\d+)",
            r"id: (\d+)",
            r"id (\d+)",
            r"(\d+)"  # fallback to any number
        ]
        
        query_id = None
        for pattern in id_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                query_id = int(match.group(1))
                break
        
        # Extract the question by removing the query ID part
        question = prompt
        if query_id:
            for pattern in id_patterns:
                question = re.sub(pattern, "", question, flags=re.IGNORECASE).strip()
        
        # Clean up the question
        question = re.sub(r'\s+', ' ', question).strip()
        question = question.strip('?. ')
        
        return query_id, question
        
    except Exception as e:
        print(f"Error extracting query details: {e}")
        return None, ""

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
    """Extract specific information from a Dune query based on the question."""
    # default response
    response = "Invalid response"
    metadata_dict = None

    try:
        clients = APIClients(api_keys)
        
        print(f"\nProcessing question: {prompt}")
        
        # Extract query ID and question
        query_id, question = extract_query_details(prompt)
        if not query_id:
            return "Could not determine Dune query ID. Please provide a valid query ID.", "", None, None
        
        # Get query results
        results = get_dune_results(clients, query_id)
        if not results:
            return f"Could not fetch results for query {query_id}. Please verify the query exists and has recent results.", "", None, None
        
        # Extract specific information
        answer = extract_specific_info(clients, results, question)
        if not answer:
            return "Error extracting information. Please try again.", "", None, None
        
        # Store all context in metadata
        metadata_dict = {
            "query_details": {
                "query_id": query_id,
                "question": question,
                "total_row_count": results['metadata']['total_row_count'],
                "returned_row_count": results['metadata']['returned_row_count'],
                "column_names": results['metadata']['column_names'],
                "execution_time": results['metadata']['execution_time'],
                "last_refresh_time": results['metadata']['last_refresh_time']
            },
            "raw_data": {
                "results": results['result']
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Return just the specific answer as the response
        response = answer

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e), "", None, None

    return (
        response,
        "",
        metadata_dict,
        None,
    ) 
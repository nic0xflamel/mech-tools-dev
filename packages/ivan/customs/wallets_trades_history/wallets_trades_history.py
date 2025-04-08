"""
A mech tool for querying Flipside data about wallet transactions.
"""

import functools
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, Callable
from flipside import Flipside
from openai import OpenAI, RateLimitError

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

class APIClients:
    """Class for managing API clients."""
    def __init__(self, api_keys: Any):
        self.flipside_key = api_keys["flipside"]
        
        if not self.flipside_key:
            raise ValueError("Missing required Flipside API key")
            
        self.flipside = Flipside(self.flipside_key, "https://api-v2.flipsidecrypto.xyz")
        
        # Only initialize OpenAI if the key exists in api_keys
        try:
            self.openai_key = api_keys["openai"]
            self.openai = OpenAI(api_key=self.openai_key)
        except (KeyError, TypeError):
            self.openai = None

def with_key_rotation(func: Callable):
    """Decorator to handle API key rotation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except RateLimitError as e:
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

def parse_prompt(prompt: str) -> Tuple[List[str], int]:
    """Parse the prompt to extract wallet addresses and time interval."""
    # Default time interval (7 days)
    default_days = 7

    # Extract wallet addresses using regex and convert to lowercase
    wallet_pattern = r'0x[a-fA-F0-9]{40}'
    wallets = [wallet.lower() for wallet in re.findall(wallet_pattern, prompt)]
    
    # Require at least one wallet address
    if not wallets:
        raise ValueError("No valid wallet addresses found in prompt. Please provide at least one Ethereum address (0x...)")
    
    # Extract time interval
    time_patterns = {
        r'(\d+)\s*days?': 1,
        r'(\d+)\s*weeks?': 7,
        r'(\d+)\s*months?': 30,
        r'(\d+)\s*years?': 365
    }
    
    days = default_days
    for pattern, multiplier in time_patterns.items():
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            days = int(match.group(1)) * multiplier
            break
        
    return wallets, days

def generate_sql_query(wallets: List[str], days: int) -> str:
    """Generate SQL query for wallet transactions."""
    wallet_list = "('" + "','".join(wallets) + "')"
    
    return f"""
    SELECT DISTINCT
      to_varchar(amount_in_usd + amount_out_usd, '$999,999,999') AS "USD Value",
      to_varchar(amount_in_usd, '$999,999,999') AS "Amount Bought",
      symbol_in AS "Bought Symbol",
      to_varchar(amount_out_usd, '$999,999,999') AS "Amount Sold",
      symbol_out AS "Sold Symbol",
      block_timestamp AS "Time",
      trader AS "Trader",
      tx_hash AS "Transaction Hash", 
      blockchain AS "Blockchain"
    FROM crosschain.defi.ez_dex_swaps
    WHERE trader IN {wallet_list}
      AND amount_out_usd IS NOT NULL
      AND amount_in_usd IS NOT NULL
      AND block_timestamp > CURRENT_TIMESTAMP() - interval '{days} day'
    ORDER BY 1 DESC
    LIMIT 100;
    """

def format_query_results(rows: List[Any]) -> str:
    """Format query results into a readable summary."""
    if not rows:
        return "No transactions found in the specified time period."
        
    summary = []
    for row in rows:
        summary.append({
            "usd_value": row[0],
            "amount_bought": row[1],
            "bought_symbol": row[2],
            "amount_sold": row[3],
            "sold_symbol": row[4],
            "time": row[5],
            "trader": row[6],
            "tx_hash": row[7],
            "blockchain": row[8]
        })
    return summary

def generate_analysis_prompt(data: List[Dict], original_prompt: str, days: int) -> str:
    """Generate a prompt for analysis."""
    return f"""Copy trading analysis for {days}d:
1. Top traded tokens & emerging pairs
2. Trade timing & size patterns
3. Key alpha signals to monitor
4. Risk/reward ratio of strategy

Context: {original_prompt}
Data: {str(data)}

Focus on actionable signals for copy trading. What tokens should followers watch?"""

@with_key_rotation
def run(
    prompt: str,
    api_keys: Any,
    **kwargs: Any,
) -> MechResponse:
    """Run the Flipside query and return results with analysis."""
    try:
        # Initialize clients
        clients = APIClients(api_keys)
        
        # Parse wallets and days from prompt
        wallets, days = parse_prompt(prompt)
        
        # Generate SQL query
        sql = generate_sql_query(wallets, days)
        
        # Run query
        query_result = clients.flipside.query(sql)
        
        if not query_result.rows:
            return "No transactions found in the specified time period.", "", None, None
        
        # Format results
        formatted_data = format_query_results(query_result.rows)
        
        # Generate analysis if OpenAI is available
        analysis = "Analysis not available - OpenAI API key not provided"
        if clients.openai is not None:
            analysis_prompt = generate_analysis_prompt(formatted_data, prompt, days)
            analysis = clients.openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}]
            ).choices[0].message.content
        
        metadata = {
            "wallets": wallets,
            "days": days,
            "query": sql,
            "raw_data": formatted_data
        }
        
        return (
            analysis,  # Main response (analysis)
            str(query_result.rows),  # Context (raw data)
            metadata,  # Metadata
            None,  # Additional data
        )
    except Exception as e:
        return str(e), "", None, None
import functools

import openai
import requests
from typing import Dict, Optional, Tuple, Any, Callable
from openai import OpenAI
from dotenv import load_dotenv
import os
import math
import statistics


# Load environment variables
load_dotenv()


class APIClients:
    def __init__(self):
        self.synth_api_key = os.getenv("SYNTH_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        
        if not all([self.synth_api_key, self.perplexity_api_key]):
            raise ValueError("Missing required API keys in environment variables")
            
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )


def get_btc_predictions(clients: APIClients) -> Optional[list]:
    synth_api_key = clients.synth_api_key
    endpoint = "https://synth.mode.network/prediction/best"
    
    # Set up parameters
    params = {
        "asset": "BTC",
        "time_increment": 300,  # 5 minutes in seconds
        "time_length": 86400   # 24 hours in seconds
    }
    
    headers = {
        "Authorization": f"Apikey {synth_api_key}"
    }
    
    response = requests.get(endpoint, params=params, headers=headers)
    if response.status_code != 200:
        print(f"API request failed with status code: {response.status_code}")
        print(f"Response content: {response.text}")  # This will show the error message from the API
        return None
    
    data = response.json()
    if not data:  # This will catch both None and empty list/dict responses
        print(f"No predictions available for parameters: {params}")
        return None
    
    return data


def process_btc_predictions(data: list) -> tuple[float, float]:
    """Process BTC predictions to determine price direction strength and volatility.
    
    Args:
        data: List of prediction data containing simulations. Each simulation is a list
              of dictionaries with 'time' and 'price' keys.
        
    Returns:
        Tuple containing:
        - direction_score: Float between -1 and 1 indicating price direction strength
          (positive = upward, negative = downward)
        - volatility_score: Float indicating price volatility based on std deviation
    """
    simulation_scores = []
    all_hourly_changes = []
    
    # Data is a list where first element contains miner_uid and prediction list
    simulations = data[0]["prediction"]
        
    for simulation in simulations:
        start_price = simulation[0]["price"]
        hourly_scores = []
        
        # Compare start price with each hourly point (12 5-min intervals = 1 hour)
        for hour in range(1, 25):  # 24 hours
            index = min(hour * 12, len(simulation)-1)  # Ensure we don't exceed array bounds
            hourly_price = simulation[index]["price"]
            
            # Calculate percent change from start to this hour
            percent_change = (hourly_price - start_price) / start_price
            
            # Normalize to a score between -1 and 1 using tanh
            hourly_score = math.tanh(percent_change * 2)  # multiply by 2 to make the curve steeper
            
            # Apply exponential weighting based on hour
            weight = math.exp(hour / 12)  # Exponential weight increases with time
            weighted_score = hourly_score * weight
            
            hourly_scores.append(weighted_score)
            all_hourly_changes.append(percent_change)
        
        # Calculate weighted average of hourly scores
        weights = [math.exp(h / 12) for h in range(1, 25)]
        simulation_score = sum(s * w for s, w in zip(hourly_scores, weights)) / sum(weights)
        simulation_scores.append(simulation_score)
    
    # Calculate direction score as average of all simulation scores
    direction_score = sum(simulation_scores) / len(simulation_scores)
    
    # Calculate volatility using standard deviation of all hourly changes
    volatility_score = statistics.stdev(all_hourly_changes)
    
    return direction_score, volatility_score


def get_macro_outlook(clients: APIClients, prediction_data: tuple[float, float]) -> Optional[str]:
    """Generate a macro outlook analysis for Bitcoin using simulated price predictions.
    
    Args:
        clients: APIClients instance containing OpenAI client
        prediction_data: Tuple containing direction_score and volatility_score
        
    Returns:
        A string containing the macro outlook analysis, or None if analysis fails
    """
    if not prediction_data:
        return None
        
    try:
        direction_score, volatility_score = prediction_data
        
        prompt = f"""Please provide a comprehensive macro outlook report for the crypto market's short and medium term prospects.

Using the provided price direction score ({direction_score:.4f}, between -1 and 1) and price volatility score ({volatility_score:.4f}, based on the standard deviation of price changes) for Bitcoin over the next 24 hours, but without mentioning the actual numerical values in your response, please create a detailed macro outlook report that includes:

Please create a detailed macro outlook report that includes:
1. Overall price trend analysis including the price direction score
2. Key price levels and potential support/resistance zones
3. Volatility assessment including the price volatility score
4. Risk factors and potential price movement catalysts
5. Summary and conclusion

Focus on integrating the scores with CURRENT macro conditions to provide actionable insights."""

        response = clients.perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online", 
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating macro outlook analysis: {e}")
        return None


def run(
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run Bitcoin price prediction analysis and generate macro outlook report.
    
    Fetches Bitcoin price predictions, processes the data, and generates a comprehensive 
    macro outlook analysis report using AI.
    
    Args:
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tuple containing:
        - str: The macro outlook report or error message
        - Optional[str]: Empty string (unused)
        - Optional[Dict[str, Any]]: None (unused) 
        - Any: None (unused)
    """
    try:
        clients = APIClients()

        btc_predictions = get_btc_predictions(clients=clients)

        if not btc_predictions:
            return f"Failed to get BTC predictions from Synth subnet", "", None, None

        prediction_data = process_btc_predictions(btc_predictions)

        macro_outlook = get_macro_outlook(clients=clients, prediction_data=prediction_data)
        
        if not macro_outlook:
           return f"Failed to generate macro outlook for BTC", "", None, None

        return macro_outlook, "", None, None

    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return str(e), "", None, None

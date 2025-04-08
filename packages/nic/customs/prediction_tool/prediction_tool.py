from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from openai import OpenAI
import functools

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]


class APIClients:
    def __init__(self, api_keys: Any):
        self.openai_api_key = api_keys["openai"]
        self.perplexity_api_key = api_keys["perplexity"]

        if not all([self.openai_api_key, self.perplexity_api_key]):
            raise ValueError("Missing required API keys")

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key, base_url="https://api.perplexity.ai"
        )


class ResearchComponent:
    def __init__(self, clients: APIClients):
        self.clients = clients

    def _get_research(self, prompt: str) -> str:
        """Make a research query using Perplexity."""
        try:
            response = self.clients.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant focused on providing accurate, well-sourced information. For token price questions, always include current price, recent price action, and market sentiment. Be direct and concise.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in research query: {e}")
            return f"Research failed: {str(e)}"

    def research_context(self, question: str) -> str:
        """Research the general context and background of the question."""
        prompt = f"""Analyze the current context for: {question}
For token prices, include:
- Current price and market cap
- Recent price movements (7d, 30d)
- Trading volume trends
- Market sentiment indicators

For other predictions:
- Current state of affairs
- Recent significant developments
- Key market or industry trends"""
        return self._get_research(prompt)

    def research_factors(self, question: str) -> str:
        """Research key factors that could influence the outcome."""
        prompt = f"""What are the main factors that will influence: {question}

For token prices, analyze:
- Token utility and adoption metrics
- Project development activity
- Competition and market positioning
- Upcoming token events (unlocks, burns, etc.)
- Market correlation factors (BTC, ETH, sector trends)

For other predictions:
- Economic factors
- Technical developments
- Regulatory considerations
- Market dynamics"""
        return self._get_research(prompt)

    def research_dates(self, question: str) -> str:
        """Research relevant dates and timelines."""
        prompt = f"""What are the critical dates and milestones relevant to: {question}

For token prices:
- Upcoming protocol updates
- Token unlock schedules
- Partnership announcements
- Market events that could impact price
- Historical price action dates

For other predictions:
- Key upcoming events
- Development milestones
- Regulatory deadlines"""
        return self._get_research(prompt)

    def research_alternatives(self, question: str) -> str:
        """Research alternative scenarios and possibilities."""
        prompt = f"""What are the most likely scenarios for: {question}

For token prices:
- Bull case: What could drive significant upside?
- Base case: Most likely price trajectory
- Bear case: Major risks and potential downsides
Include specific price levels for each case.

For other predictions:
- Best case scenario
- Most likely outcome
- Worst case scenario"""
        return self._get_research(prompt)

    def research_existing_predictions(self, question: str) -> str:
        """Research existing predictions and expert opinions."""
        prompt = f"""What specific predictions exist for: {question}

For token prices:
- Analyst price targets
- Community sentiment and predictions
- Technical analysis forecasts
- On-chain metric projections

For other predictions:
- Expert forecasts
- Industry analysis
- Market consensus"""
        return self._get_research(prompt)


def with_key_rotation(func: Any):
    """Decorator to handle API key rotation."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


def _create_prediction_prompt(question: str, research: Dict[str, str]) -> str:
    return f"""Based on the following research, provide a detailed prediction for this question. You MUST provide specific predictions with numerical ranges, even if confidence is low.

QUESTION: {question}

RESEARCH FINDINGS:

1. Context and Background:
{research['context']}

2. Key Influencing Factors:
{research['factors']}

3. Relevant Dates and Timelines:
{research['dates']}

4. Alternative Scenarios:
{research['alternatives']}

5. Expert Predictions:
{research['existing_predictions']}

For price predictions, your response MUST include specific price ranges. Even with low confidence, provide your best estimate based on the available data.

Provide your analysis in this format:

MAIN PREDICTION:
[For prices: Specific price range with timeframe, e.g., "$X to $Y by [date]" with probability]
[For other questions: Clear prediction with probability range]

KEY FACTORS DRIVING THIS PREDICTION:
- [Most important factor]
- [Second most important factor]
- [Third most important factor]

PRICE SCENARIOS (for price predictions):
1. Optimistic Case: $[price] (probability: X%) - [key drivers]
2. Base Case: $[price] (probability: Y%) - [key drivers]
3. Pessimistic Case: $[price] (probability: Z%) - [key drivers]

ALTERNATIVE SCENARIOS (for non-price predictions):
1. Optimistic Case (probability): [description]
2. Base Case (probability): [description]
3. Pessimistic Case (probability): [description]

CONFIDENCE LEVEL: [1-10]
[If confidence is low (1-4), explain why but still provide specific predictions]

TIME HORIZON: [Specific timeframe for the prediction]

CRITICAL UNCERTAINTIES:
- [Key risk factor 1]
- [Key risk factor 2]
- [Key risk factor 3]"""


@with_key_rotation
def run(**kwargs) -> MechResponse:
    """Run the task"""
    # Initialize API clients
    clients = APIClients(kwargs["api_keys"])
    researcher = ResearchComponent(clients)

    # Get the question from prompt
    question = kwargs["prompt"]

    # Gather research
    research_results = {}
    research_results["context"] = researcher.research_context(question)
    research_results["factors"] = researcher.research_factors(question)
    research_results["dates"] = researcher.research_dates(question)
    research_results["alternatives"] = researcher.research_alternatives(question)
    research_results["existing_predictions"] = researcher.research_existing_predictions(
        question
    )

    # Generate prediction using OpenAI
    prompt = _create_prediction_prompt(question, research_results)
    try:
        response = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise analytical engine specializing in future predictions, particularly for token prices. Always provide specific numerical predictions with ranges, even with low confidence. Never avoid making a prediction - if uncertain, provide a wider range and explain the uncertainties.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        response = response.choices[0].message.content

        # Create response dictionary with all data
        metadata_dict = {
            "question": question,
            "research": research_results,
            "timestamp": datetime.now().isoformat(),
        }

        return (
            response,
            "",
            metadata_dict,
            None,
        )

    except Exception as e:
        error_dict = {
            "error": str(e),
            "question": question,
            "timestamp": datetime.now().isoformat(),
        }
        return f"Failed to generate prediction: {str(e)}", "", error_dict, None

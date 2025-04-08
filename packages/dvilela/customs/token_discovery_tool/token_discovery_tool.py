"""Contains the job definitions"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from twikit import Client
from web3 import Web3

from packages.dvilela.customs.token_discovery_tool.constants import (
    ERC20_ABI,
    UNISWAP_FACTORY_ABI,
    UNISWAP_POOL_ABI,
    UNISWAP_V2_FACTORY,
)

DEFAULT_BLOCK_RANGE = 1000
DEFAULT_LIQUIDITY_THRESHOLD = 1000
DEFAULT_DEPLOYMENT_THRESHOLD = 24

BASE_TOKEN_ADDRESES_BASE = {
    "WETH": "0x4200000000000000000000000000000000000006",
    "USDC": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
    "USDT": "0xfde4C96c8593536E31F229EA8f37b2ADa2699bb2",
}

twikit_client = Client(language="en-US")


def tweet_to_json(tweet: Any, user_id: Optional[str] = None) -> Dict:
    """Tweet to json"""
    return {
        "id": tweet.id,
        "user_name": tweet.user.name,
        "user_id": user_id or tweet.user.id,
        "text": tweet.text,
        "created_at": tweet.created_at,
        "view_count": tweet.view_count,
        "favorite_count": tweet.favorite_count,
        "retweet_count": tweet.retweet_count,
        "quote_count": tweet.quote_count,
        "view_count_state": tweet.view_count_state,
    }


def get_eth_price():
    """Get the current price of Ethereum"""
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    response = requests.get(url)
    return response.json()["ethereum"]["usd"]


def find_token_age(
    web3, contract_address, block_range=DEFAULT_BLOCK_RANGE
) -> Optional[int]:
    """Find the time when a contract was created"""

    creation_block = None

    # Search in the latest 5k blocks
    end_block = web3.eth.block_number
    start_block = end_block - block_range

    while start_block <= end_block:
        mid = (start_block + end_block) // 2
        code = web3.eth.get_code(contract_address, block_identifier=mid)

        # The contract was not created yet
        if code == b"" or code.hex() == "0x":
            start_block = mid + 1
        else:
            # The contract was created here or before
            creation_block = mid
            end_block = mid - 1

    if not creation_block:
        return None

    creation_timestamp = web3.eth.get_block(creation_block)["timestamp"]
    token_age_hours = (datetime.now().timestamp() - creation_timestamp) / 3600
    return token_age_hours


def get_token_info(web3, token_address) -> Optional[Dict[str, Any]]:
    """Get token information"""
    try:
        contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)
        return {
            "address": token_address,
            "symbol": contract.functions.symbol().call(),
            "decimals": contract.functions.decimals().call(),
        }
    except Exception:
        return None


def analyze_liquidity(
    web3, pool_address, token_0_info, token_1_info
) -> Optional[float]:
    """Analyze liquidity of a pool"""
    pool_contract = web3.eth.contract(address=pool_address, abi=UNISWAP_POOL_ABI)

    base_token_address = (
        token_0_info["address"]
        if token_0_info["address"] in BASE_TOKEN_ADDRESES_BASE.values()
        else token_1_info["address"]
    )
    base_is_weth = base_token_address == BASE_TOKEN_ADDRESES_BASE["WETH"]

    try:
        reserves = pool_contract.functions.getReserves().call()
        reserve0 = reserves[0] / (10 ** token_0_info["decimals"])

        if base_is_weth:
            eth_price = get_eth_price()
            liquidity = (reserve0 * eth_price) * 2  # Total liquidity in USD
        else:
            liquidity = reserve0 * 2  # Asumes the stablecoin is worth 1 USD

        return liquidity

    except Exception:
        return 0


def find_new_tokens(
    web3,
    block_range: int = DEFAULT_BLOCK_RANGE,
    liquidity_threshold: float = DEFAULT_LIQUIDITY_THRESHOLD,
    deployment_threshold: int = DEFAULT_DEPLOYMENT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Analyze newly deployed pools and find new tokens"""
    factory = web3.eth.contract(address=UNISWAP_V2_FACTORY, abi=UNISWAP_FACTORY_ABI)
    latest_block = web3.eth.block_number
    pool_created_logs = factory.events.PairCreated.get_logs(
        from_block=web3.eth.block_number - block_range, to_block=latest_block
    )
    print(f"Found {len(pool_created_logs)} new pools in the last {block_range} blocks")

    if not pool_created_logs:
        return None

    BASE_ADDRESSES = list(BASE_TOKEN_ADDRESES_BASE.values())

    new_tokens = []

    for log in pool_created_logs:
        token0_address = log.args.token0
        token1_address = log.args.token1
        pool_address = log.args.pair

        token_0_info = get_token_info(web3, token0_address)
        token_1_info = get_token_info(web3, token1_address)

        # Ignore tokens with missing information
        if not token_0_info or not token_1_info:
            print(f"Token info not found for {token0_address} or {token1_address}")
            continue

        liquidity = analyze_liquidity(web3, pool_address, token_0_info, token_1_info)

        # Ignore tokens with low liquidity
        if liquidity < liquidity_threshold:
            print(
                f"Ignoring pool with low liquidity [{token_0_info['symbol']}/{token_1_info['symbol']}]: ${liquidity}"
            )
            continue

        print(
            f"Pool [{token_0_info['symbol']}/{token_1_info['symbol']}] has enough liquidity ({liquidity} >= {liquidity_threshold})"
        )

        # Check if the token is paired with a base token and is less than 24 hours old
        if token_0_info["address"] in BASE_ADDRESSES:
            if find_token_age(web3, token_1_info["address"]) < deployment_threshold:
                new_tokens.append(token_1_info | {"liquidity": liquidity})
            else:
                print(
                    f"Token {token_1_info['symbol']} was deployed more than {deployment_threshold} hours ago. Ignoring."
                )

        if token_1_info["address"] in BASE_ADDRESSES:
            if find_token_age(web3, token_0_info["address"]) < 24:
                new_tokens.append(token_0_info | {"liquidity": liquidity})
            else:
                print(
                    f"Token {token_0_info['symbol']} was deployed more than {deployment_threshold} hours ago. Ignoring."
                )

    print(f"Found {len(new_tokens)} new tokens with enough liquidity")
    return new_tokens


async def get_tweets(token_name) -> Optional[List]:
    """Get recent tweets about a token"""
    token_name = token_name if token_name.startswith("$") else f"${token_name}"
    try:
        tweets = await twikit_client.search_tweet(
            f"{token_name} -is:retweet", product="Top", count=100
        )
        return [tweet_to_json(t) for t in tweets]
    except Exception as e:
        print(f"Exception while getting the tweets: {e}")
        return None


async def is_popular(token_symbol) -> Optional[bool]:
    """
    Check if a token is popular based on the number of likes, retweets and quotes
    """
    tweets = await get_tweets(token_symbol)

    if tweets is None:
        return None

    total_tweets = len(tweets)
    total_likes = sum(tweet["favorite_count"] for tweet in tweets)
    total_retweets = sum(tweet["retweet_count"] for tweet in tweets)
    total_quotes = sum(tweet["quote_count"] for tweet in tweets)

    return (
        total_tweets > 100
        or total_likes > 1000
        or total_retweets > 1000
        or total_quotes > 100
    )


async def twikit_login(twitter_credentials: str):
    """Login into Twitter"""

    twitter_credentials = json.loads(twitter_credentials)

    with tempfile.TemporaryDirectory() as temp_dir:
        cookies = twitter_credentials["cookies"]
        cookies_path = Path(temp_dir) / "twikit_cookies.json"
        with open(cookies_path, "w", encoding="utf-8") as f:
            json.dump(cookies, f)

        await twikit_client.login(
            auth_info_1=twitter_credentials["email"],
            auth_info_2=twitter_credentials["user"],
            password=twitter_credentials["password"],
            cookies_file=str(cookies_path),
        )
        print("Logged into Twitter")


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


def discover_tokens_tool(
    rpc: Optional[str] = None,
    twitter_credentials: Optional[str] = None,
    block_range: int = DEFAULT_BLOCK_RANGE,
    liquidity_threshold: float = DEFAULT_LIQUIDITY_THRESHOLD,
    deployment_threshold: int = DEFAULT_DEPLOYMENT_THRESHOLD,
):
    """
    Searches for newly deployed ERC-20 tokens.

    rpc: and rpc to connect to a blockchain
    twitter_credentials: a dictionary containing twitter credentials
    block_range: the number of blocks to parse for newly deployed pools
    liquidity_threshold: the min liquidity (in dollars) of a pool to be considered liquid enough
    deployment_threshold: the max age (in hours) of a token since its deployment for it to be considered
    """

    # Use public RPC for Base if none provided
    if rpc is None or rpc == "...":
        rpc = "https://base.publicnode.com"

    if twitter_credentials is None or twitter_credentials == "...":
        twitter_credentials = os.getenv("TWITTER_CREDENTIALS", None)

    block_range = int(block_range)
    liquidity_threshold = int(liquidity_threshold)
    deployment_threshold = int(deployment_threshold)

    # Get tokens
    web3 = Web3(Web3.HTTPProvider(rpc))
    new_tokens = find_new_tokens(
        web3, block_range, liquidity_threshold, deployment_threshold
    )

    # Check popularity on Twitter
    if new_tokens and twitter_credentials:
        print("Checking popularity on Twitter")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(twikit_login(twitter_credentials))

        for token in new_tokens:
            token["is_popular"] = loop.run_until_complete(is_popular(token["symbol"]))
            print(f"Is {token['symbol']} popular? {token['is_popular']}")

    return new_tokens


def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Searches for newly deployed ERC-20 tokens"""

    # Use public RPC for Base
    rpc = "https://base.publicnode.com"

    # Twitter credentials
    twitter_credentials = kwargs.get("api_keys", {}).get("twitter", None)

    block_range = kwargs.get("block_range", DEFAULT_BLOCK_RANGE)
    liquidity_threshold = kwargs.get("liquidity_threshold", DEFAULT_LIQUIDITY_THRESHOLD)
    deployment_threshold = kwargs.get(
        "deployment_threshold", DEFAULT_DEPLOYMENT_THRESHOLD
    )

    new_tokens = discover_tokens_tool(
        rpc, twitter_credentials, block_range, liquidity_threshold, deployment_threshold
    )

    # Format the response as a string
    if not new_tokens:
        response = "No new tokens found matching the criteria."
    else:
        response = f"Found {len(new_tokens)} new tokens:\n\n"
        for token in new_tokens:
            response += f"- {token['symbol']}: {token['address']}\n"
            response += f"  Liquidity: ${token['liquidity']:.2f}\n"
            if 'is_popular' in token:
                response += f"  Popular on Twitter: {token['is_popular']}\n"
            response += "\n"

    # Create metadata dictionary with structured data
    metadata_dict = {
        "discovery_details": {
            "block_range": block_range,
            "liquidity_threshold": liquidity_threshold,
            "deployment_threshold": deployment_threshold,
            "rpc_endpoint": rpc,
        },
        "raw_data": {
            "tokens": new_tokens
        },
        "timestamp": datetime.now().isoformat()
    }

    return (
        response,
        "",
        metadata_dict,
        None,
    )

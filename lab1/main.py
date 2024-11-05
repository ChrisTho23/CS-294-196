import os
import sys
from typing import Dict, List

import numpy as np
from autogen import ConversableAgent
from dotenv import load_dotenv

load_dotenv()

data_path = "/Users/christho/Private/CS-294-196/lab1/restaurant-data.txt"

SCORE_MAP = {
    "awful": 1,
    "horrible": 1, 
    "disgusting": 1,
    "bad": 2,
    "unpleasant": 2,
    "offensive": 2,
    "average": 3,
    "uninspiring": 3,
    "forgettable": 3,
    "good": 4,
    "enjoyable": 4,
    "satisfying": 4,
    "awesome": 5,
    "incredible": 5,
    "amazing": 5
}

def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    # This function takes in a restaurant name and returns the reviews for that restaurant. 
    # The output should be a dictionary with the key being the restaurant name and the value being a list of reviews for that restaurant.
    # The "data fetch agent" should have access to this function signature, and it should be able to suggest this as a function call. 
    # Example:
    # > fetch_restaurant_data("Applebee's")
    # {"Applebee's": ["The food at Applebee's was average, with nothing particularly standing out.", ...]}
    reviews = []
    with open(data_path, "r") as db:
        for line in db:
            if (line.split(".")[0]).lower() == restaurant_name.lower():
                reviews.append([".".join(line.split(".")[1:])[1:-2]])

    return {
        restaurant_name: reviews
    }


def calculate_overall_score(restaurant_name: str, food_scores: List[int], customer_service_scores: List[int]) -> Dict[str, float]:
    # TODO
    # This function takes in a restaurant name, a list of food scores from 1-5, and a list of customer service scores from 1-5
    # The output should be a score between 0 and 10, which is computed as the following:
    # SUM(sqrt(food_scores[i]**2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    # The above formula is a geometric mean of the scores, which penalizes food quality more than customer service. 
    # Example:
    # > calculate_overall_score("Applebee's", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    # {"Applebee's": 5.048}
    # NOTE: be sure to that the score includes AT LEAST 3  decimal places. The public tests will only read scores that have 
    # at least 3 decimal places.
    N = len(food_scores)
    avg_score = sum([np.sqrt(food_scores[i]**2 * customer_service_scores[i]) for i in range(N)]) * 1/(N * np.sqrt(125)) * 10
    return {
        restaurant_name: avg_score
    }

def get_all_restaurant_names() -> str:
    names = []
    with open(data_path, "r") as db:
        for line in db:
            names.append(line.split(".")[0])

    return " ".join(list(set(names)))

# Do not modify the signature of the "main" function.
def main(user_query: str):
    entrypoint_agent_system_message = "You are a restaurant reviewer. Based on an input query, "
    "query a database for reviews of the restaurant in question and evaluate it based on these reviews."
    # example LLM config for the entrypoint agent
    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}
    # the main entrypoint/supervisor agent
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent", 
        system_message=entrypoint_agent_system_message, 
        llm_config=llm_config
    )
    entrypoint_agent.register_for_llm(
        name="fetch_restaurant_data", 
        description="Fetches the reviews for a given restaurant nane."
    )(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)

    available_names = get_all_restaurant_names()
    data_fetch_agent_system_message = (
        "Fetch the reviews for the restaurant named in the query. "
        f"If the restaurant name from the query is not in this list of available restaurants: {available_names}, "
        "correct the spelling to the closest matching restaurant name. "
        "You must use the fetch_restaurant_data function to get the reviews."
    )
    data_fetch_agent = ConversableAgent(
        "data_fetch_agent",
        system_message=data_fetch_agent_system_message,
        llm_config=llm_config
    )
    data_fetch_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)

    review_scorer_agent_system_message = (
        "You are a scoring bot that MUST analyze each review separately and return scores in two lists.\n"
        "For EACH review, find exactly two keywords:\n"
        "1. One keyword describing food quality\n"
        "2. One keyword describing customer service\n\n"
        "Convert each keyword to a score (1-5) using these exact mappings:\n"
        "- Score 1: awful, horrible, disgusting\n"
        "- Score 2: bad, unpleasant, offensive\n"
        "- Score 3: average, uninspiring, forgettable\n"
        "- Score 4: good, enjoyable, satisfying\n"
        "- Score 5: awesome, incredible, amazing\n\n"
        "Return your results in this EXACT format:\n"
        "food_scores: [score1, score2, ...]\n"
        "customer_service_scores: [score1, score2, ...]\n\n"
        "Example input: ['The food was average. The service was unpleasant.']\n"
        "Example output:\n"
        "food_scores: [3]\n"
        "customer_service_scores: [2]\n\n"
        "Example input with multiple reviews:\n"
        "['Food was good, service horrible.', 'Food awful, service enjoyable.']\n"
        "Example output:\n"
        "food_scores: [4, 1]\n"
        "customer_service_scores: [1, 4]"
    )
    review_analysis_agent = ConversableAgent(
        "review_analysis_agent",
        system_message=review_scorer_agent_system_message,
        llm_config=llm_config
    )

    scorer_agent_system_message = "Compute the average score of a given restaurant based on the "
    "food score and the customer service score lists using the calculate_overall_score function."
    scorer_agent = ConversableAgent(
        "scorer_agent",
        system_message=scorer_agent_system_message,
        llm_config=llm_config
    )
    scorer_agent.register_for_llm(
        name="calculate_overall_score", 
        description="Calculate overall review score based off food and service quality scores."
    )(calculate_overall_score)

    chat_results = entrypoint_agent.initiate_chats(
        [
            {
                "recipient": data_fetch_agent,
                "message": user_query,
                "max_turns": 2,
                "summary_method": "last_msg"
            },
            {
                "recipient": review_analysis_agent,
                "message": "Score the food and customer service of the restaurant.",
                "max_turns": 1,
                "summary_method": "last_msg"
            },
            {
                "recipient": scorer_agent,
                "message": "Compute the average score of the restaurant.",
                "max_turns": 2,
                "summary_method": "last_msg"
            }
        ]
    )
    
# DO NOT modify this code below.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please ensure you include a query for some restaurant when executing main."
    main(sys.argv[1])

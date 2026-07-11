 
import os
import re
import json
from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS

load_dotenv()
# before running this code make sure to set the GROQ_API_KEY in your environment variables or in a .env file.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")
 

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        return "Error in calculation"


def extract_keywords(text: str) -> list:
    """Extract keywords from text."""
    try:
        words = text.split()
        keywords = list(set([w.lower() for w in words if len(w) > 4]))
        return keywords[:5]
    except Exception:
        return []
 
# a tool for real time info with duckduckgo_search 

def duckduckgo_search(query: str) -> str:
    """Perform a DuckDuckGo search and return the top result."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=1)
        if results:
            return results[0]['body']
        else:
            return "No results found."
    except ImportError:
        return "DuckDuckGo search tool is not available. Please install 'duckduckgo_search' package."
    
 
tools = {
    "calculator": {
        "func": calculator,
        "description": "Useful for evaluating mathematical expressions. Input should be a valid math expression (e.g., '20 + 5')."
    },
    "keyword_extractor": {
        "func": extract_keywords,
        "description": "Useful for extracting important keywords from a piece of text. Input is the text to be processed."
    },
    
    "duckduckgo_search": {
        "func": duckduckgo_search,
        "description": "Useful for performing a DuckDuckGo search to get real-time information. Input is the search query."
    }
    
}

groq_client = Groq(api_key=GROQ_API_KEY)


def agent(query: str):
    tool_descriptions = "\n".join([f"- {name}: {details['description']}" for name, details in tools.items()])
    tool_names = ", ".join(tools.keys())

    current_prompt = f"""
                You are a helpful assistant that thinks step-by-step to answer user queries. You have access to the following tools:

                {tool_descriptions}

                Your response MUST be a JSON object with the following schema:
                {{
                "thought": "Your reasoning and plan on what to do next.",
                "action": "The action to take. Must be one of [{tool_names}] or 'final_answer'.",
                "action_input": "The input for the chosen action. For 'final_answer', this is your response to the user."
                }}

                History of previous actions and observations will be provided. Now, answer the user's query.
                ---
                User Query: {query} 
                """

    try:
      
        max_iterations = 5
        for _ in range(max_iterations):
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user", 
                        "content": current_prompt,
                    }
                ],
                model="openai/gpt-oss-120b",
            )
            text_response = chat_completion.choices[0].message.content

            try:
                # Extract JSON from the response, handling markdown code blocks
                json_match = re.search(r'```json\n(.*)\n```', text_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text_response
                
                parsed_response = json.loads(json_str)
                action = parsed_response.get("action")
                action_input = parsed_response.get("action_input", "")

            except (json.JSONDecodeError, AttributeError):
                # If the model fails to return valid JSON, treat it as a general response
                return {"type": "general", "result": text_response.strip()}

            if action == "final_answer":
                return {"type": "final_answer", "result": action_input}

            if action in tools:
                tool_function = tools[action]["func"]
                observation = tool_function(action_input)
                
                # Append the thought, action, and observation to the prompt for the next iteration
                current_prompt += f"""
                    Thought: {parsed_response.get('thought')}
                    Action: {action}
                    Action Input: {action_input}
                    Observation: {observation}
                    """
            else:
                current_prompt += f"\nObservation: Unknown action '{action}'. Please choose from [{tool_names}].\n"

        return {"type": "error", "result": "Agent failed to reach a conclusion after maximum iterations."}

    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        return {"type": "error", "result": "Sorry, I encountered an internal error."}


queries = [
    # "Calculate 20 + 5",
    # "Extract keywords from 'Artificial Intelligence is transforming industries'",
    # "What is machine learning?",
    "what is the latest news on AI advancements?",
]

for q in queries:
    print("Query:", q)
    print("Response:", agent(q))
    print("-" * 50)

# 🎯 Interactive Mode

while True:
    user_input = input("Enter query (type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    print("Response:", agent(user_input))
    print("-" * 50)
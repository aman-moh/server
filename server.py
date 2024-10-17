from flask import Flask, jsonify
from flask_cors import CORS
import nest_asyncio
from camel.agents.chat_agent import ChatAgent
from camel.configs.openai_config import ChatGPTConfig
from camel.messages.base import BaseMessage
from camel.models import ModelFactory
from camel.tasks.task import Task
from camel.toolkits import (
    OpenAIFunction,
    SearchToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.workforce import Workforce
import json
import traceback
import logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)

nest_asyncio.apply()

@app.route('/api/home', methods=['GET'])
def return_home():
    try:
        transcript = """Hacker funds can be so powerful for growth. Here's what you need to know about hosting hackathons. After hosting my first one, I want to break down what happened and three valuable lessons that I wish on you before. As we did make some mistakes, but in the end, I think it ended up quite unique. Okay, so before I break down the key insights, I want to very, very quickly go over the hackathon just so you've got a little bit more context to what I'm talking about. So the title was Multi-Agent System Solutions, Kamu AI Hacker Fund. So the main theme was, because Kamu is a Multi-Agent System Framework, they're going to create solutions with this technology. Amongst this, they're going to work with our sponsors, someone over, FyreKrill, and Quadrant, to create these solutions with. And the general idea was, we're going to start with some talks from different speakers. I've got to say an massive thank you to the full-the hackathon starts. And then the last thing I want to go over, so we actually base in London, but we actually did this for a work trip to San Francisco. And now you know that I'm going to go over the first insight, which might seem kind of obvious, but I'm going to say anyway, preparation is everything. So really think what goes involved into hosting one of these things, we've got to start our deep-end view, we've got to start our post-films, content to monitor, invite speakers, set up food, start the documentation, and of course much else."""
        
        result = perform_analysis(transcript)
        
        response = jsonify({
            "hook_script": result
        })
        return response
    except Exception as e:
        logging.error(f"Error in return_home: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def perform_analysis(transcript):
    search_toolkit = SearchToolkit()
    search_tools = [
        OpenAIFunction(search_toolkit.search_google),
        OpenAIFunction(search_toolkit.search_duckduckgo),
    ]

    # Create worker agents
    hook_segmentor = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="hook_segmentor",
            content="You are to find the hook of the transcript provided, your output should just be the hook_script and nothing else",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig().as_dict(),
        ),
        tools=search_tools,
    )

    hook_classifier = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="hook_age_classifier",
            content="you need to classify the reading age of the hook and nothing else",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig().as_dict(),
        ),
        tools=search_tools,
    )

    hook_type_classifier = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="hook_type_classifier",
            content="you need to classify the type of the hook and nothing else"
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig().as_dict(),
        ),
    )

    proof_checker_agent = ChatAgent(
        BaseMessage.make_assistant_message(
            role_name="Proof checker agent",
            content="You are you to proof check this tweet",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig().as_dict(),
        ),
    )

    # Create Workforce & Add Agents
    workforce = Workforce('packaging_analysis_workforce')

    workforce.add_single_agent_worker(
        "Proof checker agent, an agent that can check for grammer and spelling mistkes in tweets",
        worker=proof_checker_agent,
    ).add_single_agent_worker(
        "hook_classifier, this agent classifys parts of a hook", worker=hook_classifier
    ).add_single_agent_worker(
        "An agent who finds the hook in transcripts", worker=hook_segmentor
    ).add_single_agent_worker(
        "An agent who finds the type hook in transcripts", worker=hook_type_classifier
    )

    # Create Task & Assign to Workforce
    human_task = Task(
        content=(
            "You are to do deep analysis on the transcript of a social media video. "
            "You are to first find the hook of the video and then you should get insights on the hook: (hook type, reading age and hook length). "
            "The output should be a JSON formatted exactlylike this but in JSON format,:\n\n"
            "{\n"
            "  \"worker_node\": 139550241668976,\n"
            "  \"task\": \"assessing the length of hooks to evaluate their effectiveness and engagement potential\",\n"
            "  \"hook_script\": {\n"
            "    \"content\": \"Hacker funds can be so powerful for growth. Here's what you need to know about hosting hackathons.\",\n"
            "    \"sentences\": [\n"
            "      {\n"
            "        \"sentence\": \"Hacker funds can be so powerful for growth.\",\n"
            "        \"length_in_words\": 8\n"
            "      },\n"
            "      {\n"
            "        \"sentence\": \"Here's what you need to know about hosting hackathons.\",\n"
            "        \"length_in_words\": 9\n"
            "      }\n"
            "    ],\n"
            "    \"total_length_in_words\": 17\n"
            "  }\n"
            "}"
        ),
        additional_info=transcript,
        id='0',
    )

    task = workforce.process_task(human_task)
    result = task.result

    # Try to parse the result as JSON
    try:
        result_json = json.loads(result)
        hook_script = result_json.get("hook_script", {})
    except json.JSONDecodeError:
        # If parsing fails, assume the entire result is the hook script content
        hook_script = {
            "content": result,
            "sentences": [],
            "total_length_in_words": len(result.split())
        }

    return hook_script

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

import uuid
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


model = ChatOpenAI(model="gpt-4o",temperature=0)

def generate_topic(state: State):
    """LLM call to generate a topic for the joke"""
    msg = model.invoke("Give me a funny topic for a joke")
    return {"topic": msg.content}


def write_joke(state: State):
    """LLM call to write a joke based on the topic"""
    msg = model.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_topic", generate_topic)
workflow.add_node("write_joke", write_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_topic")
workflow.add_edge("generate_topic", "write_joke")
workflow.add_edge("write_joke", END)

# Compile
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
graph
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}
state = graph.invoke({}, config)
print("Final state:")
print(state["topic"])
print()
print(state["joke"])



# The states are returned in reverse chronological order.
states = list(graph.get_state_history(config))
print("State history:")
for state in states:
    print(state.next)
    print(state.config["configurable"]["checkpoint_id"])
    print()

# This is the state before last (states are listed in chronological order)
selected_state = states[1]
print("Selected state:")
print(selected_state.next)
print(selected_state.values)

# Update the topic in the selected state and re-run the workflow from that point.
new_config = graph.update_state(selected_state.config, values={"topic": "chickens"})
print(new_config)

state = graph.invoke(None, new_config)
print("Final state after update:")
print(state["topic"])
print()
print(state["joke"])
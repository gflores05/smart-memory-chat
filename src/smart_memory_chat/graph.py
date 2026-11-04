import sqlite3
from typing import Annotated, TypedDict

from langchain.messages import SystemMessage
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_typesafe import Noul, TypeSafeClassifier
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from pydantic import SecretStr

from smart_memory_chat.settings import Settings


class ChatBotState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


class ChatBotGraph:
  def __init__(self, settings: Settings):
    self.llm = ChatOpenAI(
      api_key=SecretStr(settings.openai_api_key),
      model=settings.ai_model,
      temperature=settings.ai_model_temperature,
    )
    self.classifier = TypeSafeClassifier(api_key=SecretStr(settings.type_safe_api_key))
    self.graph: StateGraph | None = None
    self.conn = sqlite3.connect(settings.database_name, check_same_thread=False)

  def require_filter(self, state: ChatBotState) -> str:
    """If we have more than 8 messages it will filter them based on their importance."""
    if len(state["messages"]) > 8:
      return "filter"
    return "continue"

  def filter_relevant_messages(self, state: ChatBotState):
    """Filter messages to keep only the most relevant ones."""
    messages = state["messages"]

    relevant_messages = []
    classifier_state = "This is a conversation between a human and an AI assistant. The following messages are part of the conversation:\n\n"

    classifier_questions = {}
    for i, msg in enumerate(messages[1:-1]):
      classifier_state += f"Message {i + 1} ({isinstance(msg, HumanMessage) and 'Human' or 'AI Assistant'}): {msg.content}\n"
      classifier_questions[f"message_{i + 1}_user_info"] = Noul(
        instructions=f"Does the message {i + 1} contain important information about the human or their preferences?"
      )

      classifier_questions[f"message_{i + 1}_relevant"] = Noul(
        instructions=f"Is the message {i + 1} relevant to understanding the conversation?"
      )

    classification = self.classifier.invoke(
      {"state": classifier_state, "questions": classifier_questions}
    )

    for i, msg in enumerate(messages[1:-1]):
      if (
        classification.nouls[f"message_{i + 1}_user_info"].noul >= 0.8
        or classification.nouls[f"message_{i + 1}_relevant"].noul >= 0.8
      ):
        relevant_messages.append(msg)

    # System message + relevant messages + the last message (which is the most recent assistant message).
    filtered_messages = [messages[0], *relevant_messages, messages[-1]]

    # Clear the existing history first, then set the filtered list as the new one.
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *filtered_messages]}

  def chatbot_node(self, state: ChatBotState):
    """Processes messages and generates responses."""
    system_prompt = "You are a helpful assistant that remembers previous conversations."
    messages = [SystemMessage(content=system_prompt), *state["messages"]]
    response = self.llm.invoke(messages)
    return {"messages": [response]}

  def build_graph(self):
    self.graph = StateGraph(ChatBotState)
    self.graph.add_node("chatbot", self.chatbot_node)
    self.graph.add_node("filter", self.filter_relevant_messages)
    self.graph.add_edge(START, "chatbot")
    self.graph.add_conditional_edges(
      "chatbot",
      self.require_filter,
      {
        "filter": "filter",
        "continue": END,
      },
    )

    return self.graph

  def compile_graph(self):
    if not self.graph:
      graph = self.build_graph()

    sqlite_saver = SqliteSaver(self.conn)

    return graph.compile(checkpointer=sqlite_saver)

import sqlite3
from operator import add
from typing import Annotated, TypedDict

from langchain.messages import SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from pydantic import SecretStr

from smart_memory_chat.settings import Settings


class ChatBotState(TypedDict):
  messages: Annotated[list[BaseMessage], add]


class ChatBotGraph:
  def __init__(self, settings: Settings):
    self.llm = ChatOpenAI(
      api_key=SecretStr(settings.openai_api_key),
      model=settings.ai_model,
      temperature=settings.ai_model_temperature,
    )
    self.graph: StateGraph | None = None
    self.conn = sqlite3.connect(settings.database_name, check_same_thread=False)

  def chatbot_node(self, state):
    """Node that processes messages and generates responses."""
    system_prompt = "You are a helpful assistant that remembers previous conversations."
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = self.llm.invoke(messages)
    return {"messages": [response]}

  def build_graph(self):
    self.graph = StateGraph(ChatBotState)
    self.graph.add_node("chatbot", self.chatbot_node)
    self.graph.add_edge(START, "chatbot")

    return self.graph

  def compile_graph(self):
    if not self.graph:
      graph = self.build_graph()

    sqlite_saver = SqliteSaver(self.conn)

    return graph.compile(checkpointer=sqlite_saver)

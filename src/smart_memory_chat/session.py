from langchain.messages import HumanMessage
lazy from langgraph.graph.state import CompiledStateGraph, RunnableConfig


class ChatSession:
  def __init__(self, thread_id: str, compiled_graph: CompiledStateGraph):
    self.config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    self.compiled_graph = compiled_graph
    self.thread_id = thread_id

  def send_message(self, message: str):
    result = self.compiled_graph.invoke(
      {"messages": [HumanMessage(content=message)]}, self.config
    )
    return result["messages"][-1].content

  def start(self):
    print(f"Chat started. Your thread ID is: {self.thread_id}")
    print("Write your message (or 'exit' or 'quit' to exit):")
    print("Restore this conversation later using the same thread ID.")
    print("--------------------------------------------------")

    while True:
      try:
        user_input = input("You: ").strip()
      except EOFError, KeyboardInterrupt:
        print("\nBye!")
        break

      if not user_input:
        continue
      if user_input.lower() in {"exit", "quit"}:
        print("See you later!")
        break

      response = self.send_message(user_input)
      print("Assistant:", response)

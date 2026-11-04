import argparse
import uuid

from smart_memory_chat.graph import ChatBotGraph
from smart_memory_chat.session import ChatSession
from smart_memory_chat.settings import get_settings

parser = argparse.ArgumentParser(description="A simple chat application.")

parser.add_argument(
  "-ti",
  "--thread-id",
  type=str,
  default="",
  help="The thread id to restore the chat from.",
)


def main() -> None:
  args = parser.parse_args()
  thread_id = args.thread_id or str(uuid.uuid4())
  settings = get_settings()
  graph = ChatBotGraph(settings)
  compiled_graph = graph.compile_graph()
  session = ChatSession(thread_id, compiled_graph)

  session.start()

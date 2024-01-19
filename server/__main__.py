import lmql
import sys
import os
from duckyai.server import chatserver

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python app.py <file>"

    file = sys.argv[1]
    absolute_path = os.path.abspath(file)
    os.chdir(os.path.dirname(absolute_path))

    # Start the WebSocket chat server
    chatserver(absolute_path).run()
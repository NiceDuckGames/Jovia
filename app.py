import sys
import subprocess
import os
import lmql

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(project_root)

if __name__ == "__main__":
    file = sys.argv[1]
    absolute_path = os.path.abspath(file)
    subprocess.run([sys.executable, "-m", "duckyai.server", absolute_path], cwd=project_root)
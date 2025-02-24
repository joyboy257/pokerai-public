import os

# Directories to check for missing __init__.py files
directories_to_check = [
    "ai",
    "ai/Abel",
    "ai/Kane",
    "game",
    "tests",
    "logs",
    "models/checkpoints",
]

def create_init_file(directory):
    init_file = os.path.join(directory, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# This file allows Python to recognize this directory as a package.\n")
        print(f"Created: {init_file}")

def check_and_create_init_files():
    for directory in directories_to_check:
        if os.path.exists(directory):
            create_init_file(directory)
        else:
            print(f"Directory does not exist: {directory}")

if __name__ == "__main__":
    check_and_create_init_files()

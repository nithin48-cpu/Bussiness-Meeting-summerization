import os
from pathlib import Path
import yaml

project_name = "meeting_summary_generation"

List_of_dirs = [f'{project_name}/data/__init__.py',
                f'{project_name}/transform/__init__.py',
                f'{project_name}/model/__init__.py',
                f'{project_name}/train/__init__.py',
                f'{project_name}/parms.yaml',
                f'{project_name}/main.py', ]

for file_path in List_of_dirs:
    file_path = os.path.join(os.getcwd(), Path(file_path))
    dir, file_name = os.path.split(file_path)

    if dir != '':
        os.makedirs(dir, exist_ok=True)
        print("created the dir", dir)
    if not os.path.exists(file_path):
        file = open(file_path, 'w')
        print("created the file", file_name, "in the dir.", dir)
    else:
        print("The file ", file_name, "and", dir, "already exists")

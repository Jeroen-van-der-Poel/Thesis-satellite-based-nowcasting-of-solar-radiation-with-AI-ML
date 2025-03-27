@ -1,21 +0,0 @@
#!/usr/bin/env bash
# we follow https://matt.sh/python-project-structure-2024 and use poetry and pyroject.toml
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

# set python executable
PYTHON=python3.9

# create and activate virtual environment
$PYTHON -m venv .venv
source $DIR/.venv/bin/activate
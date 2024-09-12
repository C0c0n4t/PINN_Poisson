#!/bin/bash

# find python
if [ -z "$(python3 --version)" ] ; then
    echo "No python found"
    exit 1
fi

echo "Python that's detected is $(python3 --version)"

# create the virtual environment, activate and set up requirements
if [ ! -d ".venv" ] ; then
    python3 -m venv .venv
    source ./.venv/bin/activate
    pip install -r ./requirements.txt
else
    source ./.venv/bin/activate
    pip install -r ./requirements.txt
fi

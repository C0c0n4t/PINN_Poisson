#!/bin/bash

# find python
name=""
if [ ! -z "$(python --version)" ] ; then
    name="python"
elif [ ! -z "$(python3 --version)" ] ; then
    name="python3"
else
    echo "No python found"
    exit 1
fi

echo "Python that's detected is $($name --version)"

# create the virtual environment, activate and set up requirements
if [ ! -d ".venv" ] ; then
    $name -m venv .venv
    source ./.venv/bin/activate
    pip install -r ./requirements.txt
else
    source ./.venv/bin/activate
    pip install -r ./requirements.txt
fi

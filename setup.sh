#!/bin/bash

if ! [ -d $PWD/venv ]; then
     echo "Creating virtual environment."
     python3 -m venv $PWD/venv
fi

echo "Activating virtual environment and installing required packages."
source $PWD/venv/bin/activate
pip install -r requirements.txt

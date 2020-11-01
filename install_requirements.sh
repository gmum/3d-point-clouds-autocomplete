#!/bin/bash

set -e

conda install -y cudatoolkit=10.0
pip install -r requirements.txt
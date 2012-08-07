#!/bin/bash

if [ $# -ne 1 ]; then
    echo "no argument supplied"
    exit
fi

RMNET "$1"

newRG "$1"

newSO "$1"
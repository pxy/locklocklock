#!/bin/bash
sed 's/^[ \t]*//' "$1" | sort -n -k 1 > "$1.out"
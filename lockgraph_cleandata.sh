#!/bin/bash
sed 's/^[ \t]*//' "$1" | sort -n -k 1 > "$1.out"

# use colrm 14 16 if necessary
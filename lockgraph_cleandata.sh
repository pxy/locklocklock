#!/bin/bash
sed 's/^[ \t]*//' "$1" | sort -n -k 1 | colrm 14 16 > "$1.out"
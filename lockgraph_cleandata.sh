#!/bin/bash


if [ $# -ne 2 ]; then
	echo "Wrong number of arguments."
	exit
fi

if [ ! -f 'BaseX67.jar' ]; then
	echo "Missing BaseX jar file in this directory."
	exit
fi

echo "Creating: $(dirname $1)/$2_try.dat"
java -cp BaseX67.jar org.basex.BaseX mutrace_tools/try_ts_mxID_tID.xq -i "$1" | sed 's/^[ \t]*//' | sort -n -k 1 > "$(dirname $1)/$2_try.dat"

echo "Creating: $(dirname $1)/$2_acq.dat"
java -cp BaseX67.jar org.basex.BaseX mutrace_tools/acq_ts_mxID_tID.xq -i "$1" | sed 's/^[ \t]*//' | sort -n -k 1 > "$(dirname $1)/$2_acq.dat"

echo "Creating: $(dirname $1)/$2_rel.dat"
java -cp BaseX67.jar org.basex.BaseX mutrace_tools/rel_ts_mxID_tID.xq -i "$1" | sed 's/^[ \t]*//' | sort -n -k 1 > "$(dirname $1)/$2_rel.dat"

echo "Creating: $(dirname $1)/$2_addr.dat"
java -cp BaseX67.jar org.basex.BaseX mutrace_tools/mx_trace.xq -i "$1" | sed 's/^[ \t]*//' | sort -n -k 1 > "$(dirname $1)/$2_addr.dat"

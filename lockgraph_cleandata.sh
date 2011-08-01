#!/bin/bash

# this tool requires that the baseX xquery database jar file exists in the java classpath.
# fetch from http://files.basex.org/releases/latest, e.g.
# wget http://files.basex.org/releases/latest/BaseX671.jar

if [ $# -ne 2 ]; then
	echo "Wrong number of arguments."
	exit
fi

#if [ ! -f 'BaseX.jar' ]; then
#	echo "Missing BaseX jar file in this directory."
#	exit
#fi

echo "Creating: $(dirname $1)/$2_try.dat"
java org.basex.BaseX mutrace_tools/try_ts_mxID_tID.xq -i "$1" | sed 's/^[ \t]*//' | sort -n -k 1 > "$(dirname $1)/$2_try.dat"

echo "Creating: $(dirname $1)/$2_acq.dat"
java org.basex.BaseX mutrace_tools/acq_ts_mxID_tID.xq -i "$1" | sed 's/^[ \t]*//' | sort -n -k 1 > "$(dirname $1)/$2_acq.dat"

echo "Creating: $(dirname $1)/$2_rel.dat"
java org.basex.BaseX mutrace_tools/rel_ts_mxID_tID.xq -i "$1" | sed 's/^[ \t]*//' | sort -n -k 1 > "$(dirname $1)/$2_rel.dat"

echo "Creating: $(dirname $1)/$2_addr.dat"
java org.basex.BaseX mutrace_tools/mx_trace.xq -i "$1" | sed 's/^[ \t]*//' | sort -n -k 1 > "$(dirname $1)/$2_addr.dat"

echo "Creating: $(dirname $1)/$2_cond.dat"
java org.basex.BaseX mutrace_tools/condtime_tID_mxID.xq -i "$1" | sed 's/^[ \t]*//' > "$(dirname $1)/$2_cond.dat"

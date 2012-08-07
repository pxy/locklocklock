#!/bin/bash

# this tool requires that the baseX xquery database jar file exists in the java classpath.
# fetch from http://files.basex.org/releases/latest, e.g.
# wget http://files.basex.org/releases/latest/BaseX671.jar

# arg 1 : path to where the mutrace xml file is found and where the output files will be written
# arg 2 : the new name of the files


if [ $# -ne 1 ]; then
	echo "Wrong number of arguments."
	exit
fi

if [ ! -f $1 ]; then
	echo "The first argument should be a the path to a xml file"
	exit
fi

#if [ ! -f 'BaseX.jar' ]; then
#	echo "Missing BaseX jar file in this directory."
#	exit
#fi

fd=$(dirname $1)
fn=$(basename $1 ".xml")

echo "Creating: ${fd}/${fn}_try.dat"
java org.basex.BaseX -i "$1" mutrace_tools/try_ts_mxID_tID.xq | sed 's/^[ \t]*//' | sort -n -k 1 > "${fd}/${fn}_try.dat"

echo "Creating: ${fd}/${fn}_acq.dat"
java org.basex.BaseX -i "$1" mutrace_tools/acq_ts_mxID_tID.xq | sed 's/^[ \t]*//' | sort -n -k 1 > "${fd}/${fn}_acq.dat"

echo "Creating: ${fd}/${fn}_rel.dat"
java org.basex.BaseX -i "$1" mutrace_tools/rel_ts_mxID_tID.xq | sed 's/^[ \t]*//' | sort -n -k 1 > "${fd}/${fn}_rel.dat"

echo "Creating: ${fd}/${fn}_addr.dat"
java org.basex.BaseX -i "$1" mutrace_tools/mx_trace.xq | sed 's/^[ \t]*//' | sort -n -k 1 > "${fd}/${fn}_addr.dat"

echo "Creating: ${fd}/${fn}_cond.dat"
java org.basex.BaseX -i "$1" mutrace_tools/condtime_tID_mxID.xq | sed 's/^[ \t]*//' > "${fd}/${fn}_cond.dat"



(: Prints the stackframe closest to the lock construct. :)

for $mx in root()/mutrace/mutex_list/mutex
return (data($mx/id), substring-after(data($mx/stacktrace/frame[last()]), 'gcc-pthreads/bin/'), substring-after(data($mx/stacktrace/frame[last() - 2]), 'gcc-pthreads/bin/') , "&#xA;")

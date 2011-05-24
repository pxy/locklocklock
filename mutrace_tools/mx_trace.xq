(:
for $mx in root()/mutrace/mutex_list/mutex
return (data($mx/id), substring-after(data($mx/stacktrace/frame[last()]), 'gcc-pthreads/bin/'), substring-after(data($mx/stacktrace/frame[last() - 2]), 'gcc-pthreads/bin/') , "&#xA;")
:)
for $mx in root()/mutrace/mutex_list/mutex
return (data($mx/id), data($mx/stacktrace/frame[last()]), data($mx/stacktrace/frame[last() - 2]), data($mx/stacktrace/frame[last() - 3]), "&#xA;")

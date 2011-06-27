
for $mx in root()/mutrace/mutex_list/mutex
for $ts in $mx/lock/timestamp
return (data($ts), data($ts/@tid), data($mx/id), "&#xA;")

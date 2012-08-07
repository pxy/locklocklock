
for $mx in root()/mutrace/mutex_list/mutex
	return (data($mx/id), data($mx/nsec_locked_total) div data($mx/n_locked[1]), "&#xA;")
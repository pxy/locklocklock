<table>
<tr>
<th>id</th><th>n_locked</th><th>n_owner_changed</th><th>nsec_locked_max</th><th>nsec_locked_total</th><th>count</th><th>avg locktime</th>
</tr>
{
for $mx in root()/mutrace/mutex_list/mutex[data(n_owner_changed)>0]
return <tr>
	<td> {data($mx/id)} </td>
	<td> {data($mx/n_locked[1])} </td>
	<td> { data($mx/n_owner_changed) } </td>
	<td> { data($mx/nsec_locked_max) } </td>
	<td> { data($mx/nsec_locked_total) } </td>
	<td> { count($mx/timestamp/trylock)} </td>
	<td> { data($mx/nsec_locked_total) div data($mx/n_locked[1]) } </td>
</tr>
}
</table>

(: distinct-values(for $mx in root()/mutrace/mutex_list/mutex[4]/trylock/timestamp/@tid return data($mx)) :)
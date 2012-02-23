#!/usr/bin/env python

# A script to facilitate the generation of plots containing values for
# the rates of each stage in dedup and the memory overhead in the
# chunkstage.
#
# The configurations have to be specified manually, and the function
# requires the existance of the usual _try, _acq, and _rel files, as
# well as a ts_chunk file.

from lockgraph import *

class Runconf(object):
    def __init__ (self, th_dist, chunk_l, classes):
        self.dist = th_dist
        self.classes = classes
        self.chunk_l = chunk_l

kalkyl222 = Runconf([2,2,2], 54, [range(2), range(2,4), range(4,6)])
kalkyl224 = Runconf([2,2,4], 54, [range(2), range(2,4), range(4,8)])

halvan333 = Runconf([3,3,3], 52, [range(3),range(3,6), range(6,9)])
halvan101010 = Runconf([10,10,10], 54, [range(10), range(10,20), range(20,30)])

def chunk_oh_vs_rate(conf):
    dist = conf.dist
    classes = conf.classes
    chunk_l = conf.chunk_l

    machine = "halvan_"
    basedir = '/Users/jonatanlinden/Documents/dedup_meas/halvan/memcont/'
    cls = 0 # HACK chunk class
    size = 1000
    fname = ".".join(map(str, dist))
    fname2 = ":".join(map(str, dist))
    lt = ParsedLockTrace(basedir + 'dedup.random.' + fname2 + 'th')
    lt.delete_thread(imin(lt.start_ts()))
    lt.set_classes(classes)

    lt.analyze()
    anchor_l = idx_of_sublist(lt.serv_times().mask, [False, False, True])
    compress_l = idx_of_sublist(lt.serv_times().mask, [True, True, False])

    lt_anchor = sub_lt_by_class(lt, 1)
    lt_chunk = sub_lt_by_class(lt, 0, start=lt_anchor.end)
    lt_anchor.time_line(timelines.SERV, use_class=True)
    lt_chunk.time_line(timelines.SERV, use_class=True)

    lt_anchor_l = interval_analysis(lt_anchor, 1, anchor_l, size)
    anchor_splits = timestamps_by_cnt_at_lock(lt_anchor.tl[1], anchor_l, size)

    lt_chunk_l = interval_analysis(lt_chunk, 0, chunk_l, size)
    chunk_splits = timestamps_by_cnt_at_lock(lt_chunk.tl[0], chunk_l, size)

    chunk_rate = [rate_of_lock(x, 0, chunk_l) for x in lt_anchor_l] + [rate_of_lock(x, 0, chunk_l) for x in lt_chunk_l]
    compress_rate = [rate_of_lock(x, 2, compress_l) for x in lt_anchor_l] + [rate_of_lock(x, 2, compress_l) for x in lt_chunk_l]
    anchor_rate = [rate_of_lock(x, 1, anchor_l) for x in lt_anchor_l]
    anchor_rate.extend([0.0] * (len(chunk_rate) - len(anchor_rate)))

    splits = map (lambda x: x.end, lt_anchor_l)
    splits.extend(map (lambda x: x.end, lt_chunk_l))
    start = lt_anchor_l[0].start
    end   = splits.pop(-1)

    # load timeline with chunk overhead time
    tl = unpack_timeline(basedir + 'ts_chunk_' + fname2 + '.bin', 3)
    tl = sorted([(a, b-a, c-b) for (a,b,c) in tl], key=op.itemgetter(0))
    mem_oh_parts = split_tl(tl, splits, start=start, end = end)
    avg_mem_oh = tuple_avg_of_chunks(mem_oh_parts, -1, 3)

    with open(basedir + "data/" + 'thru_vs_oh_' + machine + fname + '.dat', 'w') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        w.writerows(zip(anchor_rate, chunk_rate, compress_rate, avg_mem_oh))



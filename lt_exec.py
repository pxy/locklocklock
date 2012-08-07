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
    def __init__ (self, th_dist, chunk_l, classes, cls, relpath=None):
        self.dist    = th_dist
        self.classes = classes
        self.chunk_l = chunk_l
        self.cls     = cls
        self.relpath = relpath


#kalkyl222 = Runconf([2,2,2], 54, [range(2), range(2,4), range(4,6)])
#kalkyl224 = Runconf([2,2,4], 54, [range(2), range(2,4), range(4,8)])
halvan111 = Runconf([1,1,1], 54, [[0],[1],[2]], [1,0,2])
halvan222 = Runconf([2,2,2], 50, [range(2), range(2,4), range(4,6)], [1,0,2])
halvan333 = Runconf([3,3,3], 52, [range(3),range(3,6), range(6,9)], [1, 0, 2])
halvan444 = Runconf([4,4,4], 54, [range(4), range(4,8), range(8,12)], [1,0,2])
halvan555 = Runconf([5,5,5], 54, [range(5), range(5,10), range(10,15)], [1, 0, 2])
halvan458 = Runconf([4,5,8], 53, [range(4), range(4, 12), range(12,17)], [0, 2, 1])
halvan888 = Runconf([8,8,8], 54, [range(8), range(8, 16), range(16, 24)], [1, 0, 2]) 
halvan101010 = Runconf([10,10,10], 54, [range(10), range(10,20), range(20,30)], [1, 0, 2])

halvanx555 = Runconf([5,5,5], 53, [range(5), range(5,10), range(10,15)], [0,2,1], relpath='exclusive/')

tintin555 = Runconf([5,5,5], 53, [range(5), range(5,10), range(10,15)], [1,0,2])
tintin444 = Runconf([4,4,4], 53, [[0, 9, 10, 11], range(1,5), range(5,9)], [2, 1 ,0])
tintin333 = Runconf([3,3,3], 54, [range(3),range(3,6), range(6,9)], [1, 0, 2])
tintin222 = Runconf([2,2,2], 52, [range(2), range(2,4), range(4,6)], [1,0,2])
tintin111 = Runconf([1,1,1], 53, [[0], [1], [2]], [2,1,0])

halvan = [halvan111, halvan222, halvan333, halvan444, halvan458, halvan555, halvan888, halvan101010]
tintin = [tintin111, tintin222, tintin333, tintin444, tintin555]
halvanx = [halvanx555]

machine = "tintin_"
gbasedir = '/Users/jonatanlinden/Documents/dedup_meas/tintin/memcont/'
timestep = 2000000000

def chunk_oh_vs_rate(conf):
    dist = conf.dist
    classes = conf.classes
    chunk_l = conf.chunk_l
    fname = ".".join(map(str, dist))
    fname2 = ":".join(map(str, dist))
    basedir = gbasedir
    if conf.relpath:
        basedir += conf.relpath
    anchor_cls, chunk_cls, compress_cls = conf.cls
    size = 1000
    lt = ParsedLockTrace(basedir + 'dedup.random.' + fname2 + 'th')
    lt.delete_thread(imin(lt.start_ts()))

    lt.set_classes(classes)

    lt.analyze()
    al = [False,False,False]
    al[compress_cls] = True
    cl = [True, True, True]
    cl[compress_cls] = False
    anchor_l = idx_of_sublist(lt.serv_times().mask, al)
    compress_l = idx_of_sublist(lt.serv_times().mask, cl)

    lt_anchor = sub_lt_by_class(lt, anchor_cls)
    lt_chunk_only = sub_lt_by_class(lt, chunk_cls, start=lt_anchor.end)
    lt_anchor.time_line(timelines.SERV, use_class=True)
    lt_chunk_only.time_line(timelines.SERV, use_class=True)

    lt_anchor_l = [lt_anchor]
    lt_chunk_only_l = [lt_chunk_only]
    #lt_anchor_l = interval_analysis(lt_anchor, anchor_cls, anchor_l, timechunk=timestep)
    #lt_chunk_only_l = interval_analysis(lt_chunk_only, chunk_cls, chunk_l, timechunk=timestep)

    chunk_rate = [rate_of_lock(x, chunk_cls, chunk_l) for x in lt_anchor_l] + [rate_of_lock(x, chunk_cls, chunk_l) for x in lt_chunk_only_l]
    compress_rate = [rate_of_lock(x, compress_cls, compress_l) for x in lt_anchor_l] + [rate_of_lock(x, compress_cls, compress_l) for x in lt_chunk_only_l]
    anchor_rate = [rate_of_lock(x, anchor_cls, anchor_l) for x in lt_anchor_l]
    anchor_rate.extend([1.0] * (len(chunk_rate) - len(anchor_rate)))

    splits = map (lambda x: x.end, lt_anchor_l)
    splits.extend(map (lambda x: x.end, lt_chunk_only_l))
    start = lt_anchor_l[0].start
    end   = splits.pop(-1)
    
    a_l = len(lt_anchor_l)
    c_l = len(lt_chunk_only_l)

    anchor_ths   = [dist[0]] * a_l + [0] * c_l
    chunk_ths    = [dist[1]] * (a_l + c_l)
    compress_ths = [dist[2]] * (a_l + c_l)


    #************ BEGIN size of hashtable
    # lock = 0
    # inserts = cumulative_sum([len(filter(lambda y: y[3] == lock, x.tl[chunk_cls])) for x in lt_anchor_l]) # number of access by chunk stage to chunk lock in each chop of the analysis
    # removes = cumulative_sum([len(filter(lambda y: y[3] == lock, x.tl[compress_cls])) for x in lt_anchor_l])

    # inserts2 = cumulative_sum([len(filter(lambda y: y[3] == lock, x.tl[chunk_cls])) for x in lt_chunk_only_l]) # number of access by chunk stage to chunk lock in each chop of the analysis
    # inserts2 = map (op.add, inserts2, [inserts[-1]] * len(inserts2))
    # removes2 = cumulative_sum([len(filter(lambda y: y[3] == lock, x.tl[compress_cls])) for x in lt_chunk_only_l])
    # removes2 = map (op.add, removes2, [removes[-1]] * len(removes2))

    # inserts.extend(inserts2)
    # removes.extend(removes2)
    # no_items_in_q = np.array(inserts) - np.array(removes)
    #************* END HASHTABLE SIZE


    # load timeline with chunk overhead time
    tl = unpack_timeline(basedir + 'ts_chunk_' + fname2 + '.bin', 3)
    tl = sorted([(a, b-a, c-b) for (a,b,c) in tl], key=op.itemgetter(0))
    mem_oh_parts = split_tl(tl, splits, start=start, end = end)
    avg_mem_oh = tuple_avg_of_chunks(mem_oh_parts, -1, 3)
    megatuple = zip(anchor_ths, chunk_ths, compress_ths, anchor_rate, chunk_rate, compress_rate, avg_mem_oh)
    return megatuple


if __name__ == '__main__':
    megatuples = map (chunk_oh_vs_rate, tintin)
    for m,h in zip(megatuples, tintin):
        with open(gbasedir + (h.relpath or "") + "data/" + 'thru_vs_oh_' + machine + ".".join(map(str,h.dist)) + '.dat', 'w') as f:
            
            w = csv.writer(f, delimiter=' ', lineterminator='\n')
            w.writerows(m)




from typing import List


class DAGnode:
    depends: List
    depending: List
    expr: object

    def __init__(self, expr):
        self.expr = expr
        self.depends = list()
        self.depending = list()


class DAG:
    nodes: List[DAGnode]

    def __init__(self, nodes):
        self.nodes = nodes
        self.root = list()
        for n in self.nodes:
            if not n.depends:
                self.root.append(n)
        self.best = 0

    def _red(self, mexpo, avail, picked, cnt, order):
        # Compute exposed
        expo = list()
        for av in avail:
            for d in av.depends:
                if d in picked and d not in expo:
                    expo.append(d)

        if len(expo) < mexpo:
            return self._test(mexpo, avail, picked, cnt, order)

        # First test all exposed node and check for straight reduction
        connected = dict()
        for e in expo:
            connected[e] = e

        for e in expo:
            for d in e.depending:
                if d not in connected:
                    connected[d] = e
                else:
                    f = connected[d]
                    while f != connected[f]:
                        f = connected[f]
                    connected[e] = f
                    fp = connected[d]
                    while fp != connected[fp]:
                        n = connected[fp]
                        connected[fp] = f
                        fp = n

        groups = dict()
        for d in connected.keys():
            f = connected[d]
            while f != connected[f]:
                f = connected[f]
            if f in groups:
                groups[f].append(d)
            else:
                groups[f] = [d]

        reduce_group = dict()
        tot_red = 0
        for d, l in groups.items():
            nexpo = 0
            navail = []
            for e in l:
                if e in expo:
                    nexpo += 1
                if e in avail:
                    navail.append(e)

            if nexpo + len(navail) == len(l) and nexpo >= len(navail):
                reduce_group[d] = navail[:]
                tot_red += nexpo

        for d, l in reduce_group.items():
            for av in l:
                order.append(av)
                avail.remove(av)
                picked[av] = True
                for d in av.depending:
                    cnt[d] -= 1
                    if cnt[d] == 0:
                        avail.append(d)
                for d in av.depends:
                    cnt[d] += 1

        if len(reduce_group) > 0:
            return self._red(mexpo, avail, picked, cnt, order)
        else:
            return self._test(mexpo, avail, picked, cnt, order)

    def _test(self, mexpo, avail, picked, cnt, order):
        """ Test if it is possible to have less than expo exposed nodes
        """
        if not avail:
            self.best = mexpo
            return order

        # exposed list
        expo = list()
        for av in avail:
            for d in av.depends:
                if d in picked and d not in expo:
                    expo.append(d)

        d_cnt = dict()
        for e in expo:
            for d in e.depending:
                if d in d_cnt:
                    d_cnt[d] += 1
                else:
                    d_cnt[d] = 1

        sorting = list()
        # Sorting the availables
        for idx, av in enumerate(avail):
            fpot = 0
            paff = 0
            for d in av.depending:
                if cnt[d] == 1:
                    fpot += 1
                if d in d_cnt:
                    paff += d_cnt[d]
            rpot = 0
            for d in av.depends:
                if cnt[d] == len(d.depending) - 1:
                    rpot += 1
            if len(expo) >= mexpo:
                sorting.append((rpot, fpot, paff, -len(av.depending), idx))
            else:
                sorting.append((fpot, rpot, paff, -len(av.depending), idx))
        idx = sorted(sorting)[0][-1]
        av = avail[idx]
        order.append(av)
        avail.remove(av)
        picked[av] = True
        for d in av.depending:
            cnt[d] -= 1
            if cnt[d] == 0:
                avail.append(d)
        for d in av.depends:
            cnt[d] += 1

        return self._red(mexpo, avail[:], dict(picked), dict(cnt), order)

    def test(self, mexpo):
        cnt = dict()
        for n in self.nodes:
            cnt[n] = len(n.depends)
        return self._test(mexpo, self.root[:], dict(), cnt, [])

    def sequence(self, mexpo=12, frac=0.5):
        avail = self.root[:]
        picked = dict()
        cnt = dict()
        for n in self.nodes:
            cnt[n] = len(n.depends)
        order = []
        frac = len(self.nodes) * frac

        while avail:
            # exposed list
            expo = list()
            for av in avail:
                for d in av.depends:
                    if d in picked and d not in expo:
                        expo.append(d)

            d_cnt = dict()
            for e in expo:
                for d in e.depending:
                    if d in d_cnt:
                        d_cnt[d] += 1
                    else:
                        d_cnt[d] = 1

            sorting = list()
            # Sorting the availables
            for idx, av in enumerate(avail):
                fpot = 0
                paff = 0
                for d in av.depending:
                    if cnt[d] == 1:
                        fpot += 1
                    if d in d_cnt:
                        paff += d_cnt[d]
                rpot = 0
                for d in av.depends:
                    if cnt[d] == len(d.depending) - 1:
                        rpot += 1
                l = len(self.nodes)
                if len(expo) >= mexpo:
                    sorting.append((rpot * l + fpot, paff, -len(av.depending), idx))
                elif len(expo) >= frac:
                    p = (len(expo) - frac) / frac
                    sorting.append((fpot * (l ** (1 - p)) + rpot * (l ** p), paff, -len(av.depending), idx))
                else:
                    sorting.append((fpot * l + rpot, paff, -len(av.depending), -idx))
            idx = abs(sorted(sorting, reverse=True)[0][-1])
            av = avail[idx]
            order.append(av)
            avail.remove(av)
            picked[av] = True
            for d in av.depending:
                cnt[d] -= 1
                if cnt[d] == 0:
                    avail.append(d)
            for d in av.depends:
                cnt[d] += 1

            # Start reducing
            while True:
                # Compute exposed
                expo = list()
                for av in avail:
                    for d in av.depends:
                        if d in picked and d not in expo:
                            expo.append(d)

                if len(expo) < mexpo:
                    break

                # First test all exposed node and check for straight reduction
                connected = dict()
                for e in expo:
                    connected[e] = e

                for e in expo:
                    for d in e.depending:
                        if d not in connected:
                            connected[d] = e
                        else:
                            f = connected[d]
                            while f != connected[f]:
                                f = connected[f]
                            connected[e] = f
                            fp = connected[d]
                            while fp != connected[fp]:
                                n = connected[fp]
                                connected[fp] = f
                                fp = n

                groups = dict()
                for d in connected.keys():
                    f = connected[d]
                    while f != connected[f]:
                        f = connected[f]
                    if f in groups:
                        groups[f].append(d)
                    else:
                        groups[f] = [d]

                reduce_group = dict()
                tot_red = 0
                for d, l in groups.items():
                    nexpo = 0
                    navail = []
                    for e in l:
                        if e in expo:
                            nexpo += 1
                        if e in avail:
                            navail.append(e)

                    if nexpo + len(navail) == len(l) and nexpo >= len(navail):
                        reduce_group[d] = navail[:]
                        tot_red += nexpo

                for d, l in reduce_group.items():
                    for av in l:
                        order.append(av)
                        avail.remove(av)
                        picked[av] = True
                        for d in av.depending:
                            cnt[d] -= 1
                            if cnt[d] == 0:
                                avail.append(d)
                        for d in av.depends:
                            cnt[d] += 1

                if len(reduce_group) == 0:
                    break

        return order


if __name__ == '__main__':
    tot = 1024
    nodes = [DAGnode(i) for i in range(tot)]
    for i in range(tot):
        j = ((i + 1) >> 1)
        if j > 0:
            nodes[i].depends.append(nodes[j - 1])
        j = (i + 1) << 1
        if j < tot:
            nodes[i].depending.append(nodes[j - 1])
            nodes[i].depending.append(nodes[j])

    dag = DAG(nodes)
    for n in dag.sequence(20, 0.5):
        print(n.expr)

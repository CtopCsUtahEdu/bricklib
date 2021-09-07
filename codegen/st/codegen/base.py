from typing import List, Tuple, Dict
from st.grid import Grid, GridRef
from st.expr import Expr
import st.expr
from st.alop import BinaryOperators
from st.codegen.reduction import Reduction
import st.codegen.dag_opt as dag_opt
from st.codegen.buffer import Buffer, Shift, BufferRead
from st.codegen.backend.base import CodeBlock
import sys

""" Code generation is using a mix of scatter and reduce

Generic code generation helps create the basic group based on reuse

Platform-specific codegen are called with the grouping:

These platform specific codegens are inherited from Backend
"""


class CodeGen:
    FMA = True

    def __init__(self, backend, layout, *, dag_msize=5, K=2, klimit = 20, scatter_thres=1.5, min_fixed=5, unroll=False,
                 dimsplit=False, stride=None):
        # will be set by others
        self.LAYOUTREL = False
        self.TILE_DIM = None
        self.backend = None
        self.layout = None
        self.FOLD = None
        self.TILE = False
        self.STRIDE = None
        self.DIRECT = False
        self.K = K
        self.KLIMIT = klimit
        if stride:
            self.STRIDE = stride[:]

        # configurables
        self.THRES = scatter_thres
        self.MSIZE = dag_msize
        self.MIN_FIXED = min_fixed
        self.UNROLL = unroll
        self.DIM_SPLIT = dimsplit
        self.ALIGNED = True

        # dynamics
        self.grids = []
        self.TILE_SIZE = 0
        self.config(backend, layout)

    def config(self, backend=None, layout=None):
        self.backend = backend
        self.layout = layout
        backend.setCodeGen(self)
        backend.setLayout(layout)
        layout.setBackend(backend)
        layout.setCodeGen(self)
        layout.checkConfig()
        backend.checkConfig()
        self.TILE_DIM = layout.dim[:]
        self.FOLD = layout.fold[:]
        self.TILE_SIZE = 1
        for d in self.TILE_DIM:
            self.TILE_SIZE *= d
        if self.STRIDE is None:
            self.STRIDE = [1] * len(self.TILE_DIM)

    def split(self, rst, nxt, dim, left_dim, f):
        proj = dict()
        for n, indices in rst.items():
            offs = f(n)
            for idx in indices:
                o = offs[idx][:]
                del o[dim]
                o = str(o)
                if o not in proj:
                    proj[o] = []
                proj[o].append((n, idx))

        m = 0
        for exprs in proj.values():
            m = max(m, len(exprs))

        low = 0
        high = 0

        for exprs in proj.values():
            l = len(exprs)
            if l * self.THRES < m:
                low += l
            else:
                high += 1

        # Have to be tall but not flat
        if m >= low / left_dim and high * self.THRES < m ** (left_dim - 1):
            for exprs in proj.values():
                l = len(exprs)
                if l * self.THRES >= m:
                    for n, idx in exprs:
                        rst[n].remove(idx)
                        if n not in nxt:
                            nxt[n] = []
                        nxt[n].append(idx)

    def _calculate(self):
        def conv_reduction(outputs: List[Grid]):
            """ Convert the outputs to a corresponding expression DAG
            :param outputs: Grids to be written
            :return: a graph with each output replaced by a tuple (Grid, rhs)
            """
            from st.expr import BinOp
            fixed = dict()

            # Determine how many times each of the node is referenced using DFS
            def indexing_node(node: Expr, idx: int):
                cnt = 0
                if 'node_index' in node.attr:
                    if node.attr['sub_cnt'] >= self.MIN_FIXED:
                        node.attr['fixed'] = True
                        fixed[node.attr['node_index']] = node
                    return idx, node.attr['sub_cnt']
                else:
                    node.attr['node_index'] = idx
                    idx += 1
                    cnt += 1

                if isinstance(node, GridRef):
                    node.attr['sub_cnt'] = cnt
                    return idx, cnt

                for child in node.children:
                    idx, ccnt = indexing_node(child, idx)
                    cnt += ccnt

                node.attr['sub_cnt'] = cnt
                return idx, cnt

            idx = 0
            for out in outputs:
                rhs: Expr = out.out[1]
                idx, _ = indexing_node(rhs, idx)
                fixed[rhs.attr['node_index']] = rhs
                rhs.attr['fixed'] = True

            def to_reduction(node: Expr):
                """ Recursively turn nodes into reduction

                :param node: The transforming node
                :return: The transformed node
                """
                if node.get_attr('fixed'):
                    node = fixed[node.attr['node_index']]
                if isinstance(node, st.expr.UnOp) and node.subexpr.get_attr('atomic'):
                    node.attr['atomic'] = True
                if not isinstance(node, GridRef):
                    for idx in range(len(node.children)):
                        node.children[idx] = to_reduction(node.children[idx])
                if isinstance(node, BinOp) and node.operator in Reduction.op:
                    ret = Reduction(Reduction.op[node.operator])
                    if isinstance(node.lhs, Reduction) and \
                            'fixed' not in node.lhs.attr \
                            and ret.operator == node.lhs.operator:
                        for t in node.lhs.children:
                            ret.children.append(t)
                        for t in node.lhs.terms_op:
                            ret.terms_op.append(t)
                    else:
                        ret.children.append(node.lhs)
                        ret.terms_op.append(Reduction.op[node.operator])

                    if isinstance(node.rhs, Reduction) and \
                            'fixed' not in node.rhs.attr \
                            and ret.operator == node.rhs.operator:
                        op = node.operator
                        if op == Reduction.op[op]:
                            # The operators are of the same direction
                            #   op -> op
                            for t in node.rhs.children:
                                ret.children.append(t)
                            for t in node.rhs.terms_op:
                                ret.terms_op.append(t)
                        else:
                            # The operators are of the different direction
                            #   op -> op
                            for t in node.rhs.children:
                                ret.children.append(t)
                            for t in node.rhs.terms_op:
                                if op == t:
                                    ret.terms_op.append(Reduction.op[op])
                                else:
                                    ret.terms_op.append(op)
                    else:
                        ret.children.append(node.rhs)
                        ret.terms_op.append(node.operator)

                    if ret.operator == BinaryOperators.Mul and len(ret.children) == 2 and self.FMA:
                        distributable = False
                        coeff_idx = None
                        for idx, child in enumerate(ret.children):
                            if child.get_attr('atomic') or child.get_attr('fixed'):
                                coeff_idx = idx
                            if isinstance(child, Reduction) \
                                    and ret.terms_op != BinaryOperators.Div \
                                    and 'fixed' not in child.attr:
                                distributable = True
                                for grandchild in child.children:
                                    if not (grandchild.get_attr('atomic') or
                                            'fixed' in child.attr):
                                        distributable = False

                        if coeff_idx is not None and distributable:
                            sum_child = ret.children[abs(1 - coeff_idx)]
                            distributed = Reduction(sum_child.operator)
                            distributed.attr['distributed'] = True
                            for idx, child in enumerate(sum_child.children):
                                distributed.terms_op.append(sum_child.terms_op[idx])
                                red = Reduction(BinaryOperators.Mul)
                                red.children.append(ret.children[coeff_idx])
                                red.children.append(sum_child.children[idx])
                                red.terms_op.append(BinaryOperators.Mul)
                                red.terms_op.append(BinaryOperators.Mul)
                                distributed.children.append(red)
                            ret = distributed

                    ret.attr['node_index'] = node.attr['node_index']
                    if 'fixed' in node.attr:
                        ret.attr['fixed'] = True
                        fixed[node.attr['node_index']] = ret

                    return ret
                return node

            for out in fixed.values():
                to_reduction(out)

            return fixed

        def gridref_count(ref_col: List[Dict[Grid, Dict[str, List[int]]]]) -> Dict[Grid, Dict[str, int]]:
            """ Convert a list of node refs to ref_count """
            ret = {}
            loff = 0
            for refs in ref_col:
                for grid, refdict in refs.items():
                    for off in refdict.values():
                        loff = max(len(off), loff)
            # pad everyone while calc
            for refs in ref_col:
                for grid, refdict in refs.items():
                    if grid not in ret:
                        ret[grid] = dict()
                    nrefdict = dict()
                    for off_str, off in refdict.items():
                        off = off + [None] * (len(self.TILE_DIM) - len(off))
                        off_str = str(off)
                        nrefdict[off_str] = off
                        if off_str in ret[grid]:
                            ret[grid][off_str] += 1
                        else:
                            ret[grid][off_str] = 1
                    refs[grid] = nrefdict
            return ret

        def _rmOff(ref_cnt: Dict[Grid, Dict[str, int]], rds: Dict[Grid, Dict[str, List[int]]], off: List[int]) -> int:
            """ Removing the read with specified offset

            :param ref_cnt: Dict[Grid, Dict[off_str, int]]
            :param rds: Dict[Grid, Dict[off_str, off]]
            :param off: [vec]
            :return: int, how many reads are reduced when removing reads
            """
            ret = 0
            for grid, ref_dict in rds.items():
                for off_str, o in ref_dict.items():
                    shifted = o[:]
                    for idx, s in enumerate(off):
                        if shifted[idx] is not None:
                            shifted[idx] += s
                    shifted_str = str(shifted)
                    v = ref_cnt[grid][shifted_str] - 1
                    if v == 0:
                        del ref_cnt[grid][shifted_str]
                    else:
                        ref_cnt[grid][shifted_str] = v
                        ret -= 1
            return ret

        def _evalOff(ref_cnt: Dict[Grid, Dict[str, int]], rds: Dict[Grid, Dict[str, List[int]]], off: List[int]) -> int:
            """ Evaluate the offset choice

            :param ref_cnt: Dict[Grid, Dict[off_str, int]]
            :param rds: Dict[Grid, Dict[off_str, off]]
            :param off: [vec]
            :return: int, how many reads are reduced when choosing this offset
            """
            ret = 0
            for grid, ref_dict in rds.items():
                for off_str, o in ref_dict.items():
                    shifted = o[:]
                    for idx, s in enumerate(off):
                        if shifted[idx] is not None:
                            shifted[idx] += s
                    shifted_str = str(shifted)
                    if shifted_str in ref_cnt[grid]:
                        ret -= 1
            return ret

        def _addOff(ref_cnt: Dict[Grid, Dict[str, int]], rds: Dict[Grid, Dict[str, List[int]]], off: List[int]):
            """ Add the offset choice

            :param ref_cnt: Dict[Grid, Dict[off_str, int]]
            :param rds: Dict[Grid, Dict[off_str, off]]
            :param off: [vec]
            """
            for grid, ref_dict in rds.items():
                for off_str, o in ref_dict.items():
                    shifted = o[:]
                    for idx, s in enumerate(off):
                        if shifted[idx] is not None:
                            shifted[idx] += s
                    shifted_str = str(shifted)
                    if shifted_str in ref_cnt[grid]:
                        ref_cnt[grid][shifted_str] += 1
                    else:
                        ref_cnt[grid][shifted_str] = 1

        def optGreedy(ref_cnt: Dict[Grid, Dict[str, int]], ref_col: List[Dict[Grid, Dict[str, List[int]]]]
                      ) -> List[List[int]]:
            def _Ruse(a: List[int], b: List[int]) -> int:
                x = [i - j for i, j in zip(a, b)]
                pos = 0
                u = 1
                for idx, d in enumerate(self.FOLD):
                    pos += x[idx] % d * u
                    u *= d
                if pos == 0:
                    pos = 1
                    for diff, d in zip(x, self.TILE_DIM):
                        pos *= max(0, d - abs(diff))
                    return pos
                else:
                    return 0

            for grid in ref_cnt.keys():
                ref_cnt[grid] = dict()

            offset = []
            cnt = len(ref_col)
            loc = 0
            while cnt > 0:
                cnt -= 1
                ruse = 1
                if loc >= len(offset):
                    offset.append([0] * len(self.TILE_DIM))
                    best = _evalOff(ref_cnt, ref_col[loc], offset[loc])
                else:
                    best = _rmOff(ref_cnt, ref_col[loc], offset[loc])
                for idx in range(len(offset)):
                    if idx != loc:
                        ruse += _Ruse(offset[loc], offset[idx])
                ruse += sum([abs(o) for o in offset[loc]])
                best = best + 1 / ruse
                bk = best
                no = offset[loc]
                for grid, offset_dict in ref_col[loc].items():
                    for off_str, off in offset_dict.items():
                        for toff_str, off_cnt in ref_cnt[grid].items():
                            toff = eval(toff_str)
                            tshift = [0 if t is None or o is None else t - o
                                      for t, o in zip(toff, off)]
                            tshift = [t - t % s for t, s in zip(tshift, self.STRIDE)]
                            ruse = 1
                            for idx in range(len(offset)):
                                if idx != loc:
                                    ruse += _Ruse(tshift, offset[idx])
                            ruse += sum([abs(o) for o in offset[loc]])
                            nbest = _evalOff(ref_cnt, ref_col[loc], tshift) + 1 / ruse
                            if nbest < best:
                                best = nbest
                                no = tshift[:]
                _addOff(ref_cnt, ref_col[loc], no)
                offset[loc] = no[:]
                if best < bk:
                    cnt = len(ref_col)
                loc = (loc + 1) % len(ref_col)

            return offset

        def count_ref(ref_cnt):
            cnt = 0
            for grid, offset_dict in ref_cnt.items():
                for off in offset_dict:
                    cnt += 1
            return cnt

        def concretize(node):
            from st.expr import BinOp
            if isinstance(node, GridRef):
                # Report this grid reference to the parent above
                off = []
                for i, idx in enumerate(node.indices):
                    while idx.n >= len(off):
                        off.append(None)
                    off[idx.n] = node.offsets[i]
                node.attr['refs'] = {node.grid: {str(off): off}}
                return node.attr['refs']
            ret = dict()
            ref_col = list()
            for child in node.children:
                if not child.get_attr('fixed'):
                    concret_child = concretize(child)
                else:
                    concret_child = dict()
                ref_col.append(concret_child)
                for grid, refdict in concret_child.items():
                    if grid not in ret:
                        ret[grid] = dict()
                    for offset_str, offset in refdict.items():
                        if offset_str not in ret[grid]:
                            ret[grid][offset_str] = offset[:]
            node.attr['refs'] = ret

            # Calculate whether should concretize the reduction operator
            if isinstance(node, Reduction) and len(node.children) >= 4:
                ref_cnt = gridref_count(ref_col)
                init_cnt = count_ref(ref_cnt)

                offset = optGreedy(ref_cnt, ref_col)

                end_cnt = count_ref(ref_cnt)

                node.attr['cnt'] = init_cnt

                if init_cnt > end_cnt * self.THRES:
                    node.attr['offset'] = offset[:]
                    node.attr['cnt'] = end_cnt
                    ret = dict()

                return ret

            return ret

        def conv2dag(node, dag):
            ret = list()
            for child in node.children:
                ret += conv2dag(child, dag)[:]

            if node.get_attr('node_index') in fixed or (isinstance(node, Reduction) and node.get_attr('offset')):
                if node not in dag:
                    dag[node] = dag_opt.DAGnode(node)
                dag[node].depends = ret
                return [dag[node]]

            return ret

        # Gather expression and create expression DAG
        fixed = conv_reduction(self.grids)

        for grid in self.grids:
            grid.out = fixed[grid.out[1].get_attr('node_index')]

        # A stencil expression may have
        # Subexpression that are used more than once (multiple out - explicit)
        #   * This will have to be initialized but not necessarily use scatter
        #   * The benefit of reuse is purely computational way
        #   * The drawback is only memory usage
        #   * Scatter may mitigate some of its benefits
        # Associative operators that consists of terms that share data reads related with shifts (multiple in)
        #   * This may trigger a scatter
        # Common expressions that are related with shifts (multiple out - hidden)
        #   * This will trigger a buffered subexpression

        # A stencil computation have challenges:
        # Expensive vector read
        # Limited data cache/registers
        # Limited instruction cache
        # Redundant computations

        # Tools we have:
        # Temporary buffers to stage partial result
        # Associative operators

        # Buffered subexpression can be evaluated as
        #   * scatter
        #   * gather

        # Evaluation order for scatter:
        #   * Initialization / Reduction
        #   * Scatter

        # decide which ones to instantiate
        for out in fixed.values():
            concretize(out)

        # turn everything into a DAG
        dag = dict()
        for out in fixed.values():
            ret = conv2dag(out, dag)
            if out not in dag:
                dag[out] = dag_opt.DAGnode(out)
                dag[out].depends = ret

        print("Total of {} fixed".format(len(dag)))

        nodes = list(dag.values())
        for n in nodes:
            for d in n.depends:
                dag[d.expr].depending.append(n)
        dag = dag_opt.DAG(nodes)

        # Topological sort with constraint on the maximum constraints of cache usage
        #   * Topology
        #   * Cache: Maximum cache usage
        #   * Fusion: Exposed reuse
        #   * Cache & fusion are related

        seq = dag.sequence(self.MSIZE)

        cnt = dict()
        leaves = list()
        for n in nodes:
            if not n.depends:
                leaves.append(n)
            cnt[n] = len(n.depends)

        last = 0
        groups = list()
        while leaves:
            red_list = list()
            offset = list()
            lsize = 0
            nlast = last
            last_split = 0
            while nlast < len(seq):
                if seq[nlast] not in leaves:
                    break
                if not seq[nlast].expr.get_attr('offset'):
                    if red_list and red_list[-1].expr.get_attr('offset'):
                        break
                    else:
                        red_list.append(seq[nlast])
                        nlast += 1
                        continue
                elif red_list and not red_list[-1].expr.get_attr('offset'):
                    break
                red_list.append(seq[nlast])
                terms = list()
                for red in red_list:
                    ref_col = []
                    for child in red.expr.children:
                        ref_col.append(child.get_attr('refs'))
                    terms = terms + ref_col

                ref_cnt = gridref_count(terms)
                noff = optGreedy(ref_cnt, terms)
                nsize = count_ref(ref_cnt)
                dsplit = 0
                if self.DIM_SPLIT:
                    rst = dict()
                    for n in red_list:
                        rst[n] = list(range(len(n.expr.children)))

                    for dim in range(len(self.TILE_DIM)):
                        cdim = dict()
                        self.split(rst, cdim, dim, len(self.TILE_DIM) - dim, lambda n: n.expr.get_attr('offset'))
                        for n, d in cdim.items():
                            if d:
                                dsplit += 1
                                break

                    for n, d in rst.items():
                        if d:
                            dsplit = 0
                            break

                if (len(red_list) > 1 and (
                            nsize >= lsize + seq[nlast].expr.get_attr('cnt') or
                            nsize + self.K * len(red_list) >= self.KLIMIT
                        )) or (dsplit < last_split):
                    red_list.pop()
                    break
                else:
                    lsize = nsize
                    offset = noff
                    last_split = dsplit
                nlast += 1

            groups.append(red_list)
            last = 0
            for red in red_list:
                l = len(red.expr.children)
                leaves.remove(red)
                red.expr.attr['offset'] = offset[last:(last + l)]
                last += l
            last = nlast

            for n in red_list:
                for d in n.depending:
                    cnt[d] -= 1
                    if cnt[d] == 0:
                        leaves.append(d)

        print("A total of {} stages".format(len(groups)))

        iteration = []
        for d in self.TILE_DIM:
            iteration.append((0, d))

        buf_dict = dict()
        for n in nodes:
            buf_dict[n.expr] = Buffer(n.expr)
            buf_dict[n.expr].iteration = iteration[:]

        for g in self.grids:
            buf_dict[g.out].grid = g

        def replaceBuffer(node, iter_space):
            # update iteration if the result is shifted
            if isinstance(node, Shift):
                shifted_iter = list()
                for space, shift in zip(iter_space, node.shifts):
                    shifted_iter.append((space[0] + shift, space[1] + shift))
            else:
                shifted_iter = iter_space

            for idx in range(len(node.children)):
                if not isinstance(node.children[idx], BufferRead):
                    node.children[idx] = replaceBuffer(node.children[idx], shifted_iter)

            if node in buf_dict:
                niter = list()
                for a, b in zip(buf_dict[node].iteration, shifted_iter):
                    niter.append((min(a[0], b[0]), max(a[1], b[1])))

                buf_dict[node].iteration = niter[:]
                return BufferRead(buf_dict[node])

            return node

        for g in self.grids:
            g.out = replaceBuffer(g.out, iteration)

        assigned = dict()
        avail = list()
        cnt = dict()
        buf_cnt = 0
        for n in nodes:
            cnt[n] = len(n.depending)

        for g in groups:
            for n in g:
                b = buf_dict[n.expr]
                if avail:
                    b.name = avail[-1]
                    avail.pop()
                else:
                    b.name = "buf{}".format(buf_cnt)
                    buf_cnt += 1
                if b.name in assigned:
                    assigned[b.name].append(b)
                else:
                    assigned[b.name] = [b]

                if cnt[n] == 0:
                    avail.append(buf_dict[n.expr].name)

                for d in n.depends:
                    cnt[d] -= 1
                    if cnt[d] == 0:
                        avail.append(buf_dict[d.expr].name)

        self.groups = [[buf_dict[n.expr] for n in g] for g in groups]
        # Buffer needs to be adjusted to the size of the iteration
        print("A total of {} buffers".format(buf_cnt))

    def gencode(self, grid_out: List[Grid], outfile=sys.stdout):
        self.grids = grid_out[:]
        layout = self.layout
        backend = self.backend

        toplevel = CodeBlock()
        fold = self.FOLD[:]
        while len(fold) < len(self.TILE_DIM):
            fold.append(1)
        backend.prequel(toplevel)

        self._calculate()
        if backend is None or layout is None:
            raise ValueError("Run CodeGen.config to set vectorizer and layout")

        buffers = dict()

        for g in self.groups:
            grouplevel = CodeBlock()

            # A computation is consists of three components
            # * construct for vector compute
            # * reference to output
            # * reference of input grid

            def loadData(compGroup, offset, exprs: List[Tuple[Expr, List[int]]], read_bufs, rel=None):
                # Generate the corresponding read
                reading = dict()
                read = dict()
                compGroup.append("// New offset {}".format(str(offset)))

                def prepare_all(expr: Expr, shift: List[int]):
                    if isinstance(expr, GridRef):
                        if expr.grid not in reading:
                            reading[expr.grid] = dict()
                            read[expr.grid] = dict()
                        noff = [o1 + o2 for o1, o2 in zip(expr.offsets, shift)]
                        off = tuple(noff)
                        real_off = [o1 + o2 for o1, o2 in zip(off, offset)]
                        if off in read[expr.grid]:
                            return
                        else:
                            backend.declare_reg(backend.vecreg_name(expr.grid, off), compGroup)
                            read[expr.grid][off] = tuple(real_off)

                        # need to instantiate the read
                        noff[0] = expr.offsets[0] + shift[0] - shift[0] % fold[0]
                        noff0 = noff[0]
                        noff[0] //= fold[0]
                        lreg = tuple(noff)
                        noff[0] = (noff0 + fold[0] - 1) // fold[0]
                        rreg = tuple(noff)
                        if lreg == rreg:
                            real_off[0] = real_off[0] - real_off[0] % fold[0]
                            read_off = tuple(real_off)
                            # only need to read rreg
                            if rreg in reading[expr.grid]:
                                assert (reading[expr.grid][rreg] == read_off)
                            else:
                                reading[expr.grid][rreg] = read_off
                        else:
                            # need to read both
                            roff0 = real_off[0]
                            real_off[0] = real_off[0] - real_off[0] % fold[0]
                            read_off = tuple(real_off)
                            if lreg in reading[expr.grid]:
                                assert (reading[expr.grid][lreg] == read_off)
                            else:
                                reading[expr.grid][lreg] = read_off
                            real_off[0] = roff0 + fold[0] - roff0 % fold[0]
                            read_off = tuple(real_off)
                            if rreg in reading[expr.grid]:
                                assert (reading[expr.grid][rreg] == read_off)
                            else:
                                reading[expr.grid][rreg] = read_off
                    else:
                        for child in expr.children:
                            prepare_all(child, shift)

                for expr, shift in exprs:
                    prepare_all(expr, shift)

                read_block = CodeBlock()
                vecs = dict()
                for n, regs in reading.items():
                    vecs[n] = dict()
                    for reg in sorted(regs.keys()):
                        assigned = False
                        if read_bufs[n][reg] != regs[reg]:
                            # this need reading
                            for reg_b, reg_b_val in read_bufs[n].items():
                                if regs[reg] == reg_b_val:
                                    read_block.append("{} = {};".format(backend.vecbuf_name(n, reg),
                                                                        backend.vecbuf_name(n, reg_b)))
                                    assigned = True
                        else:
                            assigned = True
                        if assigned:
                            vecs[n][regs[reg]] = backend.vecbuf_name(n, reg)
                            read_bufs[n][reg] = regs[reg]

                def read_vec(n, ovec, pos, vec_index):
                    if pos in vecs[n]:
                        read_block.append("{} = {};".format(ovec, vecs[n][pos]))
                        return vec_index
                    dim = 0
                    while dim < len(self.TILE_DIM) and pos[dim] % fold[dim] == 0:
                        dim += 1
                    if dim == len(self.TILE_DIM):
                        # this is perfectly aligned
                        backend.read_aligned(n, pos, ovec, read_block, rel)
                    else:
                        # This requires a blend on the dim-th dimension
                        posa = list(pos)
                        posa[dim] = posa[dim] + fold[dim] - posa[dim] % fold[dim]
                        posa = tuple(posa)
                        if posa in vecs[n]:
                            namea = vecs[n][posa]
                        else:
                            namea = backend.vectmp_name(vec_index)
                            backend.declare_vec(namea, read_block)
                            vec_index += 1
                            vec_index = read_vec(n, namea, posa, vec_index)
                        posb = list(pos)
                        posb[dim] = posb[dim] - posb[dim] % fold[dim]
                        posb = tuple(posb)
                        if posb in vecs[n]:
                            nameb = vecs[n][posb]
                        else:
                            nameb = backend.vectmp_name(vec_index)
                            backend.declare_vec(nameb, read_block)
                            vec_index += 1
                            vec_index = read_vec(n, nameb, posb, vec_index)
                        backend.merge(ovec, nameb, namea, dim, pos[dim] % fold[dim], read_block)
                    vecs[n][pos] = ovec
                    return vec_index

                vec_index = 0
                for n, regs in reading.items():
                    for reg in sorted(regs.keys()):
                        pos = regs[reg]
                        if read_bufs[n][reg] != pos:
                            name = backend.vecbuf_name(n, reg)
                            vec_index = read_vec(n, name, pos, vec_index)

                for n, regs in reading.items():
                    for reg, val in regs.items():
                        read_bufs[n][reg] = val

                for n, regs in read.items():
                    for reg, val in regs.items():
                        if val not in vecs[n]:
                            name = backend.vectmp_name(vec_index)
                            backend.declare_vec(name, read_block)
                            vec_index += 1
                            vec_index = read_vec(n, name, val, vec_index)
                        backend.store_vecbuf(vecs[n][val], backend.vecreg_name(n, reg), read_block)

                if len(read) > 0:
                    compGroup.append(read_block)

            def genScatter(compGroup, offset, srcs: Dict[Buffer, List[int]], read_bufs, rel=None, dimrels=None):
                exprs = []
                for n, coll in srcs.items():
                    for idx in coll:
                        exprs.append((n.rhs.children[idx], n.rhs.get_attr('offset')[idx]))
                if self.ALIGNED:
                    if rel is not None and self.LAYOUTREL:
                        loadData(compGroup, offset, exprs, read_bufs, dimrels)
                    else:
                        loadData(compGroup, offset, exprs, read_bufs, rel)
                # Generate the compute
                vec = backend.genVectorLoop(compGroup)
                for n, coll in srcs.items():
                    for idx in coll:
                        shift = n.rhs.get_attr('offset')[idx]
                        noff = [b + a for a, b in zip(shift, offset)]
                        if self.DIRECT:
                            lhs = backend.genStoreLoc(n.grid, shift, offset, rel, dimrels)
                        else:
                            lhs = backend.gen_lhs(n, noff, rel, dimrels)
                        vec.append(lhs +
                                   " {}= {};".format(n.rhs.terms_op[idx].value,
                                                     backend.gen_rhs(n.rhs.children[idx], shift, offset, rel, dimrels)))

            def find_refs(reads, expr: Expr, shift):
                if isinstance(expr, GridRef):
                    if expr.grid not in reads:
                        reads[expr.grid] = dict()
                    noff = [o1 + o2 for o1, o2 in zip(expr.offsets, shift)]
                    noff[0] = expr.offsets[0] + shift[0] - shift[0] % fold[0]
                    noff0 = noff[0]
                    noff[0] //= fold[0]
                    reads[expr.grid][tuple(noff)] = None
                    if noff0 % fold[0] != 0:
                        # more than one read is needed
                        noff[0] = (noff0 + fold[0] - 1) // fold[0]
                        reads[expr.grid][tuple(noff)] = None
                else:
                    for child in expr.children:
                        find_refs(reads, child, shift)

            def collect_refs(reads, expr: Expr, shift):
                if isinstance(expr, GridRef):
                    if expr.grid not in reads:
                        reads[expr.grid] = dict()
                    off = [None] * len(shift)
                    for idx, o in zip(expr.indices, expr.offsets):
                        off[idx.n] = o
                    noff = [o1 + o2 if o1 is not None else None for o1, o2 in zip(off, shift)]
                    reads[expr.grid][tuple(noff)] = None
                else:
                    for child in expr.children:
                        collect_refs(reads, child, shift)

            def reduce(exprs: Dict[Buffer, Expr]):
                initlevel = CodeBlock()
                grouplevel.append(initlevel)
                shift = [0] * len(self.TILE_DIM)
                # collect the necessary grid read
                if self.ALIGNED:
                    read_bufs = dict()
                    for buf, expr in exprs.items():
                        find_refs(read_bufs, expr, shift)
                    # declare all reads
                    for grid, vec_shifts in read_bufs.items():
                        for vec_shift in vec_shifts.keys():
                            backend.declare_vecbuf(grid, vec_shift, initlevel)

                reads = dict()
                for buf, expr in exprs.items():
                    collect_refs(reads, expr, shift)

                offset = [0] * len(self.TILE_DIM)

                def reduceDim(offset, cur_dim, nested: CodeBlock, rel, dimrels):
                    if cur_dim < 0:
                        create_exprs = []
                        for expr in exprs.values():
                            create_exprs.append((expr, shift))
                        comp = CodeBlock()
                        nested.append(comp)
                        if self.ALIGNED:
                            if rel is not None and self.LAYOUTREL:
                                loadData(comp, offset, create_exprs,
                                         read_bufs, dimrels)
                            else:
                                loadData(comp, offset, create_exprs, read_bufs, rel)
                        vec = backend.genVectorLoop(comp)
                        for buf, expr in exprs.items():
                            if self.DIRECT:
                                lhs = backend.genStoreLoc(buf.grid, shift, offset, rel, dimrels)
                            else:
                                lhs = backend.gen_lhs(buf, offset, rel, dimrels)
                            vec.append("{} = {};".format(lhs, backend.gen_rhs(expr, shift, offset, rel, dimrels)))

                        return
                    l = 1048576
                    r = -1048576
                    if self.ALIGNED:
                        for n, refs in reads.items():
                            for ref in refs:
                                l = min(l, l if ref[cur_dim] is None else ref[cur_dim])
                                r = max(r, r if ref[cur_dim] is None else ref[cur_dim])

                    lb = max(0, 0 - l)
                    tend = self.TILE_DIM[cur_dim] - fold[cur_dim] + 1
                    rb = min(tend, tend - r)

                    lb = lb + lb % fold[cur_dim]
                    rb = rb - rb % fold[cur_dim]
                    if lb > rb:
                        lb = rb
                    lb = lb + (rb - lb) % fold[cur_dim]
                    if rb - lb <= fold[cur_dim]:
                        # no loop is necessary
                        rb = lb

                    # unroll part
                    if self.LAYOUTREL:
                        dimrels.append(None)
                    for i in range(0, lb):
                        if i % fold[cur_dim] == 0:
                            # generate code only when it is aligned
                            offset[cur_dim] = i
                            reduceDim(offset, cur_dim - 1, nested, rel, dimrels)
                    if self.LAYOUTREL:
                        dimrels.pop()

                    # for-loop part
                    if self.LAYOUTREL:
                        dimrels.append(backend.index_name(cur_dim))
                    if lb < rb:
                        start = 0
                        if rel:
                            start = "rel"
                        subblock = CodeBlock()
                        nested.append(subblock)
                        subblock.append("long {} = {};".format(backend.rel_name(cur_dim), start))
                        idx_name = backend.index_name(cur_dim)
                        reln = st.expr.ConstRef(backend.rel_name())
                        subblock.append(
                            "for (long {} = 0; {} < {}; {} += {})".format(
                                idx_name, idx_name, rb - lb, idx_name, fold[cur_dim]))
                        newlevel = CodeBlock()
                        subblock.append(newlevel)
                        newlevel.append("long {} = {};".format(backend.rel_name(), backend.rel_name(cur_dim)))

                        for i in range(fold[cur_dim]):
                            if (i + lb) % fold[cur_dim] == 0:
                                # generate code only when it is aligned
                                offset[cur_dim] = i + lb
                                reduceDim(offset, cur_dim - 1, newlevel, reln, dimrels)

                        discardlevel = CodeBlock()
                        for i in range(fold[cur_dim]):
                            t = i + rb - fold[cur_dim]
                            if t % fold[cur_dim] == 0:
                                # generate code only when it is aligned
                                offset[cur_dim] = t
                                reduceDim(offset, cur_dim - 1, discardlevel, reln, dimrels)

                        newlevel.append("{} += {};".format(backend.rel_name(cur_dim), backend.stride(cur_dim)))
                    if self.LAYOUTREL:
                        dimrels.pop()

                    # unroll part
                    if self.LAYOUTREL:
                        dimrels.append(None)
                    for i in range(rb, self.TILE_DIM[cur_dim]):
                        if i % fold[cur_dim] == 0:
                            # generate code only when it is aligned
                            offset[cur_dim] = i
                            reduceDim(offset, cur_dim - 1, nested, rel, dimrels)
                    if self.LAYOUTREL:
                        dimrels.pop()

                reduceDim(offset, len(self.TILE_DIM) - 1, initlevel, None, [])

            def scatter(rst):
                initlevel = CodeBlock()
                grouplevel.append(initlevel)
                # collect the necessary grid read
                read_bufs = dict()
                offsets = dict()

                # collect the necessary grid read
                for n, indices in rst.items():
                    offs = n.rhs.get_attr('offset')
                    for idx in indices:
                        term_off = offs[idx]
                        find_refs(read_bufs, n.rhs.children[idx], term_off)
                        toff = tuple(term_off)
                        if toff not in offsets:
                            offsets[toff] = dict()
                        if n not in offsets[toff]:
                            offsets[toff][n] = []
                        offsets[toff][n].append(idx)

                if self.ALIGNED:
                    # declare all reads
                    for grid, vec_shifts in read_bufs.items():
                        for vec_shift in vec_shifts.keys():
                            backend.declare_vecbuf(grid, vec_shift, initlevel)

                cur_offsets = list(offsets.keys())
                offset = [0] * len(self.TILE_DIM)

                def scatterDim(offset, cur_offsets, cur_dim, nested, rel, dimrels):
                    if cur_dim < 0:
                        if not cur_offsets:
                            raise ValueError("no!!!!!")
                        all_expr = dict()
                        for off in cur_offsets:
                            for b, idx in offsets[off].items():
                                if b not in all_expr:
                                    all_expr[b] = []
                                all_expr[b] += idx
                        newlev = CodeBlock()
                        nested.append(newlev)
                        genScatter(newlev, offset, all_expr, read_bufs, rel, dimrels)

                        return
                    l = 1048576
                    r = -1048576

                    # Check the offsets of reads
                    if self.ALIGNED:
                        reads = dict()
                        for off in cur_offsets:
                            for b, indices in offsets[off].items():
                                offs = b.rhs.get_attr("offset")
                                for idx in indices:
                                    collect_refs(reads, b.rhs.children[idx], offs[idx])

                        for n, refs in reads.items():
                            for ref in refs:
                                l = min(l, l if ref[cur_dim] is None else ref[cur_dim])
                                r = max(r, r if ref[cur_dim] is None else ref[cur_dim])
                    else:
                        l = 0
                        r = 0

                    lo = 1048576
                    ro = -1048576
                    # Check the offsets of shift
                    for off in cur_offsets:
                        lo = min(lo, off[cur_dim])
                        ro = max(ro, off[cur_dim])

                    tlb = 0 - ro
                    tend = self.TILE_DIM[cur_dim] - fold[cur_dim] + 1
                    trb = tend - lo
                    
                    lb = max(tlb, 0 - min(l, 0), 0 - min(lo, 0))
                    rb = min(trb, tend - max(ro, 0), tend - max(r, 0))

                    if lb > rb:
                        lb = rb
                    lb = lb + (rb - lb) % fold[cur_dim]
                    if rb - lb <= fold[cur_dim]:
                        # no loop is necessary
                        rb = lb

                    # unroll part
                    if self.LAYOUTREL:
                        dimrels.append(None)
                    for i in range(tlb, lb):
                        nex_offsets = []
                        for off in cur_offsets:
                            t = i + off[cur_dim]
                            if t >= 0 and t < self.TILE_DIM[cur_dim] and t % fold[cur_dim] == 0:
                                nex_offsets.append(off)

                        if nex_offsets:
                            # generate code only when it is aligned
                            offset[cur_dim] = i
                            scatterDim(offset, nex_offsets, cur_dim - 1, nested, rel, dimrels)

                    if self.LAYOUTREL:
                        dimrels.pop()

                    # for-loop part
                    if self.LAYOUTREL:
                        dimrels.append(backend.index_name(cur_dim))

                    if lb < rb:
                        start = 0
                        if rel:
                            start = "rel"
                        subblock = CodeBlock()
                        nested.append(subblock)
                        subblock.append("long {} = {};".format(backend.rel_name(cur_dim), start))
                        idx_name = backend.index_name(cur_dim)
                        reln = st.expr.ConstRef(backend.rel_name())
                        subblock.append(
                            "for (long {} = 0; {} < {}; {} += {})".format(
                                idx_name, idx_name, rb - lb, idx_name, fold[cur_dim]))
                        newlevel = CodeBlock()
                        subblock.append(newlevel)
                        newlevel.append("long {} = {};".format(backend.rel_name(), backend.rel_name(cur_dim)))

                        for i in range(fold[cur_dim]):
                            nex_offsets = []
                            for off in cur_offsets:
                                t = lb + i + off[cur_dim]
                                if t >= 0 and t < self.TILE_DIM[cur_dim] and t % fold[cur_dim] == 0:
                                    nex_offsets.append(off)

                            if nex_offsets:
                                # generate code only when it is aligned
                                offset[cur_dim] = i + lb
                                scatterDim(offset, nex_offsets, cur_dim - 1, newlevel, reln, dimrels)

                        discardlevel = CodeBlock()
                        for i in range(fold[cur_dim]):
                            nex_offsets = []
                            for off in cur_offsets:
                                t = rb - fold[cur_dim] + i + off[cur_dim]
                                if t >= 0 and t < self.TILE_DIM[cur_dim] and t % fold[cur_dim] == 0:
                                    nex_offsets.append(off)

                            if nex_offsets:
                                # generate code only when it is aligned
                                offset[cur_dim] = i + rb - fold[cur_dim]
                                scatterDim(offset, nex_offsets, cur_dim - 1, discardlevel, reln, dimrels)

                        newlevel.append("{} += {};".format(
                            backend.rel_name(cur_dim), self.backend.stride(cur_dim)))
                    if self.LAYOUTREL:
                        dimrels.pop()

                    # unroll part
                    if self.LAYOUTREL:
                        dimrels.append(None)
                    for i in range(rb, trb):
                        nex_offsets = []
                        for off in cur_offsets:
                            t = i + off[cur_dim]
                            if t >= 0 and t < self.TILE_DIM[cur_dim] and t % fold[cur_dim] == 0:
                                nex_offsets.append(off)

                        if nex_offsets:
                            # generate code only when it is aligned
                            offset[cur_dim] = i
                            scatterDim(offset, nex_offsets, cur_dim - 1, nested, rel, dimrels)
                    if self.LAYOUTREL:
                        dimrels.pop()

                scatterDim(offset, cur_offsets, len(self.TILE_DIM) - 1, initlevel, None, [])

            def scatter_unrolled():
                # collect the necessary grid read
                read_bufs = dict()
                if self.ALIGNED:
                    for n in g:
                        offs = n.rhs.get_attr('offset')
                        for idx, term_off in enumerate(offs):
                            find_refs(read_bufs, n.rhs.children[idx], term_off)
                    # declare all reads
                    for grid, vec_shifts in read_bufs.items():
                        for vec_shift in vec_shifts.keys():
                            backend.declare_vecbuf(grid, vec_shift, grouplevel)
                # Generate all possible input offset
                offsets = dict()
                off = [0] * len(self.TILE_DIM)
                while True:
                    for n in g:
                        offs = n.rhs.get_attr('offset')
                        for idx, term_off in enumerate(offs):
                            noff = []
                            for o1, o2 in zip(off, term_off):
                                noff.append(o1 - o2)
                            t = tuple(reversed(noff))
                            if t not in offsets:
                                offsets[t] = dict()
                            d = offsets[t]
                            if n not in d:
                                d[n] = []
                            d[n].append(idx)
                    cur_dim = 0
                    while cur_dim < len(off) and off[cur_dim] >= self.TILE_DIM[cur_dim] - fold[cur_dim]:
                        off[cur_dim] = 0
                        cur_dim += 1
                    if cur_dim == len(off):
                        break
                    off[cur_dim] += fold[cur_dim]

                grouplevel.append('// A group with a total of {} offsets'.format(len(offsets)))
                for off in sorted(offsets.keys()):
                    # grouplevel.append('// Offset {}'.format(str(tuple(off))))
                    compGroup = CodeBlock()
                    grouplevel.append(compGroup)
                    genScatter(compGroup, tuple(reversed(off)), offsets[off], read_bufs)


            red_init = dict()
            scatter_phase = False
            for n in g:
                if n.name not in buffers:
                    n.name = backend.declare_buf(n, toplevel)
                    buffers[n.name] = True

                if n.rhs.get_attr('offset'):
                    # When using scatter
                    from st.expr import conv_expr
                    val = conv_expr(Reduction.identity[n.rhs.operator])
                    red_init[n] = val
                    scatter_phase = True
                else:
                    # When not using scatter
                    red_init[n] = n.rhs

            # Computation have periods
            # * Within a period the result are unrolled
            # * Across period only shared read are used

            reduce(red_init)

            # Generate the computation
            if scatter_phase:
                if self.UNROLL:
                    scatter_unrolled()
                else:
                    rst = dict()
                    for n in g:
                        rst[n] = list(range(len(n.rhs.children)))

                    if self.DIM_SPLIT:
                        for dim in range(len(self.TILE_DIM)):
                            cdim = dict()
                            self.split(rst, cdim, dim, len(self.TILE_DIM) - dim, lambda n: n.rhs.get_attr('offset'))
                            for n, d in cdim.items():
                                if d:
                                    scatter(cdim)
                                    break

                    for n, d in rst.items():
                        if d:
                            scatter(rst)
                            break

            store = CodeBlock()
            store_q = 0
            if self.TILE:
                if not self.DIRECT:
                    for n in g:
                        if n.grid:
                            store_q = max(store_q, n.grid.dims)
                            backend.storeTile(n, store)
                    if store_q:
                        storelev = backend.genStoreTileLoop(grouplevel, store_q)
                        storelev.append(store)
            else:
                for n in g:
                    if n.grid:
                        store_q = True
                        backend.declare_gridref(n.grid, grouplevel)
                        backend.store(n, store)
                if store_q:
                    backend.genStoreLoop(grouplevel)
                    grouplevel.append(store)

            toplevel.append(grouplevel)

        toplevel.to_str(outfile, 0)

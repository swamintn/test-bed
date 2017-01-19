#!/usr/bin/env python

from __future__ import print_function

from   math     import ceil, log
from   operator import add

from tree import display_tree


# =============================== SEGMENT TREE =================================

class SegmentTree(object):
    """
    A Segment Tree
    
    1) n represents the number of leaf nodes
    2) A segment tree is a full balanced binary tree. So, the number of internal
       nodes is n-1 and the height is ceil(log n).

    This segment tree is used to perform range minimum queries.

    See: http://www.geeksforgeeks.org/segment-tree-set-1-sum-of-given-range/
         http://www.geeksforgeeks.org/segment-tree-set-1-range-minimum-query/
    """
    INF = float("inf")

    def __init__(self, in_arr):
        self.in_arr = in_arr
        self.construct_ST(in_arr)

    def _construct_ST_util(self, arr, start, end, tree_node):
        """
        Recursive function for constructing Segment Tree
        start and end represent positions in input array, arr and tree_node
        represents the position in the tree_arr

        Computes and returns the value at self.tree_arr[tree_node]
        """
        if start == end:
            self.tree_arr[tree_node] = (start, arr[start])
            return self.tree_arr[tree_node]
        mid = start + ((end - start) // 2)
        pos1, val1 = self._construct_ST_util(arr, start, mid, 2*tree_node + 1)
        pos2, val2 = self._construct_ST_util(arr, mid+1, end, 2*tree_node + 2)
        new_pos, new_val = min([(pos1, val1), (pos2, val2)], key=lambda s: s[1])
        self.tree_arr[tree_node] = (new_pos, new_val)
        return self.tree_arr[tree_node]

    def construct_ST(self, in_arr):
        """
        Construct Segment Tree from an input array

        The actual tree is stored in a tree_arr, an array implementation of a 
        tree
        """
        self.n      = len(in_arr)
        # Height is max. number of edges from root to some leaf
        # Number of levels in the tree is height + 1
        tree_ht     = int(ceil(log(self.n, 2)))
        tree_arr_sz = 2**(tree_ht+1) - 1
        self.tree_arr = [None for _ in xrange(tree_arr_sz)]
        self._construct_ST_util(in_arr, 0, self.n - 1, 0)

    def _calc_range_util(self, seg_start, seg_end, qs, qe, tree_node):
        """
        seg_start and seg_end represent the range covered by this tree_node
        qs and qe represent the query range we want

        If segment of this node is part of given range, return the sum value
        If segment of this node is outside the given range, return infinity
        Else, recurse to the left and right sub-trees

        """
        if qs <= seg_start and qe >= seg_end:
            return self.tree_arr[tree_node]
        if qe < seg_start or qs > seg_end:
            return (None, self.INF)
        mid = seg_start + ((seg_end - seg_start) // 2)
        pos1, val1 = self._calc_range_util(seg_start, mid, qs, qe, 2*tree_node + 1)
        pos2, val2 = self._calc_range_util(mid+1, seg_end, qs, qe, 2*tree_node + 2)
        return min([(pos1, val1), (pos2, val2)], key=lambda s: s[1])

    def calc_range(self, start, end):
        """
        Get the sum for the specified range, start and end inclusive
        """
        return self._calc_range_util(0, self.n - 1, start, end, 0)


# ==================== HISTOGRAM SOLVER USING SEGMENT TREE =====================

class HistogramSolver(object):
    """Solve the maximum area histogram problem"""

    def calc_area(self, start, end, height):
        if not start or not end:
            return -float("inf")
        return ((end - start) + 1) * height

    def _recursive_solve(self, start, end, arr, st):
        """
        Returns (start_pos, end_pos, height) of max. area rectangle
        """
        if end < start:
            return (None, None, -float("inf"))
        if start == end:
            return (start, end, arr[start])
        min_pos, min_val = st.calc_range(start, end)
        no_of_bars = (end - start) + 1
        return max([
            self._recursive_solve(start, min_pos-1, arr, st),
            self._recursive_solve(min_pos + 1, end, arr, st),
            (start, end, min_val)
            ], key=lambda s: self.calc_area(*s))

    def solve(self, arr):
        st = SegmentTree(arr)
        start, end, ht = self._recursive_solve(0, len(arr) - 1, arr, st)
        print("Max. rectangle portion in histogram:")
        print("Starting position", start)
        print("Ending position", end)
        print("Height", ht)
        print("Area", self.calc_area(start, end, ht))

    def pretty_print(self, arr):
        max_sz = max(arr)
        for sz in xrange(max_sz, 0, -1):
            my_str = ""
            for el in arr:
                my_str = my_str + (" | " if el >= sz else "   ")
            print(my_str)

# ======================== HISTOGRAM SOLVER USING STACK ========================

def largest_histogram_area(hist):
    # Trick: Find Max. area at each position assuming that bar is included
    i, n = 0, len(hist)
    # Stack ops
    stack = []
    is_empty = lambda s: len(s) == 0
    top = -1
    # Max_area
    max_area = 0
    # Maintain an increasing stack of values. But save the indices and not the
    # actual elements. Equal values are allowed to be on the stack.
    while i < n:
        if is_empty(stack) or hist[i] >= hist[stack[top]]:
            stack.append(i)
            i += 1
        else:
            top_ind = stack.pop()
            while not is_empty(stack) and hist[stack[top]] == hist[top_ind]:
                top_ind = stack.pop()
            area = hist[top_ind] * (i if is_empty(stack) else (i - stack[top] - 1))
            max_area = max(max_area, area)
    while not is_empty(stack):
        top_ind = stack.pop()
        while not is_empty(stack) and hist[stack[top]] == hist[top_ind]:
            top_ind = stack.pop()
        area = hist[top_ind] * (i if is_empty(stack) else (i - stack[top] - 1))
        max_area = max(max_area, area)
    return max_area


# ========================== MAX. SQUARE SUBMATRIX =============================

def get_max_square_submatrix_area(mat):
    m, n = len(mat), len(mat[0])
    dp = [[None for j in xrange(n)] for i in xrange(m)]
    max_size = 0
    for i in xrange(m):
        for j in xrange(n):
            if i == 0 or j == 0:
                dp[i][j] = mat[i][j]
            elif mat[i][j] == 0:
                dp[i][j] = 0
            else:
                size = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                if size > max_size: max_size = size
                dp[i][j] = size
    # Return area
    return max_size*max_size

# ========================== MAX. RECTANGLE SUBMATRIX ==========================

def get_max_rectangle_submatrix_area(mat):
    m, n = len(mat), len(mat[0])
    prev_row, cur_row = [0 for _ in xrange(n)], [0 for _ in xrange(n)]
    max_area = 0
    for row in xrange(m):
        for col in xrange(n):
            cur_row[col] = (prev_row[col] + 1) if mat[row][col] else 0
        area = largest_histogram_area(cur_row)
        max_area = max(max_area, area)
        prev_row = cur_row
    return max_area

# ================================== MAIN ======================================

def main():
    in_mat = [[0,1,1,0,1,1],
              [1,1,0,1,1,1],
              [0,1,1,1,0,0],
              [1,1,1,1,0,0],
              [1,1,1,1,1,0],
              [0,1,1,1,0,1]]
    print("Input matrix:\n", "\n".join(" ".join(map(str, row)) for row in in_mat), sep="")
    print("Max. sub-square area", get_max_square_submatrix_area(in_mat))
    print("Max. sub-rectangle area", get_max_rectangle_submatrix_area(in_mat))
    print("\n")

    arr = [1,2,4,7,3,0,1]
    h = HistogramSolver()
    print("Input histogram:")
    h.pretty_print(arr)
    print(arr)
    h.solve(arr)

    arr = [1,3,3,2,5,5,7]
    h = HistogramSolver()
    print("Input histogram:")
    h.pretty_print(arr)
    print(arr)
    h.solve(arr)
    print("Solution using stacks, Area =", largest_histogram_area(arr))

if __name__ == '__main__':
    main()

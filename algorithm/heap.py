import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __lt__(self, other):
        return self.val < other.val


def merge_k_lists(lists):
    # 小顶堆
    min_heap = []

    # 初始化堆
    for node in lists:
        if node:
            heapq.heappush(min_heap, node)

    # 虚拟头节点
    dummy = ListNode()
    current = dummy

    # 合并过程
    while min_heap:
        # 弹出堆顶元素
        smallest_node = heapq.heappop(min_heap)
        current.next = smallest_node
        current = current.next

        # 将下一个节点加入堆
        if smallest_node.next:
            heapq.heappush(min_heap, smallest_node.next)

    return dummy.next


def find_kth_largest(nums, k):
    # 使用小顶堆
    min_heap = []

    # 构建大小为K的小顶堆
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    # 堆顶元素即为第K大的元素
    return min_heap[0]


# 示例
list1 = ListNode(1, ListNode(4, ListNode(5)))
list2 = ListNode(1, ListNode(3, ListNode(4)))
list3 = ListNode(2, ListNode(6))

merged_head = merge_k_lists([list1, list2, list3])

# 打印合并后的链表
current = merged_head
while current:
    print(current.val, end=" -> ")
    current = current.next
print("\n")

# 示例
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(find_kth_largest(nums, k))  # 输出: 5


def combine_unique(nums, k):
    def backtrack(start, path):
        if len(path) == k:
            results.append(path[:])
            return
        for i in range(start, len(nums)):
            # 跳过重复的元素
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    nums.sort()  # 先排序
    results = []
    backtrack(0, [])
    return results


# 示例
nums = [1, 2, 2]
k = 2
combinations = combine_unique(nums, k)
for comb in combinations:
    print(comb)


def is_valid(board, row, col, num):
    # 检查行是否有重复
    for i in range(9):
        if board[row][i] == num:
            return False

    # 检查列是否有重复
    for i in range(9):
        if board[i][col] == num:
            return False

    # 检查3 x 3的子宫格是否有重复
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    return True


def solve_shuduku(board):
    empty = find_empty(board)
    if not empty:
        return True  # 所有格子都填满，求解成功

    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_shuduku(board):
                return True

            board[row][col] = 0  # 回溯

    return False


def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)

    return None


# 示例
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

if solve_shuduku(board):
    for row in board:
        print(row)
else:
    print("No solution exists.")


def letter_combinations(digits):
    if not digits:
        return []

    phone_map = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}

    def backtrack(index, path):
        if index == len(digits):
            results.append(path)
            return

        possible_letters = phone_map[digits[index]]
        for letter in possible_letters:
            backtrack(index + 1, path + letter)

    results = []
    backtrack(0, "")
    return results


# 示例
digits = "23"
combinations = letter_combinations(digits)
for comb in combinations:
    print(comb)

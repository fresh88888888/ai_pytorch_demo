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

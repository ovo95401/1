# 1
10
1. 冒泡排序（Bubble Sort）
原理：重复遍历列表，比较相邻元素并交换，直到列表有序。

时间复杂度：O(n²)

python
复制
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
2. 快速排序（Quick Sort）
原理：分治法，选择一个基准元素，将数组分为两部分，递归排序。

时间复杂度：O(n log n)

python
复制
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
3. 归并排序（Merge Sort）
原理：分治法，将数组分成两半，分别排序后合并。

时间复杂度：O(n log n)

python
复制
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
4. 二分查找（Binary Search）
原理：在有序数组中查找目标值，每次将搜索范围减半。

时间复杂度：O(log n)

python
复制
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
5. 深度优先搜索（DFS）
原理：用于遍历或搜索树或图，沿着一条路径深入直到无法继续，再回溯。

时间复杂度：O(V + E)，V为顶点数，E为边数

python
复制
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
6. 广度优先搜索（BFS）
原理：用于遍历或搜索树或图，逐层扩展。

时间复杂度：O(V + E)

python
复制
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            print(vertex, end=" ")
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
7. Dijkstra算法
原理：用于单源最短路径问题，适用于加权图。

时间复杂度：O(V²)（使用优先队列可优化到O(E + V log V)）

python
复制
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    while heap:
        current_distance, current_node = heapq.heappop(heap)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    return distances
8. 动态规划（Dynamic Programming）
原理：将问题分解为子问题，保存子问题的解，避免重复计算。

示例：斐波那契数列

python
复制
def fibonacci(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
9. KMP算法（Knuth-Morris-Pratt）
原理：用于字符串匹配，利用部分匹配表跳过不必要的比较。

时间复杂度：O(n + m)

python
复制
def kmp_search(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length-1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = build_lps(pattern)
    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
            if j == len(pattern):
                return i - j
        else:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return -1
10. 贪心算法（Greedy Algorithm）
原理：每一步选择当前最优解，希望最终得到全局最优解。

示例：找零问题

python
复制
def coin_change(coins, amount):
    coins.sort(reverse=True)
    result = []
    for coin in coins:
        while amount >= coin:
            amount -= coin
            result.append(coin)
    return result

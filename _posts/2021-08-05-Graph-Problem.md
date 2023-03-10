---
title: Graph Problem
author: Xiao Fan
date: 2021-08-05 9:52 +0800
categories: [leetcode notes]
tags: [leetcode]
math: true
mermaid: true
---



图问题是一种经典的问题

## [Leetcode. 802 找到最终的安全状态](https://leetcode-cn.com/problems/find-eventual-safe-states/)

### 问题

在有向图中，以某个节点为起始节点，从该点出发，每一步沿着图中的一条有向边行走。如果到达的节点是终点（即它没有连出的有向边），则停止。

对于一个起始节点，如果从该节点出发，无论每一步选择沿哪条有向边行走，最后必然在有限步内到达终点，则将该起始节点称作是**安全**的。

返回一个由图中所有安全的起始节点组成的数组作为答案。答案数组中的元素应当按**升序**排列。

该有向图有`n`个节点，按0到`n - 1`编号，其中`n`是`graph`的节点数。图以下述形式给出: `graph[i]`是编号`j`节点的一个列表，满足`(i, j)`是图的一条有向边。

### 示例

 ![Illustration of graph](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/03/17/picture1.png)

示例 1：

```
输入：graph = [[1,2],[2,3],[5],[0],[5],[],[]]
输出：[2,4,5,6]
解释：示意图如上。
```


示例 2：

```
输入：graph = [[1,2,3,4],[1,2],[3,4],[0,4],[]]
输出：[4]
```

### 题解

这道题有2种解法，第一种为*DFS + 三色标记法*。

为了确定一个节点是否为安全，需要判断这个节点的路径上是否有成环的，成环即在递归的过程种发现遇到之前碰到过的节点，那么我们可以在递归的过程中暂时性地将所有遇到过的节点标记为1，如果在递归的过程中碰到了标记为1的节点，说明成环了，这时整个递归链上的所有节点都是成环的，直接退出，这时1表示这些节点都是不安全的，如果走到最后都没有成环的节点，说明整个链条上的所有节点都是安全的，将这些节点标记为2之后返回。

由于每一条边都会被加入DFS递归，同时每一个节点也都会被遍历一次，因此时间复杂度为$$O(M + N)$$,M为边的总数，N为节点总数

```java
public List<Integer> eventualSafeNodes(int[][] graph) {
    // DFS + 三色标记法,color==0表示尚未搜索，color==1表示在递归栈中或位于环中，color==2表示安全
    List<Integer> ret = new ArrayList<>();
    int len = graph.length;
    int[] color = new int[len];
    for (int i = 0; i < len; i++) {
        if (isSafe(i, graph, color)) {
            ret.add(i);
        }
    }
    return ret;
}

private boolean isSafe(int cur, int[][] graph, int[] color) {
    if (color[cur] != 0) {
        return color[cur] == 2;
    }
    color[cur] = 1;  // 标记位于递归栈上
    for (int i : graph[cur]) {
        if (!isSafe(i, graph, color)) {
            return false;
        }
    }
    color[cur] = 2;  // 递归结束都没有发现成环，说明是安全的
    return true;
}
```

第二种方法为*拓扑排序法*

首先回顾一下拓扑排序。拓扑排序即保证所有前驱节点（出度节点）一定在后驱节点（被指向的节点）的前面。首先我们统计所有节点的入度，将入度为0的节点弹出并将这个节点指向的所有节点的入度都减一，然后不断寻找入度为0的节点弹出，重复上述步骤，弹出的节点顺序即为拓扑排序的顺序

![微信图片_20210805153823](/assets/img/posts/Graph_Problem/微信图片_20210805153823.png)

根据题意，一个节点**安全**的标准是这个节点的出度为0或者这个节点指向的所有节点都是安全的。 这其实就是反向的拓扑排序，即先对图取反图，然后进行拓扑排序，将所有入度为0的节点加入答案

```java
public List<Integer> eventualSafeNodes(int[][] graph) {
    // 反图+拓扑排序
    int len = graph.length;
    List<Integer> ret = new ArrayList<>();
    List<List<Integer>> reverse = new ArrayList<>();
    int[] degrees = new int[len];  // 入度
    for (int i = 0; i < len; i++) {
        reverse.add(new ArrayList<>());
    }
    for (int i = 0; i < len; i++) {
        for (int j : graph[i]) {
            reverse.get(j).add(i);
            degrees[i]++;
        }
    }
    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < len; i++) {
        if (degrees[i] == 0) {
            queue.offer(i);
        }
    }
    while (!queue.isEmpty()) {
        int cur = queue.poll();
        ret.add(cur);
        for (int i : reverse.get(cur)) {
            degrees[i]--;
            if (degrees[i] == 0) {
                queue.offer(i);
            }
        }
    }
    Collections.sort(ret);
    return ret;
}
```



另一个典型的图问题是**最短路问题**，解决最短路的方法包括普通队列BFS、Dijkstra(优先队列BFS)、Bellman-Ford算法(动态规划)，根据不同的限制条件可以使用不同的算法。

## [Leetcode 787. K站中转内最便宜的航班](https://leetcode-cn.com/problems/cheapest-flights-within-k-stops/submissions/)

### 问题

有`n`个城市通过一些航班连接。给你一个数组`flights`，其中`flights[i] = [fromi, toi, pricei]`，表示该航班都从城市`fromi`开始，以价格`pricei`抵达`toi`。

现在给定所有的城市和航班，以及出发城市`src`和目的地`dst`，你的任务是找到出一条最多经过`k`站中转的路线，使得从`src`到`dst`的 价格最便宜 ，并返回该价格。 如果不存在这样的路线，则输出 -1。

### 示例

示例1：

![img](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/02/16/995.png)

```
输入: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 1
输出: 200
```

示例2：

```
输入: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 0
输出: 500
```

题解：

第一种解法：*普通队列BFS*

在BFS解法中，一定要注意去重，去重的最好的办法就是维护一个`prices`数组，里面存放到达某个节点最小的价格，如果在入队列的过程中发现价格高于这个最小价格，那么就不应该入这个队列。注意：在入队列的时候就应该更新`prices`数组

```java
public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
    Queue<int[]> queue = new LinkedList<>();   // queue中的第0个元素为当前位置，第1个元素为当前经过的点的个数，第2个元素为当前的价格
    int[] prices = new int[n];  // prices[i]表示从src到i点的最小价格
    Arrays.fill(prices, Integer.MAX_VALUE);
    List<int[]>[] edges = new List[n];   // edges为邻接表, edges[i].get(0)为i指向的下一个节点，edges[i].get(1)为这个边的价格
    for (int i = 0; i < n; i++) {
        edges[i] = new ArrayList<>();
    }
    for (int[] flight : flights) {
        edges[flight[0]].add(new int[] {flight[1], flight[2]});
    }
    prices[src] = 0;
    queue.offer(new int[] {src, 0, 0});
    while (!queue.isEmpty()) {
        int[] cur = queue.poll();
        if (cur[1] > k) continue;
        for (int[] edge : edges[cur[0]]) {
            if (edge[1] + cur[2] > prices[edge[0]]) continue;
            prices[edge[0]] = edge[1] + cur[2];
            queue.offer(new int[] {edge[0], cur[1] + 1, edge[1] + cur[2]});
        }
    }
    return prices[dst] == Integer.MAX_VALUE ? -1 : prices[dst];
}
```

第二种解法：*Bellman-Ford算法(动态规划)*

设`dp[i][j]`为从出发城市`src`到城市`j`经过`i`个城市所需要的最小价格，那么遍历邻接表，可以得到如下状态转移方程，即对于`flights`邻接表中从位置`t`到位置`j`的航线，经过`i`个城市到达城市`j`的最小值为原本的最小值和经过`i - 1`个城市到达城市`t`再加上从`t`到`j`的价格两者比较的最小值。


$$
dp[i][j] = \min\limits_{(t, j) \in flights}(dp[i][j], dp[i - 1][t] + price[t][j])
$$


边界条件：`i`为0时，除了`dp[0][src]`应为0，其他`dp[0][i]`都应该为$$\inf$$​​​，当`dp`为$$\inf$$​​​​​时，表示当前无法从`src`到达城市`j`，上述状态转移方程也不成立。最后返回$$\min\limits_{i \in [1, k + 1]}(dp[i][dst])$$​​即可​

```java
public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
    int[][] dp = new int[k + 2][n];  // dp[i][j]表示从src出发经过i个点（i-1中转）到达点j所需要的最低价格
    for (int i = 0; i < k + 2; i++) {
        Arrays.fill(dp[i], Integer.MAX_VALUE);
    }
    dp[0][src] = 0;
    for (int i = 1; i <= k + 1; i++) {
        for (int[] flight : flights) {
            if (dp[i - 1][flight[0]] == Integer.MAX_VALUE) {
                continue;
            } else {
                dp[i][flight[1]] = Math.min(dp[i][flight[1]], dp[i - 1][flight[0]] + flight[2]);
            }
        }
    }
    int ret = Integer.MAX_VALUE;
    for (int i = 1; i <= k + 1; i++) {
        ret = Math.min(ret, dp[i][dst]);
    }
    return ret == Integer.MAX_VALUE ? -1 : ret;
}
```


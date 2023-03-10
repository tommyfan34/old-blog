---
title: Monotone Stack Problem
author: Xiao Fan
date: 2021-08-13 22:09 +0800
categories: [leetcode notes]
tags: [leetcode]
math: true
mermaid: true
---

单调栈，顾名思义就是栈中的元素满足单调递增或者单调递减的性质，单调栈的典型应用场景是**在一维数组中以$$O(N)$$​的时间寻找第一个满足某种条件的数**。

比如要在一列人中寻找排在自己前面第一个比自己矮的人，对于站在自己前面比自己高的人`higher_peoples`，这些人不是自己的目标，而对于下一个要寻找排在他们面前第一个比他们矮的人来说，`higher_peoples`同样不是他们的目标，因为前面那个人一定比`higher_peoples`要矮，我宁愿选择前面那个人也不可能选择`higher_peoples`，因此这些`higher_peoples`将对结果无法产生任何影响，可以被排除考虑。因此我们可以维护一个栈，每次将那些比当前的人高的前面的人排除出去，这样这个栈中后面来的人一定会比栈中的人高，这就形成了一个**单调递增栈**。

## [LeetCode 84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

### 问题

给定`n`个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为`1`。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

### 示例

示例 1:

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram.jpg)

```
输入：heights = [2,1,5,6,2,3]
输出：10
解释：最大的矩形为图中红色区域，面积为 10
```


示例 2：

![img](https://assets.leetcode.com/uploads/2021/01/04/histogram-1.jpg)

```
输入： heights = [2,4]
输出： 4
```

### 限制条件

1 <= heights.length <=10^5

0 <= heights[i] <= 10^4

### 题解

这道题是一道非常经典的单调栈问题。要得到最大的柱状图面积，可以枚举这个柱状图的高，也就是说假设这个最大的柱状图的高为`heights[i]`，我们要保证这个柱状图的高的左右两边的高都要至少大于等于这个`heights[i]`，换言之，将这个问题转化为：对于某一个柱状图的高`heights[i]`，分别寻找它左边和右边第一个小于`heights[i]`的索引`left`和`right`，则这个最大的柱状图面积为`heights[i] * (right - left - 1)`。

这道题的数据量在$$10^5$$级别，也就是最终的解法需要满足$$O(N)$$。由于枚举每个高已经需要$$O(N)$$的复杂度，因此需要保证另外在$$O(N)$$的时间复杂度内对每个`heights[i]`解出其对应的`left`和`right`并保存。这就不能够使用暴力解法，因此我们采用单调栈。和前面的“寻找在前面第一个比自己矮的人”的问题一样，我们维护一个**单调递增**栈，从左向右遍历可以找到`left`，从右向左遍历可以找到`right`。

```java
public int largestRectangleArea(int[] heights) {
    // 单调递增栈
    int ret = 0;
    int len = heights.length;
    // left和right数组分别表示左边和右边第一个小于heights[i]的index
    int[] left = new int[len];
    int[] right = new int[len];
    Stack<Integer> stack = new Stack<>();   // stack中保存的是index而不是heights[i]
    // 在heights左右两边分别插入一个-1，方便后续计算
    stack.push(-1);
    for (int i = 0; i < len; i++) {
        while (!stack.isEmpty() && heights[i] <= (stack.peek() == -1 ? -1 : heights[stack.peek()])) {
            stack.pop();
        }
        left[i] = stack.peek();
        stack.push(i);
    }
    stack = new Stack<>();
    stack.push(len);
    for (int i = len - 1; i >= 0; i--) {
        while (!stack.isEmpty() && heights[i] <= (stack.peek() == len ? -1 : heights[stack.peek()])) {
            stack.pop();
        }
        right[i] = stack.peek();
        stack.push(i);
    }
    for (int i = 0; i < len; i++) {
        ret = Math.max(ret, (right[i] - left[i] - 1) * heights[i]);
    }
    return ret;
}
```

## [LeetCode 42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

### 问题

给定`n`个非负整数表示每个宽度为1的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

### 示例

示例 1：

![img](/assets/img/posts/Monotone_Stack_Problem/rainwatertrap.png)

```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
```


示例 2：

```
输入：height = [4,2,0,3,2,5]
输出：9
```

### 限制条件

n == height.length

0 <= n <= 3 * 10^4

0 <= height[i] <= 10^5

### 题解

仔细观察问题，可以发现对于每一个`i`，如果它的左边或者右边没有遇到比它大的`height[j]`，那么这个高度上就不能蓄水，否则这个高度的水平面为`min(height[left[i]], height[right[i]])`，且对于`(left[i], right[i])`范围内的所有索引都要更新一遍水平面的高度，因为对于较低洼的地方，有可能第一次取到的水平面比较低（比如对于示例1中的第6个元素来说，左边第一个比他大的高度和右边第一个比他大的高度都是1，因此取到的水平面高度为1，但是实际上应该是2），因此每次更新水平面都要按照最大的更新。

```java
public int trap(int[] height) {
    int ret = 0;
    int len = height.length;
    // 寻找左边第一个比自己大的index和右边第一个比自己大的index
    Stack<Integer> stack = new Stack<>();
    int[] left = new int[len];
    int[] right = new int[len];
    stack.push(-1);
    for (int i = 0; i < len; i++) {
        while (!stack.isEmpty() && (height[i] >= (stack.peek() == -1 ? Integer.MAX_VALUE : height[stack.peek()]))) {
            stack.pop();
        }
        left[i] = stack.peek();
        stack.push(i);
    }
    stack = new Stack<>();
    stack.push(len);
    for (int i = len - 1; i >= 0; i--) {
        while (!stack.isEmpty() && (height[i] >= (stack.peek() == len ? Integer.MAX_VALUE : height[stack.peek()]))) {
            stack.pop();
        }
        right[i] = stack.peek();
        stack.push(i);
    }
    // max为水平面高度
    int[] max = new int[len];
    for (int i = 0; i < len; i++) {
        int temp = Math.min(left[i] == -1 ? 0 : height[left[i]], right[i] == len ? 0 : height[right[i]]);
        if (left[i] == -1 || right[i] == len) {
            max[i] = Math.max(max[i], height[i]);
            continue;
        }
        // ltft[i] + 1位置到right[i] - 1位置的水平面高度为左边的第一个比自己高的height和右边第一个比自己高的height的最小值，再和当前已经被设定的水平面高度比较，取较大的那一个
        for (int j = left[i] + 1; j < right[i]; j++) {
            max[j] = Math.max(max[j], temp);
        }
    }
    for (int i = 0; i < len; i++) {
        ret += max[i] - height[i];
    }
    return ret;
}
```

## [Leetcode 456. 132模式](https://leetcode-cn.com/problems/132-pattern/)

### 问题

给你一个整数数组`nums`，数组中共有`n`个整数。132 模式的子序列 由三个整数`nums[i]`、`nums[j]`和`nums[k] `组成，并同时满足：`i < j < k`和`nums[i] < nums[k] < nums[j]`。

如果`nums`中存在**132 模式的子序列**，返回`true`；否则，返回`false`。

### 示例

示例 1：

```
输入：nums = [1,2,3,4]
输出：false
解释：序列中不存在 132 模式的子序列。
```


示例 2：

```
输入：nums = [3,1,4,2]
输出：true
解释：序列中有 1 个 132 模式的子序列： [1, 4, 2] 。
```


示例 3：

```
输入：nums = [-1,3,2,0]
输出：true
解释：序列中有 3 个 132 模式的的子序列：[-1, 3, 2]、[-1, 3, 0] 和 [-1, 2, 0] 。
```

### 限制条件

n == nums.length
1 <= n <= 2 * 10^5
-10^9 <= nums[i] <= 10^9

### 题解

我们的思路是，先寻找3和2，即寻找两个数，满足后面的数小于前面的数，然后再寻找一个数小于这两个数。由于1是在最前面的，因此可以对数组从后往前倒序查找，先查找3和2。为了能够尽可能找到1，必须要让2越大越好，同时也要维护3和2之间的位置关系。每次从后往前遍历遇到一个很大的数的时候，先不要着急更新2，而是应该先满足存在3，然后将之前遇到过的比现在的2大的数更新为2，这里我们可以想到使用一个单调递减栈来维护3，每次遇到一个大于当前栈顶的元素时，将这个栈顶pop出来作为新的2，然后将这个元素入栈作为3。

```java
public boolean find132pattern(int[] nums) {
    int len = nums.length;
    if (len < 3) return false;
    Stack<Integer> stack = new Stack<>();   // 存储3
    int last = Integer.MIN_VALUE;  // 存储2
    for (int i = len - 1; i >= 0; i--) {
        if (nums[i] < last) {
            return true;
        }
        // 维护一个单调递减栈，这样栈中pop出来的都一定大于last
        while (!stack.isEmpty() && stack.peek() < nums[i]) {
            last = stack.pop();
        }
        stack.push(nums[i]);
    }
    return false;
}
```


---
title: LIS Problem
author: Xiao Fan
date: 2021-07-26 12:52 +0800
categories: [leetcode notes]
tags: [leetcode]
math: true
mermaid: true
---

LIS问题，即最长递增子序列问题（**L**ongest **I**ncreasing **S**ubsequence问题），是一道经典的字符串问题，即计算一个整数序列中最长的严格递增的子字符串的长度。本文通过几道Leetcode问题来介绍LIS问题及其变式，下面是LIS问题的原题。

## [Leetcode 300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

### 问题

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

### 示例

示例 1：

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```


示例 2：

```
输入：nums = [0,1,0,3,2,3]
输出：4
示例 3：
```

实例3：

```
输入：nums = [7,7,7,7,7,7,7]
输出：1
```

### 题解

这道题有2种解法，首先比较容易可以想到的是动态规划，设`dp[i]`为以`nums[i]`结尾的最长递增子序列的长度，则递推公式为
$$
dp[i] = max(dp[j](j < i\ \&\& \ nums[j]<nums[i])) + 1
$$


```java
public int lengthOfLIS(int[] nums) {
    int ret = 1;
    // dp[i]表示以第i个nums结尾的subsequence的最长长度
    int[] dp = new int[nums.length + 1];
    dp[1] = 1;

    for (int i = 2; i <= nums.length; i++) {
        dp[i] = 1;
        for (int j = 1; j < i; j++) {
            if (nums[i - 1] > nums[j - 1]) {
                dp[i] = Math.max(dp[i], 1 + dp[j]);
            }
        }
        ret = Math.max(ret, dp[i]);
    }
    return ret;
}
```

上述方法的时间复杂度为$$O(N^2)$$，但是还可以将时间复杂度进一步降低到$$O(NlogN)$$

首先需要遍历整个数组，因此N的复杂度是不能减少的。我们维护一个`tails`数组，`tails[i]`表示长度为i+1的子序列的最后一个数，至于为什么维护最后一个数，是因为是否能延长子序列只和最后一个数有关，为了尽可能延长子序列，最后一个数需要尽可能的小，因此每当遍历到一个数`nums[i]`时，我们需要找到`tails`数组中比这个数大的最小的那个数`tails[j]`，然后将`tails[j]`更新为`nums[i]`，如果找不到这样的数，说明之前遇到的所有的数都比这个数小，因此对之前所有可能形成的子序列，递增子序列一定是可以延长1个的，因此将这个数加在`tails`的末尾。寻找`tails`数组中比`nums[i]`大的最小数的过程可以用二分查找，复杂度为$O(logN)$

```java
public int lengthOfLIS(int[] nums) {
    int len = nums.length;
    // tails数组为所有上升子序列中最后一个数最小的情况下组成的数组，比如tails[2]就是所有长度为3的上升子序列最小的尾数
    // 让尾数最小可以保证后面加入新的数时更有可能延长这个上升子序列
    // tails不一定和最长上升子序列相同，但tails一定是单调递增的
    List<Integer> tails = new ArrayList<>();
    for (int i = 0; i < len; i++) {
        // 利用二分法查找大于等于当前nums的tails中的最小数，更新tails
        int cur = nums[i];
        if (tails.size() == 0 || tails.get(tails.size() - 1) < cur) {
            tails.add(cur);
        } else {
            int left = 0;
            int right = tails.size() - 1;
            while (left < right) {
                int mid = (left + right) / 2;
                if (tails.get(mid) >= cur) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            tails.set(left, cur);
        }
    }
    return tails.size();
}
```

## [Leetcode 1713. 得到子序列的最小操作次数](https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/)

### 问题

给你一个数组`target` ，包含若干**互不相同**的整数，以及另一个整数数组`arr `，`arr `可能包含重复元素。

每一次操作中，你可以在`arr`的任意位置插入任一整数。比方说，如果`arr` = [1,4,1,2] ，那么你可以在中间添加 3 得到 [1,4,3,1,2] 。你可以在数组最开始或最后面添加整数。

请你返回**最少**操作次数，使得`target`成为`arr`的一个子序列。

一个数组的**子序列**指的是删除原数组的某些元素（可能一个元素都不删除），同时不改变其余元素的相对顺序得到的数组。比方说，[2,7,4] 是 [4,2,3,7,2,1,4] 的子序列（加粗元素），但 [2,4,2] 不是子序列。

### 示例

示例 1：

```
输入：target = [5,1,3], arr = [9,4,2,3,4]
输出：2
解释：你可以添加 5 和 1 ，使得 arr 变为 [5,9,4,1,2,3,4] ，target 为 arr 的子序列。
```
示例2：

```
输入：target = [6,4,8,1,3,2], arr = [4,7,6,2,3,8,6,1]
输出：3
```

### 限制条件

1 <= target.length, arr.length <= 10^5

1 <= target[i], arr[i] <= 10^9

target 不包含任何重复元素。

### 题解

这道题实际上就是将`arr`中和`target`能够对应的数转化为在`target`中对应的index，然后求这个index的最长上升子序列。注意：由于数据量为$$10^5$$级别，因此一定要使用$$O(NlogN)$$复杂度的算法

```java
public int minOperations(int[] target, int[] arr) {
    HashMap<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < target.length; i++) {
        map.put(target[i], i);
    }
    List<Integer> indices = new ArrayList<>();
    for (int i = 0; i < arr.length; i++) {
        Integer index = map.get(arr[i]);
        if (index != null) {
            indices.add(index);
        }
    }
    // 求indices的最大严格上升子序列，使用tails数组+二分法求取
    List<Integer> tails = new ArrayList<>();
    for (int i = 0; i < indices.size(); i++) {
        int cur = indices.get(i);
        if (tails.size() == 0 || cur > tails.get(tails.size() - 1)) {
            tails.add(cur);
        } else {
            // 寻找大于等于cur的tails中的最小值
            int left = 0;
            int right = tails.size() - 1;
            while (left < right) {
                int mid = (left + right) / 2;
                if (tails.get(mid) >= cur) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            tails.set(left, cur);
        }
    }
    return target.length - tails.size();
}
```

## [Leetcode 354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)

### 问题

给你一个二维整数数组`envelopes`，其中`envelopes[i] = [wi, hi]`，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。

### 示例

示例 1：

```
输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
```

示例 2：

```
输入：envelopes = [[1,1],[1,1],[1,1]]
输出：1
```

### 限制条件

1 <= envelopes.length <= 5000

envelopes[i].length == 2

1 <= wi, hi <= 10^4

### 题解

这道题是一个二维LIS问题，首先可以按照宽度（即`envelopes[i][0]`）从小到大对`envelopes`进行排序，所有可能的套娃顺序只可能从宽度从小到大排列的这个数列中产生，然后再按照长度（即`envelopes[i][1]`）寻找LIS即可。

但是需要注意的是：当宽度相等的时候，比如[1,1], [1,2], [1,3], [1,4]这四个，按照道理应该是只能形成一个信封，但是由于寻找LIS的过程只看长度，因此会计算为可以形成4个信封，出现错误。这里有一个非常巧妙的办法，就是在宽度相等的时候，对长度按照降序进行排列，这样对于同样宽度的信封，只会将长度最小的信封加入`tails`中

```java
public int maxEnvelopes(int[][] envelopes) {
    // LIS问题
    int len = envelopes.length;
    List<int[]> list = new ArrayList<>();
    for (int i = 0; i < len; i++) {
        list.add(new int[] {envelopes[i][0], envelopes[i][1]});
    }
    Collections.sort(list, (o1, o2) -> (o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0]));  // 注意：这里将高度按照了逆序排列，这样可以防止长度相同时求取高度的最长递增子序列出现多计算的问题
    List<int[]> tails = new ArrayList<>();
    for (int i = 0; i < len; i++) {
        int[] cur = list.get(i);
        if (tails.size() == 0 || ((cur[1] > tails.get(tails.size() - 1)[1]) && (cur[0] != tails.get(tails.size() - 1)[0]))) {
            tails.add(cur);
        } else {
            int left = 0;
            int right = tails.size() - 1;
            while (left < right) {
                // 寻找大于等于cur的最小的tails
                int mid = (left + right) / 2;
                if (tails.get(mid)[1] >= cur[1]) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            if (tails.get(left)[1] == cur[1]) continue;
            tails.set(left, cur);
        }
    }
    return tails.size();
}
```

## [Leetcode 435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

### 问题

给定一个区间的集合`intervals`，找到需要移除区间的最小数量，使剩余区间互不重叠。

注意:

可以认为区间的终点总是大于它的起点。
区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

### 示例

示例 1:

```
输入: [ [1,2], [2,3], [3,4], [1,3] ]

输出: 1

解释: 移除 [1,3] 后，剩下的区间没有重叠。
```


示例 2:

```
输入: [ [1,2], [1,2], [1,2] ]

输出: 2

解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
```

示例 3:

```
输入: [ [1,2], [2,3] ]

输出: 0

解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
```

### 题解

这道题也是一个二维LIS问题。首先以第一个元素为主键对`intervals`进行升序排列，然后维护一个`tails`数组，这个数组中存放的所有`interval`都满足互不重叠。在遍历`intervals`的过程中，要让`cur[0]`和`tails`中的最后一个元素的最后一个值进行比较，如果大于等于这个值，此时不重叠，那么直接在`tails`的后面加上这个元素（贪心思想），否则将`cur[1]`和`tails(tails.size() - 1)[1]`进行比较，小的那个变成`tails`的最后一个元素，这是因为最后一个元素的最后一个值越小，越有可能和后面遍历元素不重叠。

为什么可以只管数组中的第二个值而不需要考虑第一个值呢？因为`intervals`第一个值已经升序排列，因此后面遍历的元素的第一个值不可能小于`tails`元素中的第一个值，这时只有第二个值的相对位置才会对结果产生影响

```java
public int eraseOverlapIntervals(int[][] intervals) {
    List<int[]> list = new ArrayList<>();
    int len = intervals.length;
    for (int i = 0; i < len; i++) {
        list.add(new int[] {intervals[i][0], intervals[i][1]});
    }
    Collections.sort(list, (o1, o2) -> (o1[0] == o2[0] ? o1[0] - o2[0] : o1[1] - o2[1]));
    List<int[]> tails = new ArrayList<>();  // tails保证所有元素都是无重叠区间
    for (int i = 0; i < len; i++) {
        int[] cur = list.get(i);
        // 寻找终点大于起点的拥有最小终点的tails
        if (tails.size() == 0 || tails.get(tails.size() - 1)[1] <= cur[0]) {
            tails.add(cur);
        } else {
            if (cur[1] < tails.get(tails.size() - 1)[1]) {
                tails.set(tails.size() - 1, cur);
            }
        }
    }
    return len - tails.size();
}
```

## [Leetcode 面试题08.13. 堆箱子](https://leetcode-cn.com/problems/pile-box-lcci/)

### 问题

堆箱子。给你一堆`n`个箱子，箱子宽`wi`、深`di`、高`hi`。箱子不能翻转，将箱子堆起来时，下面箱子的宽度、高度和深度必须大于上面的箱子。实现一种方法，搭出最高的一堆箱子。箱堆的高度为每个箱子高度的总和。

输入使用数组`[wi, di, hi]`表示每个箱子。

### 示例

示例1:

```
输入：box = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
输出：6
```


示例2:

```
输入：box = [[1, 1, 1], [2, 3, 4], [2, 6, 7], [3, 4, 5]]
输出：10
```

### 限制条件

箱子的数目不大于3000个。

### 题解

这道题限制了数据范围为3000，说明可以使用$$O(N^2)$$的算法，我们使用动态规划，先按照深度进行降序排列，保证后面的箱子满足叠在前面箱子上的条件。设dp[i]为以排序后第i个箱子为最上面一个箱子的最大高度，则递推公式满足
$$
dp[i] = max(dp[j] (box[i][0] < box[j][0] \ \&\& \ box[i][1] < box[j][1] \ \&\& \ box[i][2] < box[j][2]))
$$


最后返回dp[i]的最大值即可

```java
public int pileBox(int[][] box) {
    List<int[]> list = new ArrayList<>();
    int len = box.length;
    for (int i = 0; i < len; i++) {
        list.add(box[i]);
    }
    Collections.sort(list, (o1, o2) -> (o2[2] - o1[2]));
    int[] dp = new int[len];
    int ret = 0;
    for (int i = 0; i < len; i++) {
        int[] cur = list.get(i);
        int res = 0;
        for (int j = 0; j < i; j++) {
            int[] com = list.get(j);
            if (cur[0] < com[0] && cur[1] < com[1] && cur[2] < com[2]) {
                res = Math.max(res, dp[j]);
            }
        }
        dp[i] = res + cur[2];
        ret = Math.max(ret, dp[i]);
    }
    return ret;
}
```

## [Leetcode 376. 摆动序列](https://leetcode-cn.com/problems/wiggle-subsequence/)

### 问题

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为**摆动序列** 。第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

例如， [1, 7, 4, 9, 2, 5] 是一个 摆动序列 ，因为差值 (6, -3, 5, -7, 3) 是正负交替出现的。

相反，[1, 4, 7, 2, 5] 和 [1, 7, 4, 5, 5] 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。
子序列 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

给你一个整数数组`nums`，返回`nums`中作为 摆动序列 的 最长子序列的长度 。

### 示例

示例 1：

```
输入：nums = [1,7,4,9,2,5]
输出：6
解释：整个序列均为摆动序列，各元素之间的差值为 (6, -3, 5, -7, 3) 。
```


示例 2：

```
输入：nums = [1,17,5,10,13,15,10,5,16,8]
输出：7
解释：这个序列包含几个长度为 7 摆动序列。
其中一个是 [1, 17, 10, 13, 10, 16, 8] ，各元素之间的差值为 (16, -7, 3, -3, 6, -8) 。
```


示例 3：

```
输入：nums = [1,2,3,4,5,6,7,8,9]
输出：2
```

### 题解

这道题严格意义上来说并不是LIS问题，在LIS问题中，如果采用动态规划的解法，由于`dp`并不一定是单调递增的数列，因此在每次迭代的过程中还需要对所有之前的`dp[j]`进行迭代来寻找最大的符合条件的`dp[j]`。但是在摆动序列中，假设我们维护2个`dp`数组，`dp[i][0]`表示以`nums[i]`为结尾的最长摆动序列（且此时结尾为波谷）的长度，`dp[i][1]`表示以`nums[i]`为结尾的最长摆动序列（且此时结尾为波峰） 的长度，这2个数组一定是单调递增的，因为不管后面的`nums`和前面的`nums`的大小关系如何，波峰序列和波谷序列中总有一个可以被延长，如果前后大小相等，则波峰序列和波谷序列不变。因此这时的时间复杂度变为$$O(N)$$

```java
public int wiggleMaxLength(int[] nums) {
    int len = nums.length;
    int[][] dp = new int[len][2];  // dp[i][0]表示以nums[i]结尾且序列中的下一个数需要大于这一个数的序列最长长度，dp[i][1]表示以nums[i]结尾且序列中的下一个数需要小于这一个数的序列最长长度
    dp[0][0] = 1;
    dp[0][1] = 1;
    for (int i = 1; i < len; i++) {
        if (nums[i] < nums[i - 1]) {
            dp[i][0] = Math.max(dp[i - 1][1] + 1, dp[i - 1][0]);
            dp[i][1] = dp[i - 1][1];
        } else if (nums[i] > nums[i - 1]) {
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = Math.max(dp[i - 1][0] + 1, dp[i - 1][1]);
        } else {
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = dp[i - 1][1];
        }
    }
    return Math.max(dp[len - 1][0], dp[len - 1][1]);
}
```

## [Leetcode 673. 最长递增子序列的个数](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/) 

### 问题

给定一个未排序的整数数组，找到最长递增子序列的个数。

### 示例

示例 1:

```
输入: [1,3,5,4,7]
输出: 2
解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。
```


示例 2:

```
输入: [2,2,2,2,2]
输出: 5
解释: 最长递增子序列的长度是1，并且存在5个子序列的长度为1，因此输出5。
```

### 题解

这道题可以用暴力动态规划来解，一个数组存储前i个数最长LIS的长度，另一个数组存储前i个数最长LIS的方案个数。

```java
public int findNumberOfLIS(int[] nums) {
    int len = nums.length;
    int[] dp1 = new int[len + 1];  // dp1[i]表示以第i个数结尾的最长递增子序列的个数
    int[] dp2 = new int[len + 1];  // dp2[i]表示以第i个数结尾的最长递增子序列的长度
    int ret = 0;
    int maxLen = 0;
    for (int i = 1; i <= len; i++) {
        int LISlen = 1;
        int LISnum = 1;
        for (int j = 1; j < i; j++) {
            if (nums[i - 1] > nums[j - 1]) {
                if (LISlen < dp2[j] + 1) {
                    LISnum = dp1[j];
                    LISlen = dp2[j] + 1;
                } else if (LISlen == dp2[j] + 1) {
                    LISnum += dp1[j];
                }
            } else if (nums[i - 1] == nums[j - 1]) {

            }
        }
        dp1[i] = LISnum;
        dp2[i] = LISlen;
        if (LISlen > maxLen) {
            ret = dp1[i];
            maxLen = LISlen;
        } else if (LISlen == maxLen) {
            ret += dp1[i];
        }
    }
    return ret;
}
```

方案的时间复杂度为
$$
O(N^2)
$$
为了进一步降低时间复杂度，我们可以设一个dp二维数组，具体方案可以看题解https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/solution/zui-chang-di-zeng-zi-xu-lie-de-ge-shu-by-w12f/，dp[i]表示长度为i的LIS的最后一个数，可以知道dp[i]单调非增，cnt\[i]\[j]表示长度为i的LIS前j个方案的前缀和的个数。

```java
public int findNumberOfLIS(int[] nums) {
    int len = nums.length;
    List<List<Integer>> dp = new ArrayList<>();  // dp[i]表示LIS长度为i的最后一个数，dp[i]单调非增
    List<List<Integer>> cnt = new ArrayList<>();  // cnt[i][j]表示长度为i的LIS种类的前缀和
    for (int i = 0; i < len; i++) {
        // 寻找放在dp的哪一个中,寻找第一个大于等于nums[i]的dp[i][-1]
        int left = 0;
        int right = dp.size() - 1;
        if (dp.size() == 0) {
            dp.add(new ArrayList<>());
            dp.get(0).add(nums[i]);
        } else if (dp.get(dp.size() - 1).get(dp.get(dp.size() - 1).size() - 1) < nums[i]) {
            dp.add(new ArrayList<>());
            dp.get(dp.size() - 1).add(nums[i]);
            left = dp.size() - 1;
        } else {
            while (left < right) {
                int mid = (left + right) / 2;
                if (dp.get(mid).get(dp.get(mid).size() - 1) >= nums[i]) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            dp.get(left).add(nums[i]);
        }
        // 计算cnt
        if (left == 0) {
            if (cnt.size() == 0) cnt.add(new ArrayList<>());
            cnt.get(0).add(cnt.get(0).size() + 1);   // 前缀和
        } else {
            // 寻找left - 1中第一个严格小于nums[i]的
            int l = 0;
            int r = cnt.get(left - 1).size() - 1;
            while (l < r) {
                int m = (l + r) / 2;
                if (dp.get(left - 1).get(m) < nums[i]) {
                    r = m;
                } else {
                    l = m + 1;
                }
            }
            if (cnt.size() < left + 1) {
                cnt.add(new ArrayList<>());
            }
            int cnts = cnt.get(left - 1).get(cnt.get(left - 1).size() - 1) - (l == 0 ? 0 : cnt.get(left - 1).get(l - 1));
            if (cnt.get(left).size() != 0) {
                cnts += cnt.get(left).get(cnt.get(left).size() - 1);
            }
            cnt.get(left).add(cnts);
        }
    }
    return cnt.get(cnt.size() - 1).get(cnt.get(cnt.size() - 1).size() - 1);
}
```


---

title: LCS & LPS Problem
author: Xiao Fan
date: 2021-07-27 11:12 +0800
categories: [leetcode notes]
tags: [leetcode]
math: true
mermaid: true
---

上文提到了LIS问题及其变式，本文讨论LCS问题和LPS问题。

LCS问题即最长公共子序列问题（**L**ongest **C**ommon **S**ubsequence），求两个字符串对最长公共子序列。LPS问题即最长回文子序列问题（**L**ongest **P**alindrome **S**ubsequence），求字符串中可以形成回文串的最长子序列。由于最长回文子序列可以用LCS来解决，因此放在一起进行讨论。

## [Leetcode 1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

### 问题

给定两个字符串`text1`和`text2`，返回这两个字符串的最长公共子序列的长度。如果不存在公共子序列 ，返回 0 。

一个字符串的**子序列**是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的**公共子序列**是这两个字符串所共同拥有的子序列。

### 示例

示例 1：

```
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
```


示例 2：

```
输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc" ，它的长度为 3 。
```


示例 3：

```
输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0 。
```

### 限制条件

1 <= text1.length, text2.length <= 1000

text1 和 text2 仅由小写英文字符组成。

### 题解

首先思考，这道题可以用贪心做么？如果碰到相同的两个字符，是否可以立刻加入这个公共子序列中？答案是否定的，比如我们考虑以下2个字符串："adbc"和"abcd"，很明显这两个字符串的最大公共子序列为"abc"，但是如果我们采用贪心策略，在第一个字符串遇到d时寻找第二个字符串的d将其加入公共子序列，将会错过第二个字符串中的"bc"，因此贪心策略是错误的。

我们考虑使用二维动态规划，`dp[i][j]`表示`text1`的前i个字符和`text2`的前j个字符形成的子字符串的最长公共子序列的长度，当两个字符串遍历到相同的字符时，一定可以将`dp[i - 1][j - 1]`加一，当两个字符串遍历到不同的字符时，此时的`dp[i][j]`应该是`dp[i - 1][j]`和`dp[i][j - 1]`之间的最大值，因为你不知道是增加`text1`的第i个字符会对`dp[i - 1][j - 1]`进行延长还是增加`text2`的第j个字符会延长。因此状态转移方程为


$$
dp[i][j] = \begin{cases}
				dp[i - 1][j - 1] + 1, &  text1[i - 1] \ == \ text2[j - 1] \\
				max(dp[i - 1][j], dp[i][j - 1]), &  text1[i - 1] \ != \  text2[j - 1]
		   \end{cases}
$$


```java
public int longestCommonSubsequence(String text1, String text2) {
    int m = text1.length();
    int n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}
```

算法的时间复杂度为$$O(N^2)$$

## [Leetcode 718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

### 问题

给两个整数数组`A`和`B` ，返回两个数组中公共的、长度最长的子数组的长度。

### 示例

```
输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：
长度最长的公共子数组是 [3, 2, 1] 。
```

### 限制条件

1 <= len(A), len(B) <= 1000

0 <= A[i], B[i] < 100

### 题解

这道题和上一道的区别在于子数组是连续的，而子序列只需要保证顺序一致即可。这道题也可以用动态规划来做，`dp[i][j]`表示以`A`第i个数和`B`第j个数结尾最长可以形成的子数组长度，当`A[i]`不等于`B[j]`的情况下，`dp[i][j]`为0，否则`dp[i][j] = dp[i - 1][j - 1] + 1`，状态转移方程为


$$
dp[i][j] = \begin{cases}
			dp[i - 1][j - 1] + 1, & A[i - 1] \ == \ B[j - 1] \\
			0, & A[i - 1] \ != \ B[j - 1]
\end{cases}
$$


返回`dp[i][j]`最大值即可

```java
public int findLength(int[] nums1, int[] nums2) {
    int len1 = nums1.length;
    int len2 = nums2.length;
    int[][] dp = new int[len1 + 1][len2 + 1];
    int ret = 0;
    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            if (nums1[i - 1] == nums2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = 0;
            }
            ret = Math.max(ret, dp[i][j]);
        }
    }
    return ret;
}
```

算法的时间复杂度为$$O(N^2)$$

## [Leetcode 5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

### 问题

给你一个字符串`s`，找到`s`中最长的回文子串。

### 示例

示例 1：

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```


示例 2：

```
输入：s = "cbbd"
输出："bb"
```


示例 3：

```
输入：s = "a"
输出："a"
```


示例 4：

```
输入：s = "ac"
输出："a"
```

### 限制条件

1 <= s.length <= 1000

s 仅由数字和英文字母（大写和/或小写）组成

### 题解

这道题有很多解法，这里我们可以用LCS的方法来解决这个LPS问题。一个字符串中的最长回文子串一定是这个字符串倒过来之后和原字符串的一个公共子字符串，但是并不一定是最长的那个子字符串，比如我们考虑"aacadeacaa"这个字符串，如果倒过来就变成"aacaedacaa"，最长公共子字符串为"aaca"，但是这并不是一个回文串，因此我们要寻找的这个公共子字符串必须满足一个要求，即原字符串`s`和颠倒的字符串`ss`中匹配到的这个公共子字符串必须是同一个子字符串，也就是说，`s`字符串中0-3的这个"aaca"和位置6-9的这个"acaa"进行颠倒后尽管能够匹配，但是这是无效的。为了进行判断，我们需要每次在找到最长的公共子字符串时判断i和j的位置，如果匹配的这个公共子字符串是由原先的那个回文串颠倒之后匹配而来的，那么应当满足`i - dp[i][j] + 1 + j - 1 == len`，这个公式由读者自行推导。

```java
public String longestPalindrome(String s) {
    String ss = (new StringBuilder(s)).reverse().toString();
    // 寻找s和ss的最长公共子串
    int len = s.length();
    int[][] dp = new int[len + 1][len + 1];
    int ret = 0;
    int start = 0;
    for (int i = 1; i <= len; i++) {
        for (int j = 1; j <= len; j++) {
            char c1 = s.charAt(i - 1);
            char c2 = ss.charAt(j - 1);
            if (c1 == c2) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = 0;
            }
            if (dp[i][j] > ret) {
                // 要保证是回文串
                int temp = j - dp[i][j] + 1;
                if (i + temp - 1 == len) {
                    start = i - dp[i][j];
                    ret = dp[i][j];
                }
            }

        }
    }
    return s.substring(start, start + ret);
}
```

## [Leetcode 516. 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/submissions/)

### 问题

给你一个字符串`s`，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

### 示例

示例 1：

```
输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb" 。
```


示例 2：

```
输入：s = "cbbd"
输出：2
解释：一个可能的最长回文子序列为 "bb" 。
```

### 限制条件

1 <= s.length <= 1000

s 仅由小写英文字母组成

### 题解

这道题和上面类似，将`s`反转后得到`ss`寻找这两个字符串的最长公共子序列，但是需要注意的是，和上一道题不同，本题不需要考虑匹配的公共子序列来自于同一个子序列。继续以"aacadeacaa"这个字符串为例，后面的"acaa"颠倒之后可以和前面的"aaca"匹配，尽管和"aaca"不是一个子字符串，但是由于子序列并不要求连续，因此"acaa"颠倒之后仍然可以成为回文子序列。

```java
public int longestPalindromeSubseq(String s) {
    // 字符串反转求LCS
    String ss = (new StringBuilder(s)).reverse().toString();
    int len = s.length();
    int[][] dp = new int[len + 1][len + 1];
    for (int i = 1; i <= len; i++) {
        for (int j = 1; j <= len; j++) {
            if (s.charAt(i - 1) == ss.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[len][len];
}
```

## [Leetcode 1771. 由子序列构造的最长回文串的长度](https://leetcode-cn.com/problems/maximize-palindrome-length-from-subsequences/)

### 问题

给你两个字符串`word1`和`word2`，请你按下述方法构造一个字符串：

从`word1`中选出某个**非空**子序列`subsequence1`。
从`word2`中选出某个**非空**子序列`subsequence2`。
连接两个子序列`subsequence1 + subsequence2`，得到字符串。
返回可按上述方法构造的最长*回文串*的 长度 。如果无法构造回文串，返回 0 。

字符串 s 的一个*子序列*是通过从`s`中删除一些（也可能不删除）字符而不更改其余字符的顺序生成的字符串。

*回文串*是正着读和反着读结果一致的字符串。

### 示例

示例 1：

```
输入：word1 = "cacb", word2 = "cbba"
输出：5
解释：从 word1 中选出 "ab" ，从 word2 中选出 "cba" ，得到回文串 "abcba" 。
```


示例 2：

```
输入：word1 = "ab", word2 = "ab"
输出：3
解释：从 word1 中选出 "ab" ，从 word2 中选出 "a" ，得到回文串 "aba" 。
```


示例 3：

```
输入：word1 = "aa", word2 = "bb"
输出：0
解释：无法按题面所述方法构造回文串，所以返回 0 。
```

### 限制条件

1 <= word1.length, word2.length <= 1000

word1 和 word2 由小写英文字母组成

### 题解

这道题可以通过将`word1`和`word2`拼接起来之后得到`s`，然后求取`s`的最长回文子序列，但是要注意的是，求取的回文子序列的长度要保证这个回文串的最左侧在`word1`内，而最右侧在`word2`内。采用动态规划，设`dp[i][j]`为`s`的位置为[i, j]的子字符串的最长回文子序列的长度，那么当`s.charAt(i) == s.charAt(j)`，即两头的字符相同时，可以延长这个最长回文子序列的长度2，即`dp[i][j] = dp[i + 1][j - 1] + 2`，如果不相同，则应该是将两头字符分别去掉一个之后获得的最长回文子序列长度的最大值，递归公式如下所示


$$
dp[i][j] = \begin{cases}
dp[i + 1][j - 1] + 2, & s[i] == s[j] \\
max(dp[i][j - 1], dp[i + 1][j]), & s[i] != s[j] \\
1, & i == j
\end{cases}
$$


```java
public int longestPalindrome(String word1, String word2) {
    int ret = 0;
    String s = word1 + word2;
    int len = s.length();
    int[][] dp = new int[len][len];  // dp[i][j]表示在[i, j]范围内的子串最长回文子序列的长度
    for (int i = len - 1; i >= 0; i--) {
        dp[i][i] = 1;   // base case
        for (int j = i + 1; j < len; j++) {
            if (s.charAt(i) == s.charAt(j)) {
                dp[i][j] = dp[i + 1][j - 1] + 2;
                // 只有当i和j上的字符相同的时候，才会更新最长的长度。其中i要在word1中，j要在word2中
                if (i < word1.length() && j >= word1.length()) {
                    ret = Math.max(ret, dp[i][j]);
                }
            } else {
                dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }
    return ret;
}
```


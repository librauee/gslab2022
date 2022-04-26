# 运行环境

* Windows
* Python3.7
* 其他库参照requirements.txt

# 运行方法

直接运行run.py即可生成提交文件sub.csv

# 特征工程

* 提取PE静态特征，包括字节直方图、字节熵直方图和字符串信息等特征【1】；
* 对可读性字符串作tfidf特征；
* 提取PE头文件信息特征；
* 将PE反汇编，提取操作码序列，做tfidf 以及w2v特征；
* 提取ember特征【2】


【1】获奖方案 | 恶意软件赛题@Petrichor战队解题思路 https://mp.weixin.qq.com/s/NLHkw_wL64xyQGgr0xiriA
【2】Anderson HS, Roth P. Ember: an open dataset for training static pe malware machinelearning models[J]. arXiv preprint arXiv:1804.04637, 2018.


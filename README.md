# Dot.Animacy
Generate and classify animated dot motions

## TODO List

**Initial Plan / Idea**

#### 1. 关于motion

考虑：追逐，角力，阻拦，合作，etc.

#### 2. 关于数据

- 下载2d动画、游戏数据
  - 如果可行，试一下3d的动画、视频数据看看能不能用
  - 荒野乱斗 / 猫和老鼠（游戏） / speedrunner(game) / stickman video
  - 猫和老鼠（动画） / youtube动物捕猎 / 足球比赛 / 拳击比赛的拳套
  - JAAD （行人数据） / UCF101 / Kinetics (可能都不合适) / condensed moives
  - pygame写简单脚本制作
- 看一下SAM或者YOLO的识别效果，需要box和中心位置，用以改变dot大小和位置
  - [不重要] 涉及到只有一个人的动作，也可以考虑类似于四个点的运动可以让人感受到“只因你太美”的视频那种？用舞蹈数据集
    - This might be totally another different question. 多少个点可以让人类看懂一个视频。
  - [重要] 要是不微调效果不好可能需要标一点数据？这太扯淡了吧。还没去了解一般需要多少
- not animated: 给定一些点，随机运动？

#### 3. 关于判别和生成

- 不考虑GAN，太难训练了
- Transformer会不会很容易就过拟合了...
- 考虑到背景是白色的，图片组成只有点，或许可以转换成轨迹？然后用LSTM？或者diffusion？数据会不会有点少？
- CFM（条件流匹配）似乎有一定可行性？而且还能用于判别？
  - 数据数据数据....可能不够，可能分布不同？
  - 以前没有实现过会不会遇到什么问题。

#### 4. 关于dot animacy问题定义

- 只要人能说出这个视频有什么意义或者视频里面的点具体在干什么事情就算是animation？


## Timeline

- [11/30 or 12/1] - related works 调研， 主要是
  - animacy, 尤其是 dot animacy 过往的方法 （11/28）
  - 图像识别 segmentation 相关的研究  (11/28 or 11/29)
  - 时间序列相关的生成方法  (11/30 or 12/1)
  - 衡量animacy的指标  (11/30)
- [12/7 or 12/9] - 数据收集
  - 跑通sam或yolo  (12/8)
  - 随机运动的生成  (12/4)
- [12/14] - 模型搭建
  - 初步验证  [12/13]
  - 设计实验  [12/14]
- [12/16 3点前] - 初稿
  - abstract
  - introduction
  - related works
  - methodology
  - experiments
  - ablation （？）
  - conclusion
  - appendix
- [待定] - 终版
  - 跑完实验，看看有啥问题
  - 待定
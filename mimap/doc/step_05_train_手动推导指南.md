# `mipmap/step_05_train.py` 手动推导与执行指南

> 目标：把这段训练代码“拆开来看”，理解 **训练架构**、**关键数据流**、**每一步在做什么**、以及背后的 **基础数学原理**。  
> 读完后你应该能自己在纸上推一遍，也能在代码里对照每一步。

---

## 1. 先建立全局直觉：这段训练到底在优化什么？

一句话版本：

- 我们有一套 **高分辨率材质贴图**（albedo / normal / roughness）。
- 我们把它们下采样成四分之一分辨率，作为“真值低分辨率目标”。
- 再创建一套“待训练”的低分辨率贴图（初值是常量/默认法线）。
- 训练时，让“待训练贴图”在不同光照和视角下渲染出来的结果，尽量逼近高分辨率材质经过 4x4 平均后的参考结果。
- 通过自动微分拿到梯度，用 Adam 更新这三张待训练贴图。

你可以把它理解成：

- **输入空间**：低分辨率材质参数（每个像素有 albedo/normal/roughness）。
- **监督信号**：高分辨率渲染结果做局部平均后的颜色。
- **损失函数**：逐像素 RGB 平方误差。

---

## 2. 训练架构（模块职责图）

这一节对应 `step_05_train.py + step_05_train.slang` 的分工。

### Python 端（编排层）

`mipmap/step_05_train.py` 主要负责：

1. 加载贴图与初始化张量。
2. 调用 Slang 内核做渲染、loss、梯度与优化。
3. 控制训练循环、学习率调度、可视化与打印 loss。

### Slang 端（数值核）

`mipmap/step_05_train.slang` 主要负责：

1. `render`：BRDF 渲染（可微）。
2. `loss`：`(render - reference)^2`（可微）。
3. `calculate_grads`：随机采样光照/视角，构造高分辨率参考平均值，并通过 `bwd_diff(loss)` 回传梯度。
4. `optimizer_step1/3`：Adam 更新与约束（`saturate`、`normalize`）。
5. `downsample1/3`：2x2 均值下采样。

---

## 3. 从数据准备开始，手动走一遍

## 第 0 步：加载高分辨率三张图

- `albedo_map`：漫反射颜色（读取时 `linearize=True`）。
- `normal_map`：法线（`scale=2, offset=-1`，把图像值映射到法线常见区间）。
- `roughness_map`：粗糙度（灰度图）。

### 第 1 步：下采样两次（每次尺寸减半）

`downsample(source, 2)` 执行两轮 2x2 平均：

- 第一次：`H x W -> H/2 x W/2`
- 第二次：`H/2 x W/2 -> H/4 x W/4`

得到：

- `lr_albedo_map`
- `lr_normal_map`
- `lr_roughness_map`

这三者是“低分辨率真值（由高分辨率数据导出）”。

### 第 2 步：构造待训练参数（可学习变量）

创建：

- `lr_trained_albedo_map` 初值 `(0.5, 0.5, 0.5)`
- `lr_trained_normal_map` 初值 `(0, 0, 1)`
- `lr_trained_roughness_map` 初值 `0.5`

理解上相当于：

- 我们在拟合“一个低分辨率材质表”，让它在渲染上尽可能复现高分辨率统计行为。

### 第 3 步：准备梯度和 Adam 状态

对三类参数都准备：

- 梯度缓存：`*_grad`
- 一阶矩：`m_*`
- 二阶矩：`v_*`

这正是 Adam 所需状态。

---

## 4. 训练循环里每一帧发生了什么？

`while app.process_events():` 可以看成“每帧训练 + 可视化一次”。

### A. 先做两个前向渲染（用于观察）

1. **高分辨率真材质渲染** `output`，再下采样到低分辨率。
2. **当前待训练低分辨率材质渲染** `lr_output`。

这两张图并排显示，方便你直观看训练是否收敛。

### B. 计算 loss 可视化

- `orig_loss_output`：用“下采样得到的低分辨率真值材质”去对比 `output` 的损失。
  - 这是一个参考下界（不是严格数学下界，但可作“合理基准”）。
- `loss_output`：用“当前待训练材质”对比 `output` 的损失。

然后显示 `loss_output`，并把均值打印出来。

### C. 学习率调度（线性退火）

- 训练前期：接近 `0.002`
- 训练后期：逐渐降到 `0.0002`
- 使用 `optimize_counter / 3000` 做进度插值

直觉：前期快走，后期小步微调。

### D. 每帧做 50 次参数更新

核心调用是：

1. `calculate_grads(...)`
2. `optimizer_step3(...)` 更新 albedo
3. `optimizer_step3(...)` 更新 normal（会 renormalize）
4. `optimizer_step1(...)` 更新 roughness

---

## 5. 最关键的数据流（建议你按这个顺序在纸上画图）

可以把一次 `calculate_grads` 看成下列链路：

1. 选定一个低分辨率像素 `pixel`。
2. 映射到高分辨率左上角 `hi_res_pixel = pixel * 4`。
3. 取这个 4x4 block 的 16 个高分辨率像素，各自调用 `render`（使用 `ref_material`）。
4. 16 个结果求平均，得到参考颜色 `sum`。
5. 在同样的 `light_dir/view_dir` 下，用当前低分辨率待训练材质在 `pixel` 渲染并与 `sum` 做 `loss`。
6. 对 `loss` 执行反向传播，梯度写入 `albedo_grad / normal_grad / roughness_grad`。
7. Adam 读取这些梯度，更新待训练贴图。

重点是第 3~4 步：

- 代码没有直接比较“低分辨率渲染 vs 下采样后的高分辨率整图”，
- 而是按像素块做局部统计监督（4x4 期望），这很像 mipmap 思路中的“局部平均等效”。

---

## 6. 基础数学原理（最小必要版）

## 6.1 渲染与损失

对某像素记：

- 预测颜色：`c_hat = render(pixel, trained_material, l, v)`
- 参考颜色：`c_ref`（由高分辨率 4x4 平均得到）

损失（逐通道平方误差）：

\[
\mathcal{L} = (c_{hat} - c_{ref})^2
\]

在实现里是 `float3` 逐分量平方。

## 6.2 自动微分

`bwd_diff(loss)(..., 1)` 的意思可以理解为：

- 对 `loss` 的输出施加单位上游梯度（种子为 1），
- 自动沿计算图回传，
- 最终触发 `get_albedo_bwd/get_normal_bwd/get_roughness_bwd`，把梯度写入对应张量。

## 6.3 Adam 更新

代码中的 Adam 核心是：

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
\]
\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]
\[
\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
\]
\[
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
\]

然后加约束：

- albedo/roughness：`saturate` 到 `[0,1]`
- normal：`normalize` 到单位向量

---

## 7. 如何“手动执行一遍”训练（实操建议）

下面是一种非常适合学习的手工演练流程。

### 练习 1：先固定一个像素，只看一次迭代

你可以暂时把 `pixel=spy.grid(...)` 改成单个像素（例如中心像素），并把 for 循环迭代改成 1。

你要观察：

1. 该像素当前的 `trained` 参数值。
2. `calculate_grads` 后对应梯度值。
3. 一次 Adam 后参数变化量。
4. loss 是变大还是变小。

这样你会非常清楚“梯度如何驱动参数移动”。

### 练习 2：只训练一种参数

例如：

- 先冻结 normal/roughness，只更新 albedo。
- 再切换为只更新 roughness。
- 最后只更新 normal。

你会看到：

- albedo 对整体亮度/颜色贡献很直接，收敛通常更快。
- normal 与 roughness 的影响更“几何/高光相关”，更依赖光照采样。

### 练习 3：控制随机性

把 `seed` 固定，或者让光照/视角固定，再观察收敛曲线。

你会理解：

- 随机采样提高泛化（不同光照视角一致）。
- 但会引入噪声，学习率与迭代次数要平衡。

---

## 8. 常见疑问（快速答）

## Q1. 为什么参考值要用高分辨率 4x4 平均？
因为一个低分辨率像素本质覆盖了更大面积，4x4 局部平均是在构造“面积意义上的参考响应”。

## Q2. 为什么 normal 更新后要 `normalize`？
法线需要保持方向向量语义；不归一会导致 BRDF 输入失真。

## Q3. 为什么 roughness/albedo 要限制在 `[0,1]`？
这是物理参数的常见有效区间，避免优化发散到无意义值。

## Q4. 每帧 50 次更新有什么作用？
图形界面一帧时间里多做若干优化步，可以更快看到收敛，同时保留实时可视化。

---

## 9. 你可以对照源码重点看的函数

- Python 入口：`downsample`, 主循环中的 `module.render / module.loss / module.calculate_grads / module.optimizer_step*`。
- Slang 核心：`render`, `loss`, `calculate_grads`, `adam_step`, `optimizer_step1/3`, `downsample1/3`。

建议阅读顺序：

1. 先看 `loss -> render`（定义目标）。
2. 再看 `calculate_grads`（定义监督与反传路径）。
3. 最后看 `optimizer_step*`（参数怎样真正改变）。

---

## 10. 一页纸总结（可记笔记）

- **训练对象**：低分辨率三张材质图（albedo/normal/roughness）。
- **监督信号**：高分辨率材质在随机光照视角下的 4x4 渲染均值。
- **损失**：逐像素 RGB MSE。
- **梯度来源**：Slang 自动微分 `bwd_diff(loss)`。
- **优化器**：Adam + 参数约束（saturate/normalize）。
- **训练策略**：每帧多步更新 + 学习率线性衰减 + 可视化监控。

如果你愿意，我下一步可以再给你一个“教学版 step_05_train_debug.py”：

- 自动打印某个像素的参数、梯度、Adam 的 `m/v`、step 大小；
- 每 N 步保存中间材质图，帮助你做可解释性分析。

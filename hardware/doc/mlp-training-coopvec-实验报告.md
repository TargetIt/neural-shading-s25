# mlp-training-coopvec 示例实验报告

## 实验目的
1. 理解如何在 Slang MLP 训练中引入 **cooperative vector intrinsics** 进行加速。
2. 分析 coopvec 版本相对基础版本在数据布局、梯度存储和训练流程上的关键差异。
3. 为后续性能评估建立可复现实验步骤与结果记录框架。

## 实验原理
1. **网络结构保持一致**：`4 -> 16 -> 4` 两层全连接网络。
2. **输入编码与监督目标一致**：输入仍为 `[x, y, x^2, y^2]`，拟合同一组多项式函数。
3. **核心加速点**：使用 cooperative vector 能力进行训练中的矩阵相关运算（尤其是梯度累积路径）。
4. **额外存储布局**：参数缓冲由两段扩展为三段：
   - 段 1：权重/偏置（row-major）
   - 段 2：梯度（row-major）
   - 段 3：权重梯度（training-optimal）
5. **布局转换机制**：先在 training-optimal 布局累计权重梯度，再转换回 row-major，供 Adam 更新核读取。

## 实验手段
1. **代码静态分析**：阅读 `mlp-training-coopvec.cpp`、`kernels.slang`、`network.slang`。
2. **差异对照分析**：重点比对与 `mlp-training` 的流程差异（特别是梯度缓冲段与矩阵布局转换）。
3. **结果判定方式**：以每 10 次迭代输出的 loss 曲线观察收敛表现，并为性能测试预留对照口径。

## 实验条件
1. 平台：Windows / Linux（需 NVIDIA GPU）；macOS 不支持该示例运行。
2. 图形后端：Vulkan（示例使用 cooperative vector 相关 SPIR-V 能力）。
3. 数据类型：参数与梯度主要以 `half/float16` 存储（`NFloat = uint16_t`）。
4. 训练配置（由示例代码固定）：
   - 迭代次数：1000
   - 输入样本缓冲：32 个 float（按 `float2` 解释即 16 组样本）
   - loss 输出：每 10 次迭代打印一次

## 实验步骤
### Step 1：设备与着色器能力初始化
1. 创建 RHI 设备并配置目标 profile。
2. 加载 `learnGradient`、`adjustParameters` 计算核。
3. 要求 cooperative vector 相关能力可用（示例在核中声明对应 `require` 能力）。

### Step 2：构建三段式参数/梯度存储
1. 分配权重、偏置及 row-major 梯度段。
2. 额外分配 training-optimal 权重梯度段。
3. 参数随机初始化到 `[-1,1]`（转换为 half 存储）。
4. 创建并清零 Adam 状态。

### Step 3：准备网络地址表与训练输入
1. 将各层权重地址、training-optimal 梯度地址、偏置地址等写入常量缓冲。
2. 创建随机输入样本缓冲。
3. 创建 loss 缓冲用于每轮 loss 输出。

### Step 4：执行训练循环（1000 次）
1. 清零 loss。
2. 清零 training-optimal 权重梯度区。
3. 调度 `learnGradient` 计算梯度（权重梯度写入 training-optimal 布局）。
4. 执行 cooperative vector 矩阵布局转换：training-optimal -> row-major。
5. 调度 `adjustParameters`，用 Adam 更新参数。
6. 每 10 次迭代读取并打印 loss。

### Step 5：记录并分析结果
1. 检查 loss 是否随训练整体下降。
2. 观察 half 精度训练下的稳定性（必要时关注数值波动）。
3. 与基础版本对照时，保持相同任务与迭代设置，比较收敛速度与执行效率。

## 实验结果
1. **结果形式**：输出 `Loss after 10/20/.../1000 iterations: <value>`。
2. **功能性结论**：示例实现了 cooperative vector 参与的端到端训练闭环。
3. **工程性结论**：相比基础版本，coopvec 版本引入“training-optimal 梯度缓存 + 布局转换”机制，是其主要实现差异与潜在加速来源。
4. **对实验设计的意义**：该示例适合作为“硬件特性对神经网络训练影响”的对照实验对象。

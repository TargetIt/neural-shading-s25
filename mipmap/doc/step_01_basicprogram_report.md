# Step 01：基础 BRDF 渲染实验报告

## 1. 实验目的
- 搭建 SlangPy + Python 的最小可运行渲染流程。
- 验证材质贴图（albedo/normal/roughness）驱动的 BRDF 着色结果。
- 建立后续 Mipmap、损失计算与训练实验的基线参考。

## 2. 实验原理
- 在 `step_01_basicprogram.slang` 中定义 `MaterialParameters`，按像素读取 albedo、normal、roughness。
- 使用 `eval_brdf(...)` 计算每个像素的反射响应，结合光照与视线方向输出颜色。
- Python 端每帧调用 `module.render(...)`，将结果写入输出张量并显示到窗口。

## 3. 实验手段
- 语言与框架：Python + SlangPy。
- 数据输入：三张 2K 材质贴图（漫反射、法线、粗糙度）。
- 执行方式：逐帧全分辨率渲染并 `blit` 到窗口。

## 4. 实验条件
- 输出窗口：`1024x1024`。
- 光照方向：归一化 `float3(0.2, 0.2, 1.0)`。
- 观察方向：`float3(0, 0, 1)`。
- 光照强度：shader 内固定 `1.5`。
- 贴图预处理：
  - albedo 开启 `linearize=True`；
  - normal 使用 `scale=2, offset=-1` 映射到 `[-1,1]`；
  - roughness 以灰度读入。

## 5. 实验步骤
1. 初始化 `App` 与 `step_01_basicprogram.slang` 模块。
2. 读取三类材质贴图到 GPU Tensor。
3. 在事件循环内创建与 albedo 同尺寸输出张量。
4. 调用 `render(pixel=call_id(), material=..., light_dir, view_dir)` 执行逐像素着色。
5. 将输出显示到窗口并刷新。

## 6. 实验结果
- 成功得到**全分辨率参考图像**，能够体现法线细节与高光粗糙度差异。
- 该结果构成后续 Step 02~05 的“高质量基准”：
  - Step 02 用于对比“先降采样输入再渲染”；
  - Step 03 用于对比“先渲染再降采样”；
  - Step 04/05 进一步围绕该基准定义误差与优化。

## 7. 注意事项
- normal 贴图必须先做 `scale=2, offset=-1`，否则法线方向会错误，渲染高光明显异常。
- albedo 建议在线性空间参与 BRDF 计算（`linearize=True`），避免 gamma 空间下能量不正确。
- `light_dir`、`view_dir` 与 normal 都应归一化，保证 BRDF 计算稳定。

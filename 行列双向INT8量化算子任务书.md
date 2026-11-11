# 题目2-行列双向INT8量化算子开发任务书

## 1. 任务概述

使用 Ascend C 实现行列双向 INT8 量化算子，对同一 FP16 矩阵同时进行按行和按列量化，输出两份 INT8 矩阵及对应的行列绝对最大值。

## 2. 核心开发要求

### 2.1 功能实现要求

对一个 FP16 矩阵同时输出按整行和整列绝对最大值量化的两份 INT8 矩阵，以及两组 FP32 绝对最大值。只覆盖稠密路径，不包含 threshold 或 COO 异常值输出。

### 2.2 算子工程模式

采用 ACLNN 算子工程化开发，提供 Ascend C 内核、OpDef 注册、shape/dtype 推导、tiling、可构建安装包、两阶段 ACLNN API 和独立 C++ 调用样例。安装包安装到用户目录，不覆盖系统 CANN。

同时需要支持kernel直调工程开发，可复用同一tiling计算函数或者等价的参数构造逻辑

提供 PyTorch NPU 接口 `run(inputs, attributes) -> tuple[Tensor,...]`，调用本题 ACLNN API，作为精度、性能和内存验收入口；同时提供 Kernel 直调入口进行交叉验证。计算和数据搬运均在 NPU 上完成，使用当前 stream，不修改输入。输出须为同一设备上的独立连续张量。

支持多阶段实现。框架入口须校验输入个数、维度、dtype、连续性、尺寸和属性，对非法或未知属性报错。C++ ACLNN 调用方须保证设备指针有效、ND 张量描述正确，且输出存储不重叠。

### 2.3 接口定义

#### 算子语义

`r[i]=max_j(abs(x[i,j]))`，`c[j]=max_i(abs(x[i,j]))`。

非零统计量 s 对应的量化定义为 `int8(clip(rint(fp32(x*fp32(127/s))),-127,127))`；rint 使用最近偶数舍入。统计量为零时对应量化结果全零。输出统计量是绝对最大值，不是反量化 scale。qCol 的形状和行主序布局仍为 [M,N]。

#### 参数说明

| 参数 | 形状 | 类型 | 说明 |
|---|---|---|---|
| x | [M,N] | FP16 | ND 连续、有限值 |
| qRow、qCol | [M,N] | INT8 | 分别按整行、整列量化 |
| rowAbsmax | [M] | FP32 | 行绝对最大值 |
| colAbsmax | [N] | FP32 | 列绝对最大值 |

#### 算子约束

范围为 1≤M≤4096、2≤N≤4096，且 N 为偶数。输入为有限 FP16 值；不包含 NaN/Inf、稀疏异常值或反向。拒绝类型、维度、连续性及尺寸错误。输出范围固定为 [-127,127]。

## 3. 验收标准

### 3.1 软硬件环境要求

- **适配硬件**：昇腾 910C（Atlas A3）单个逻辑设备，48 个 Vector Core、24 个 Cube Core。
- **CANN 版本**：9.1.0。
- **Python 版本**：3.12.13
- **框架版本**：PyTorch 2.10.0、torch_npu 2.10.0.post4+git5dd8ef3。

开发镜像及依赖见 3.5。使用其他软硬件版本时须重新验证。

### 3.2 精度要求

行列绝对最大值要求逐位一致。常规 INT8 要求逐元素一致；仅当 FP32 标杆量化前数值距半整数≤2e-5 时，允许相差 1，并记录涉及元素数。专门的精确中点及全零用例必须完全一致。不能全局采用±1 容差。

精度测试覆盖 `case.json` 中全部 **27** 个用例。使用 `generate_data.py` 生成固定种子输入，以 `golden.py` 的 CPU 结果为标杆，按 3.5 验证 `run()` 输出及存储要求。

### 3.3 性能要求

分别比较 FP32 Abs/ReduceMax/Round/Cast 组合，以及两次现有 DynamicQuant 加转置和独立 absmax 的组合。仅使用通过相同精度检查的实现，逐用例选择较快者。

各用例须满足 `提交实现耗时 / 最快合格基线耗时 ≤ 下表对应的耗时比上限`。提交实现与基线须在同一 NPU 环境、相同输入下重新测量，每次调用均分配新输出。下表耗时为参考值，验收以同机实测比值为准。

| 用例 ID | 基线 μs | 目标耗时上限us |
|---|---:|---:|
| RowColInt8Quant_004_float16_random | 148.01 | 32 |
| RowColInt8Quant_005_float16_random | 116.56 | 35 |
| RowColInt8Quant_006_float16_random | 167.50 | 73 |
| RowColInt8Quant_007_float16_random | 371.88 |            244 |
| RowColInt8Quant_008_float16_random | 121.47 | 42 |

#### 基线计算定义

基线代码如下，`candidates(inputs, attrs)` 返回各候选实现的调用函数。输入张量须位于 NPU。

```python
import torch
import torch_npu
import torch.nn.functional as F

def candidates(inputs, attrs):
    x = inputs[0]

    def row_col_composition():
        a = x.float()
        absolute = a.abs()
        r = absolute.amax(dim=1)
        c = absolute.amax(dim=0)

        def quant(scale):
            safe = torch.where(scale == 0, torch.ones_like(scale), scale)
            return (a * (127.0 / safe)).round().clamp(-127, 127).to(torch.int8)
        return (quant(r[:, None]), quant(c[None, :]), r, c)

    def two_dynamic_quant():
        (qr, _) = torch_npu.npu_dynamic_quant(x)
        (qc, _) = torch_npu.npu_dynamic_quant(x.t().contiguous())
        absolute = x.float().abs()
        return (qr, qc.t().contiguous(), absolute.amax(dim=1), absolute.amax(dim=0))
    return {'fp32_absmax_round_cast': row_col_composition, 'two_dynamic_quant_transpose': two_dynamic_quant}
```

计时前，提交实现和候选基线均须通过 3.2 的精度检查，每个用例至少保留一个合格基线。分别预热 50 次并同步设备，再测量 5 轮，每轮连续调用 200 次。从每轮开始提交前计时，到该轮设备操作全部完成为止，除以 200 得到单次耗时，取 5 轮中位数。输入传输和标杆比较不计入耗时。记录每轮数据、入选基线及排除原因。

同时记录设备事件时间，其中包含提交间隙，不作为纯内核耗时。

### 3.4 内存要求

ACLNN 显式工作区上限为 `16 MiB + 4*ceil(M/64)*ceil(N/256)*(64+256)` 字节。16 MiB 是本环境 CANN 为核间同步保留的系统工作区，其余为行列局部统计，均计入工作区。不分配完整 FP32 矩阵或全矩阵转置作为自定义内核工作区。在输入已分配、运行时已预热的前提下，活跃张量分配峰值增量不超过输出字节数、上述工作区上限及 16 KiB 分配对齐余量之和。释放并同步后，重复 20 次调用的活跃张量分配须回到调用前水平。

内存按 PyTorch NPU 分配器的活跃张量字节数统计，包含输出和工作区，不计整卡 HBM、驱动缓存和片上 UB。报告须说明片上缓冲设计，并通过 3.5 的非法元数据、stream、输入提前释放和重复调用测试。

### 3.5 自验要求

#### 测试shape

必须执行 case.json 中全部 **27 个精度用例**，其中标记为 performance 的 **5 个**同时用于性能和内存验收。下表与 case.json 一一对应；尺寸元组分别表示 (M,N)。

| 用例 ID | 尺寸 | dtype | 输入特征/属性 | 性能及内存 |
|---|---|---|---|---|
| RowColInt8Quant_000_float16_random | (1, 2) | float16 | 随机 | 否 |
| RowColInt8Quant_001_float16_random | (2, 2) | float16 | 随机 | 否 |
| RowColInt8Quant_002_float16_random | (3, 6) | float16 | 随机 | 否 |
| RowColInt8Quant_003_float16_random | (17, 66) | float16 | 随机 | 否 |
| RowColInt8Quant_004_float16_random | (257, 1026) | float16 | 随机 | 是 |
| RowColInt8Quant_005_float16_random | (128, 4096) | float16 | 随机 | 是 |
| RowColInt8Quant_006_float16_random | (1024, 4096) | float16 | 随机 | 是 |
| RowColInt8Quant_007_float16_random | (4096, 4096) | float16 | 随机 | 是 |
| RowColInt8Quant_008_float16_random | (4096, 256) | float16 | 随机 | 是 |
| RowColInt8Quant_009_float16_zeros | (17, 66) | float16 | 全零 | 否 |
| RowColInt8Quant_010_float16_zero_axes | (17, 66) | float16 | 零行和零列 | 否 |
| RowColInt8Quant_011_float16_extreme | (17, 66) | float16 | 极值和次正规数 | 否 |
| RowColInt8Quant_012_float16_ties_row | (1, 8) | float16 | 行方向精确中点 | 否 |
| RowColInt8Quant_013_float16_ties_col | (8, 2) | float16 | 列方向精确中点 | 否 |
| RowColInt8Quant_014_float16_boundary | (1, 4096) | float16 | 随机 | 否 |
| RowColInt8Quant_015_float16_boundary | (4096, 2) | float16 | 随机 | 否 |
| RowColInt8Quant_016_float16_boundary | (64, 254) | float16 | 随机 | 否 |
| RowColInt8Quant_017_float16_boundary | (65, 258) | float16 | 随机 | 否 |
| RowColInt8Quant_018_float16_boundary | (33, 510) | float16 | 随机 | 否 |
| RowColInt8Quant_audit_019_float16_random | (4096, 2) | float16 | 随机 | 否 |
| RowColInt8Quant_audit_020_float16_random | (1, 4096) | float16 | 随机 | 否 |
| RowColInt8Quant_audit_021_float16_random | (64, 256) | float16 | 随机 | 否 |
| RowColInt8Quant_audit_022_float16_random | (65, 258) | float16 | 随机 | 否 |
| RowColInt8Quant_audit_023_float16_random | (32, 254) | float16 | 随机 | 否 |
| RowColInt8Quant_audit_024_float16_random | (33, 256) | float16 | 随机 | 否 |
| RowColInt8Quant_audit_025_float16_subnormal | (65, 258) | float16 | 正负最小次正规数 | 否 |
| RowColInt8Quant_audit_026_float16_signed_zero | (33, 66) | float16 | 正负零 | 否 |

工程测试必须覆盖两个独立 stream、提前释放输入后另一 stream 的分配压力，以及直调和 ACLNN 各重复 20 次调用后的活跃张量回收。另用现有公开输入保留 8 次调用的全部输出，同时执行 32 次提前释放输出和同流分配写入压力；8 组输出必须保持独立且结果正确。重复调用不另计独立 case。

本题共 **39 个有效输入用例**（27 个公开 + 12 个额外种子/存储偏移），另有 **13 个非法输入/属性场景**。非法场景在三个入口执行得到 39 条检查记录；额外输入在两个入口执行得到 24 条记录。性能、内存、预热及重复调用均复用上述输入，不另计用例。

#### 环境依赖与数据生成

开发镜像：`quay.io/jd_xllm/xllm-ai:xllm-dev-cann9.1.0-a3-openeuler24.03-py3.12`。

镜像依赖：Python 3.12.13、torch 2.10.0+cpu、torch_npu 2.10.0.post4+git5dd8ef3、NumPy 2.2.6、ml_dtypes 0.6.0。torch 使用 +cpu 基础包，配合 torch_npu 和自定义内核执行 NPU 计算。

按分配的设备设置 `ASCEND_RT_VISIBLE_DEVICES`，容器内使用逻辑设备 0。运行前挂载驱动目录并加载 CANN 环境。

从本题目录执行以下命令，按 `case.json` 导出全部测试输入：

```bash
python3 generate_data.py --output-dir generated_data
```

可用 `--case 用例ID` 选择单个用例，重复传入该参数可选择多个用例。脚本生成原始 `.bin` 输入和 `manifest.json`，记录文件字节数及 SHA-256。输入生成使用 `SHA-256("20261014/" + case_name)` 的前 8 字节按小端序解释为种子，交给 NumPy 默认随机数生成器；特殊数据模式由用例的 `kind` 决定。

输入和输出顺序与 `op.json` 一致，属性为名称到数值的字典。可直接调用 `generate_data.generate(case)` 生成内存输入，无需先导出二进制文件；精度期望值调用 `golden.calculate(inputs, attributes)` 获取，`golden.calc_expect_func` 也提供标杆调用入口。使用其他测试框架时，按上述接口接入。

#### 精度自验证步骤

1. 逐条读取 `case.json`，生成确定性输入，并从 `attr_desc` 构造属性字典；先以 CPU 输入计算标杆。
2. 输入按原始 16 位表示传到选定 NPU，再解释为 FP16/BF16，避免改写特殊位模式。调用提交实现 `run(inputs, attributes)` 并同步设备。
3. 检查输出个数、NPU 设备及编号、shape、dtype、连续性；逐项按 3.2 的精度规则比较。BF16 回传也按原始 16 位解释，避免浮点转换改变位模式。
4. 所有非空输出的底层存储地址必须互不相同，且不得与任一非空输入的底层存储相同；不能只比较各张量的起始指针。逐位核对全部输入未被修改。
5. 每个用例保存输出检查结果与通过／失败状态。记录量化输出的不一致元素数和半整数邻域元素数，并记录 absmax 逐位比较结果。任何失败返回非零退出码并保存错误日志，不得仅以总体状态代替逐项检查。

#### 必测覆盖清单

公开用例的 ID、输入输出 shape、dtype、属性和数据模式以本包 `case.json` 为准。性能和内存用例为其中 `performance=true` 的子集，与上表一致。规定用例不得删除、修改输入定义或跳过，可另行增加用例。

本包公开用例文件的 SHA-256 为 `f0e6fdbf77c27960762db89253044323244e7ab4b549d2bf4e4d4d6fe0d8be76`。新增用例时保留原有定义。

**非法输入和属性场景**

以下场景须在 `prepare`、`run`、`run_direct` 三个入口分别测试，均应报错。以第一个性能用例为基础，仅修改表中指定项，使用本题输入类型和 NPU 设备构造张量。

| 场景 ID | 构造／检查内容 |
|---|---|
| `attributes_type` | 属性容器为列表而非字典 |
| `bf16` | 输入为 BF16 |
| `cpu_input` | 第一个输入位于 CPU |
| `empty_m` | M=0 |
| `empty_n` | N=0 |
| `float32` | 第一个输入为 FP32 |
| `input_count` | 缺少最后一个输入 |
| `large_m` | M=4097 |
| `large_n` | N=4098 |
| `noncontiguous` | 输入为隔元素切片等非连续布局 |
| `odd_n` | N=3，为奇数 |
| `rank` | x 形状为 (2,2,2) |
| `unknown_attribute` | 包含未定义的属性 unknown |

**额外固定种子和存储偏移用例**

下列 12 个用例分别经过 `run` 和 `run_direct`，共 24 条记录。用例名的 `__extra_seed_` 前缀部分对应公开用例，复制其输入输出描述及属性，将名称替换为完整 ID，设置 `kind="random"`、`performance=false`，再调用 `generate_data.generate(case)`。数据模式统一使用 `kind="random"`，不按用例名称中的 zeros、bits 等字样生成。

| 额外用例 ID |
|---|
| `RowColInt8Quant_000_float16_random__extra_seed_0` |
| `RowColInt8Quant_001_float16_random__extra_seed_1` |
| `RowColInt8Quant_002_float16_random__extra_seed_2` |
| `RowColInt8Quant_002_float16_random__extra_seed_6` |
| `RowColInt8Quant_002_float16_random__extra_seed_7` |
| `RowColInt8Quant_003_float16_random__extra_seed_3` |
| `RowColInt8Quant_004_float16_random__extra_seed_4` |
| `RowColInt8Quant_005_float16_random__extra_seed_5` |
| `RowColInt8Quant_008_float16_random__extra_seed_8` |
| `RowColInt8Quant_009_float16_zeros__extra_seed_9` |
| `RowColInt8Quant_010_float16_zero_axes__extra_seed_10` |
| `RowColInt8Quant_011_float16_extreme__extra_seed_11` |

每个输入张量分配“元素数 + 16”的同类型设备存储，取从第 1 个元素偏移开始、长度等于原元素数的视图，恢复原 shape 后复制输入。输出必须通过相同精度检查，输入位模式保持不变，输出不得与输入或其他输出共享底层存储。

**验证器与工程检查**

- 验证器必须拒绝三类反例：CPU 输出、复用输入存储、起始指针不同但底层存储重叠的输出。
- 两个独立 stream 分别使用 `RowColInt8Quant_004_float16_random`、`RowColInt8Quant_008_float16_random`。每个 stream 至少执行 10 次准备后的调用，并检查正式 ACLNN 入口结果。
- 工程测试另外至少执行 6 次非法元数据拒绝检查；提前释放输入后的跨 stream 分配压力、重复调用回收和输出生命周期要求见前述工程要求。
- 按“用例 ID、入口”逐项检查 NPU 测试记录。精度、性能、内存和扩展测试全部通过方可验收，重复记录不重复计数。

## 4. 验收交付件

| 序号 | 交付件名称 | 交付件要求 |
|---|---|---|
| 1 | 算子设计文档 | 算法、分核、数值/地址推导、片上缓冲、工作区预算及接口设计。正式社区设计评审按任务仓流程提交。 |
| 2 | 自测用例及测试代码 | 覆盖全部公开精度用例、标记的性能/内存用例及 3.5 的工程检查；提供独立标杆、公开验证器和完整复现步骤。 |
| 3 | 自测报告 | 按[官方自验证材料模板](https://cann-op.obs.cn-north-4.myhuaweicloud.com/task_submission.zip)提交下列七件，填写真实精度、性能、内存数据并保存原始日志；另附环境和源码指纹。 |
| 4 | 待验收代码及工程 | 完整 ACLNN 源码、构建/私有安装脚本、.run 安装包、两阶段 API、独立 C++ 样例、PyTorch NPU 封装及直调交叉验证入口。提交个人代码仓地址、分支、算子目录和固定 commit。 |

交付件放在提交工程的 task_submission/，并将同名七件打包为 task_submission.zip：

```text
task_submission/
  1 自验证步骤说明.md
  2.1 精度自验证报告.xlsx
  2.2 精度自验证日志.log
  3.1 性能自验证报告.xlsx
  3.2 性能自验证日志.log
  4.1 内存自验证报告.xlsx
  4.2 内存自验证日志.log
```

## 5. PR 申请合入

https://github.com/xLLM-AI/xllm/tree/main/xllm/core/kernels/npu/xllm_ops

## 6. 参考资料

- [天数公开功能对标](https://developer.iluvatar.com/docs/ixformer#bnb_double_quant)。
- [Ascend C Kernel 直调说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/appendix/completing_kernel_launch_based_on_sample_project.md)。

## 7. 环境获取

使用发布方分配或自行准备的 910C 开发环境，按 3.1 和 3.5 配置软件及镜像。活动算力申请按官方流程办理。

## 8. 特别注意事项

1. 输入值域和布局须符合接口要求，非法元数据须报错。
2. 算子须在 NPU 上完成计算，不得返回预存结果或使用 CPU 代算。
3. 性能以本题 NPU 基线为准。测试受其他任务干扰时，保留日志并重新测量。
4. 自测报告须与实际结果一致，失败或中断的测试不得标记为通过。

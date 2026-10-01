# 从零复现一个案例：新用户 Playbook

本页面向**第一次接触 PaddleScience** 的新用户，以 [EMS 电源平面 DC IR-Drop 仿真](examples/ems_ir_drop.md) 为完整范本，给出从环境准备到提交 PR 的全流程路线图。任何一条流程都同样适用于复现仓库中其它案例。

## 0. 全景图：一个 PINN 案例由什么构成

PaddleScience 是配置驱动（Hydra + OmegaConf + pydantic）的科学计算库。一个标准案例 = 一个脚本 + 一份配置：

```
examples/<case>/
├── <case>.py          # @hydra.main 入口, 按 cfg.mode 分发 train/eval/export/infer
└── conf/<case>.yaml   # 几何、采样、模型、训练、评估、推理全部在此声明
```

运行时数据流（与 [EMS 案例](examples/ems_ir_drop.md) 的代码段落一一对应）：

```
cfg (yaml) ──► ppsci.arch.<Model>(**cfg.MODEL)        ← 模型
          ──► ppsci.equation.<PDE>                    ← sympy 表达式方程
          ──► ppsci.geometry.* + 采样                  ← 计算域
          ──► ppsci.constraint.<Constraint>            ← 方程+边界条件 → 损失
          ──► ppsci.optimizer.<Opt>(model)             ← 优化器
          ──► ppsci.validate.<Validator>               ← 验证器
          ──► ppsci.solver.Solver(...).train()         ← 训练循环
```

两类工厂模式要分清（初学者最常见的混淆点）：

| 模式 | 写法 | 用于 |
| :-- | :-- | :-- |
| `name:`-eval | `cfg.pop("name")` → `eval(name)(**cfg)` | `MODEL` / `loss` / `dataset` 等单选场景 |
| 字典键分发 | 配置项的键名即类名，内部 `name:` 只是实例标签 | `constraint` / `validator` / `EQUATION` 等多实例场景 |

## 1. 环境准备（10 分钟）

``` sh
# 1. PaddlePaddle 先单独装（CPU 或 GPU 按机器选择）
pip install paddlepaddle        # 或 pip install paddlepaddle-gpu

# 2. 克隆并安装 PaddleScience
git clone https://github.com/PaddlePaddle/PaddleScience.git
cd PaddleScience
pip install -r requirements.txt
python -m pip install -e .

# 3. 安装质量门工具（pytest 与 pre-commit 不在 requirements 里）
pip install pytest pre-commit

# 4. 验证安装
python -c "import ppsci; ppsci.utils.run_check()"
```

## 2. 跑通第一个案例（30 分钟）

从 [examples/ems_ir_drop](https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/ems_ir_drop) 开始——它无需下载数据集（参考解由内置 FDM 求解器现场计算），CPU 上约 22 分钟即可完成一次完整训练：

``` sh
cd examples/ems_ir_drop
python ems_ir_drop.py                      # 训练 800 epochs + 自动评估
```

三个立即可玩的实验（熟悉 Hydra override 语法——任何配置键都能从命令行覆盖）：

``` sh
# a. 缩短训练观察收敛趋势
python ems_ir_drop.py TRAIN.epochs=200

# b. 调整锚点监督强度, 观察它对最终精度的影响
python ems_ir_drop.py +TRAIN.anchor_points=1000 TRAIN.anchor_weight=5.0

# c. 训练完成后单独评估/导出/推理
python ems_ir_drop.py mode=eval EVAL.pretrained_model_path=<checkpoint路径>
python ems_ir_drop.py mode=export INFER.pretrained_model_path=<checkpoint路径>
```

训练日志会输出两层数字，**学会区分它们**：

- `[Train]` 各约束损失（EQ / VRM / PAD_* / EDGE / HOLE / ANCHOR）——只反映约束满足程度；
- `[Eval][Avg] FDM_ref/L2Rel.u_ref` 与焊盘压降表——**真正的验收指标**。

在边界驱动类问题里二者可能严重脱钩（训练损失下降不代表接近真解），这是 PINN 的已知病理，也是 EMS 案例采用"稀疏锚点弱监督"的原因（详见案例文档 2.3 节）。

## 3. 理解案例设计（1-2 小时）

按下面的阅读顺序啃透一个案例，胜过泛读十个：

1. **配置先行**：通读 `conf/ems_ir_drop.yaml`，每一节对应一个构建步骤（几何 → 采样规模 → 模型 → 训练）。改配置 = 改实验，不需要动代码。
2. **按脚本函数地图读代码**（`ems_ir_drop.py`）：

| 函数 | 行号 | 职责 |
| :-- | :-- | :-- |
| `LaplaceV` | L65 | 自定义 sympy PDE（继承 `ppsci.equation.pde.PDE`） |
| `_u_singularity_terms` / `_u_s` | L86 / L104 | 奇性分离解析项 $u_s$ |
| `solve_ir_drop_fd` | L156 | 内置 FDM 参考解（自研验证基准） |
| `build_geometry` | L308 | CSG 几何组装 |
| `build_anchor_points` | L348 | 从 FDM 网格抽稀疏锚点 |
| `build_constraints` | L395 | 五组约束（PDE/边界/锚点） |
| `build_validator` | L520 | 对照 FDM 的验证器 |
| `train` / `evaluate` / `export` / `inference` | L614+ | 四模式入口 |

3. **对照 API 文档**：遇到不认识的类（如 `InteriorConstraint`），在 [API 文档](api/arch.md) 查其参数语义；遇到不认识的 yaml 键，去 `ppsci/utils/config.py` 找对应 pydantic schema。
4. **读测试**：`test/` 目录镜像 `ppsci` 结构，是各类用法最权威的示例（例如 `test/equation/test_laplace.py` 展示如何对照手写梯度验证方程实现）。

## 4. 换一个案例练手（按兴趣选）

| 兴趣方向 | 推荐案例 | 特点 |
| :-- | :-- | :-- |
| 最小 PINN 入门 | `examples/laplace` | 纯方程驱动、无数据依赖、结构最简 |
| 数据驱动替代仿真 | `examples/chip_heat` | PI-DeepONet 多分支结构 |
| 工业落地范本 | `examples/ems_ir_drop` | 双基准验证 + 部署链路 + 工程叙事 |
| CFD / 流体 | `examples/cylinder` | 经典外流算例，含 transformer 变体 |

## 5. 下一步：改出你自己的案例

把一个新问题装进 PaddleScience，按 checklist 逐项推进（每一步都能在 EMS 案例中找到对应物）：

1. **数学建模**：写出控制方程 + 边界/初始条件；判断是否需要奇性分离（源项附近有 $1/r$、$\ln(1/r)$ 类奇性时必须分离，否则网络塌缩）。
2. **验证基准先行**：先做一个自研参考解（FDM/解析/文献数据），没有基准就无法验收——这是 EMS 案例最重要的方法论遗产。
3. **几何与采样**：CSG 表达计算域；为每类边界单独设约束并**保证损失键唯一**。
4. **配置组装**：按 `MODEL` → 约束 → 优化器 → 验证器顺序写 yaml；新键需在 `ppsci/utils/config.py` 加 pydantic 字段。
5. **训练诊断**：训练损失与验收指标脱钩时，先做小规模对照实验（纯回归上限、单项约束消融）定位病灶，再选对策（锚点监督/课程训练/几何简化）。
6. **完整交付**：train/eval/export/infer 四模式 + 中英双语文档 + mkdocs nav + README 行，最后跑 `pre-commit run --all-files`。

## 6. 常见坑速查

| 症状 | 原因与解法 |
| :-- | :-- |
| Hydra 报 `Key not in struct` | 新增的 yaml 键用 override 时缺 `+` 前缀：`+TRAIN.xxx=...` |
| 训练损失降但指标不动 | PINN 欺骗解/梯度病理；建立独立验收基准，勿信训练损失 |
| 方程二阶导爆炸或 NaN | 源项奇性未分离，或网络表达准不连续跳变被 EQ 惩罚压平 |
| `pytest test/` 收集不全 | `test/loss/` 部分文件无 `test_` 前缀，需 `pytest -o python_files=*.py test/loss/` |
| 权重加载失败 | URL 自动下载依赖网络；本地路径用 `EVAL.pretrained_model_path=<绝对路径>` |
| `outputs*/` 出现在 git status | Hydra 运行产物，**永远不要提交** |
| 想改配置但怕破坏环境 | Hydra override 全部命令行完成，`git checkout conf/` 即可还原 |

## 7. 求助渠道

- 各案例文档末尾的"参考资料"；
- [GitHub Issues](https://github.com/PaddlePaddle/PaddleScience/issues)；
- [开发与复现指南](development.md) —— 提 PR 前必读。

!!! tip "一条心法"

    配置驱动意味着：**先改 yaml 做实验，只有 yaml 表达不了的东西才动代码**。新用户 90% 的困惑来自试图用写代码的方式解决一个配置问题。

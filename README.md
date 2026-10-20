# OpenFOAM 的 GPU 求解器 chtMultiRegionGpuFoam（C++，2026-10-19）

用户 10-19 定：做一个真正的 OpenFOAM v2606 程序（`wmake` 编译、链接 OpenFOAM 库，像 `chtMultiRegionFoam` 一样直接读写多区算例），整个时间步都在 GPU 上算；不走“只把线性方程交给 GPU”的插件路线（实测比 24 核 CPU 慢，见 `../gpu_plugins_g4_20261019/README.md`）。算法对照已经和 OpenFOAM 对上的 Python 版 `../gpu_cht_solver_r1/`（下称 Py 版，只读保留作对照）。

**M0～M7 全部完成并通过双后端（`BACKEND=omp` 与 NAS 直通 RTX 3080 `BACKEND=cuda ARCH=sm_86`）对照验证。**

## 1. 结构

```
gpu_cht_of_r1/
  lib/                              计算库 libgpucht.so，不含 OpenFOAM 头文件；同一份 C++17 源码两种编法：
                                      make BACKEND=omp                g++ -fopenmp（Thrust OMP 后端）
                                      make BACKEND=cuda ARCH=sm_86    nvcc（RTX 3080 / A100 sm_80 / G4 sm_89 等）
    par.h                           后端层：dvec（thrust::device_vector）、for_n / sum_n / max_n / min_n、HD 宏
    vec.h                           三维向量/对称张量工具（核函数内联）
    api.h                           OpenFOAM 程序与计算库接口（仅标准 C++ 类型）：MeshIn、SolverCtl、Engine
    mesh.h / mesh.cpp               全局多区设备网格 DMesh + 单元→面邻接表（按单元收集求和，无原子操作，结果确定）
    ldu.h / ldu.cpp                 LDU 矩阵：Amul、sumA、relax（v2606 fvMatrix::relax）、setValues
    linsolve.h / linsolve.cpp       柔性 PCG、PBiCGStab、Jacobi 光滑、聚合 AMG（最细层直接 LDU，粗层 CSR）
    energy.h / energy.cpp           全区单矩阵共轭能量方程（T 形式解 he 方程，耦合界面串联导热 + 接触热阻层）
    flow.h / flow.cpp               流体区 PIMPLE：动量 UEqn、压力 pEqn（p_rgh）、多组分 YEqn、RNG/realizable k-ε + 壁面函数
    building.h / building.cpp       建筑物理：气象插值、太阳位置与正交阴影图光追（ShadowMap）、室外/室内长短波辐射分配
    engine.cpp                      Engine 实现（设备数组管理与各物理模块调度）
    core.cpp                        sync / 显存查询（cudaMemGetInfo / VmRSS）/ fatal
    Makefile
  app/chtMultiRegionGpuFoam/        OpenFOAM v2606 主程序（wmake 编译）
    chtMultiRegionGpuFoam.C         主循环（含 adjustTimeStep：maxCo / maxDi / maxDeltaT，与 v2606 逐行一致）
    regions.H / regions.C           多区网格读取、mappedPatchBase 耦合界面配对与全局网格拼装
    energyCoupling.H / .C           固体/流体热物性、T 边界条件（含 gpuCht::exteriorSurface、externalWallHeatFluxTemperature）、
                                    fvOptions（scalarSemiImplicitSource、fixedTemperatureConstraint）、建筑模块初始化与写回
    fluidCoupling.H / .C            流体区场（U, p_rgh, p, rho, phi, Y_i, k, epsilon, nut, alphat）、边界条件、
                                    内嵌 libpython3 零拷贝调用冻结 HVAC 盘管/新风闭环控制器（postProcessing/hvac/hvac.json）
    Make/files, Make/options
  test/test_lib.cpp                 不依赖 OpenFOAM 的库级自检（双材料棒解析解、48³ 立方体 AMG、ShadowMap 太阳光追自检）
  tools/
    nas.sh                          同步源码到 NAS 并在隔离容器（支持 --gpus all 直通 RTX 3080）内执行命令
    build_app.sh                    一键编译 libgpucht.so、test_lib 与 chtMultiRegionGpuFoam（BACKEND=omp|cuda）
  verify/
    compare_of.py                   OpenFOAM 场对场最大误差 / 均方根误差对比脚本
    m1.sh ~ m7.sh                   M1～M7 分级验证脚本
```

NAS 环境：`user@gpu-host`，工作目录 `/path/to/work/gpu_cht_of_r1`（容器内挂载为 `/w`，Py 版只读挂载为 `/py`）。容器镜像 `local/openfoam2606-cuda`（OpenFOAM v2606 + g++ 13.3 + CUDA 12.8 挂载于 `/usr/local/cuda`），NAS 显卡 **NVIDIA GeForce RTX 3080 (20 GB VRAM, `sm_86`, Driver 580.142)** 通过 `--gpus all` 直通容器。

## 2. 核心设计要点与对齐的 OpenFOAM v2606 细节

- **几何量 100% 取自 OpenFOAM 原生对象**：直接读取 `mesh.weights()`、`nonOrthDeltaCoeffs()`、`nonOrthCorrectionVectors()`、`leastSquaresVectors::New(mesh).pVectors()/nVectors()`（含边界面与耦合面）、`patch.deltaCoeffs()`。无需在设备端重算最小二乘伪逆矩阵，既保证与 OpenFOAM 逐位一致，又省去每单元 3×3 逆矩阵显存。主机端 `MeshIn` 的双精度几何向量在 `Engine(mi)` 上传设备后立即释放（全楼节省 **19.53 GB 主机内存**）。
- **单矩阵全区共轭能量方程**：将所有流体区与固体区拼成同一个 LDU 矩阵求解 `he`（以 `T` 为未知量，列缩放 `cp`），耦合界面按两侧串联导热导纳（含 `thicknessLayers`/`kappaLayers` 接触热阻）直接进矩阵非对角项；在 `fvMatrix::relax` 和 `setValues` 中，界面导纳按 OpenFOAM 分区边界系数语义计入 `bDiag`/`bAbs` 而不计入内部面 `sumOff`。
- **OpenFOAM v2606 关键实现细节（逐项对表源码）**：
  1. `solveSolid.H` 中 `hEqn.relax()` 在设置 `finalIteration` 之前调用，故固体区松弛始终查 `"h"`，流体区在最后一次外迭代查 `"hFinal"`。
  2. `relaxEquation` / `relaxField` 在 `finalIter` 时严格查 `<name>Final`（无回退）。
  3. 动量 `UEqn.relax()` 对向量场将各分量边界系数最大值并入 `bAbs`、最小值并入 `bMin`。
  4. `pEqn.H` 中 `corr == 0` 查 `"p_rgh"`，仅当 `finalIter && corr == nCorr - 1` 时查 `"p_rghFinal"`；`p_rgh.storePrevIter()` 每时间步开头存一次；重启时 `fixedFluxPressure` 保留初始文件里的 `gradient()`。
  5. `YEqn.H` 多变量对流 `div(phi,Yi_h)` 同时限制所有组分 `Y_m` 与比焓 `he` 的最小面通量限制器权重，方程求解后立即更新混合物热物性 `thermo.correct()`（`Cp, e, kappa, mu, rho0, T0, beta`）。
  6. 湍流 `realizableKE` / `RNGkEpsilon`：壁面单元包含在耦合界面补丁（`IFACE_A`/`IFACE_B`）上的流体槽；`epsilonWallFunction` 按补丁自身 `Cmu, kappa, E` 计算 `yPlusLam` 并按面积加权壁面单元 `G` 与 `epsilon`（`setValues` 强加）；求解后执行 `bound(k, kMin)` 与 `bound(epsilon, epsMin)` 的邻域面加权回退。
  7. 自适应时间步 `adjustTimeStep`：流体 Courant 数 `0.5 * sumSurface(|phi|) / rho / V * dt` 与固体扩散数 `max(sqr(deltaCoeffs) * kappa/(rho*Cp)) * dt` 及其阻尼增长公式 `min(min(maxDeltaTFluid, 1.0 + 0.1*maxDeltaTFluid), 1.2)` 与 OpenFOAM v2606 完全一致。

## 3. M0～M7 验证结果汇总（OMP 与 CUDA 双后端全部通过）

| 里程碑 | 验证脚本 | 算例规模与内容 | 对照基准与实测误差（`BACKEND=omp` & `BACKEND=cuda`） |
| --- | --- | --- | --- |
| **M0** | `test/test_lib.cpp` | 双材料棒解析解、48³ 立方体 PCG/AMG/BiCGStab、ShadowMap 光追 | 双材料棒与解析解最大差 `2.84e-14 K`；PCG+AMG（37 步）与 PCG+Jacobi（265 步）差 `1.67e-12`；ShadowMap 地面阴影面积 `4.000 m²`（解析值 `4.000`），透射比 `0.965` |
| **M1** | `verify/m1.sh` | 斜六面体非正交双固体区锯齿界面、时变固定值 + `externalWallHeatFluxTemperature`（含辐射） | 对照 OpenFOAM v2606 `chtMultiRegionFoam`（10 步到 `t=50 s`）：两固体区 `T` 最大差 **`1.14e-11 K`**（达写出精度上限） |
| **M2** | `verify/m2.sh` | V5 共轭方腔（空气 Boussinesq + 4 固体壁面，4000 步稳态起热面阶跃 +5 K，30 步 × 15 外迭代） | 对照 `chtMultiRegionFoam`：首步动量初始残差对到 **12 位有效数字**；`t=60 s` 时 `air/T` 最大差 `4.10e-5 K`（量程 5 K），`air/U` 最大差 `4.9e-7 m/s`，固体 `T` 差 `≤ 1.8e-5 K` |
| **M3** | `verify/m3.sh` | 湿空气双组分（`dryAir + waterVapour`）+ 温度梯度腔体（含多变量 `limitedLinear`、物性随组分变化） | 对照 `chtMultiRegionFoam`：`air/waterVapour` 最大差 **`4.36e-9`**，`air/T` 最大差 `4.64e-6 K`，`air/U` 最大差 `1.69e-8 m/s`，`solid/T` 差 `3.05e-7 K` |
| **M4** | `verify/m4.sh` | V5 湍流共轭方腔（`realizableKE` + `kqRWallFunction` / `epsilonWallFunction` / `nutkWallFunction` / `alphatJayatillekeWallFunction`） | 对照 `chtMultiRegionFoam`：`air/T` 最大差 **`2.72e-6 K`**，`air/U` `1.88e-7 m/s`，`air/k` `1.54e-8`，`air/epsilon` `1.12e-8`，`air/nut` `2.05e-8`，四固体区 `T` 差 `≤ 6.2e-7 K` |
| **M5** | `verify/m5.sh` | 全楼边界族与 `fvOptions`：Part 1 含 `flowRateInlet/Outlet`、`pressureInletOutletVelocity`、`prghTotalPressure`、`scalarSemiImplicitSource`（`h` 与 `waterVapour`）、`fixedTemperatureConstraint`；Part 2 为 `v7_small` 真实切块（7 区 28,284 单元） | 对照 `chtMultiRegionFoam`：Part 1 `air/T` 最大差 **`9.99e-7 K`**，`air/waterVapour` `3.79e-11`；Part 2 `v7_small` `air/T` rms `4.46e-4 K`，6 个固体区 `T` 最大差 `5.4e-7 ~ 4.4e-5 K`，`waterVapour` 最大差 `2.83e-8` |
| **M6** | `verify/m6.sh` | Part 1：V5 共轭方腔开启 `adjustTimeStep` 对照 `chtMultiRegionFoam`；Part 2：`v7_full`（7 区 359,854 单元）开启 `gpuCht::exteriorSurface`、气象、太阳 ShadowMap、室内长波与 HVAC 闭环控制器对照 Py 版 | Part 1：`CoNum`/`DiNum` 与 `deltaT` 序列（`1.66666666667`×4, `2.5`×2）与 OpenFOAM **逐位一致**，`air/T` 差 `4.10e-5 K`；Part 2：`glazing_5mm/T` 差 **`1.60e-9 K`**，其余 5 固体区 `T` 差 `3.5e-6 ~ 2.1e-5 K`，`hvac.json` 盘管冷量相对差 `< 6e-7` |
| **M7** | `verify/m7.sh` | CUDA (`sm_86`) vs OMP 一致性与加速比测试（`v7_full` 35.99 万单元）+ 全楼 `r3d_mr`（45 区 **42,454,693 单元**）精确显存预算验证 | `v7_full` 上 CUDA 仅用 **0.760 GB** 显存、耗时 **1.29 s**（比 Py 版快 **7.0×**，比 6 线程 OMP 快 **6.0×**），CUDA 与 OMP 固体 `T` 差 `≤ 3.4e-13 K`；全楼 `r3d_mr` 峰值显存 **61.55 GB (57.32 GiB)**，低于 96 GB 预算 **34.45 GB** |

## 4. 全楼算例 `r3d_mr`（45 区，4,245 万单元）显存预算明细

通过 `v7_full` 实测 `engine.memGB()` 标定并逐数组核对 `libgpucht.so` 中所有 `dvec` 分配，全楼模型（`nC = 42,454,693`，其中 `air` 区 `nc = 22,723,989` 单元、内部面 `nI = 130,948,546`、区间耦合面 `nIface = 11,781,318`、边界面 `nB = 2,199,607`）的 GPU 显存占用如下：

| 模块 | 包含的设备数组（`dvec`） | 显存占用 (GB, $10^9$ B) | 显存占用 (GiB, $2^{30}$ B) |
| --- | --- | ---: | ---: |
| **1. DMesh** | 全局拓扑 (`own, nei, cfPtr, cfEnt, cellRegion`) + 几何与最小二乘向量 (`V, C, Sf, Cf, magSf, w, nonOrthDC, corrVec, lsP, lsN, dcIA, dcIB, dcB, lsB`) | 20.43 GB | 19.03 GiB |
| **2. Energy** | 45 区单矩阵 `Ldu` (`diag, src, bDiag, bAbs, bSrc, lower, upper`) + 单元场 (`T, T0, rho, rho0, cp, e, e0, kappa, alphat, Su, Sp`) + 边界/耦合面 + 梯度缓冲 | 10.57 GB | 9.84 GiB |
| **3. Flow (`air`)** | 动量/压力 `MU, Mp` + 速度/压力/密度/通量新旧场 + `Rinv, gU, gP, gRho` + 双组分 `Y` + `realizableKE` 湍流场与壁面函数数组（M7 复用临时梯度/插值缓冲省去 2.44 GB） | 26.44 GB | 24.62 GiB |
| **4. Building** | 太阳正交阴影图 `ShadowMap` + 室外/室内辐射面属性与净辐射热流数组 | 0.20 GB | 0.19 GiB |
| **常驻合计** | **网格 + 全部物理场初始化完成后常驻显存** | **57.65 GB** | **53.69 GiB** |
| **5. Solvers** | 线性求解器峰值工作区（全区能量 `PBiCGStab` 与流体压力 `GAMG + PCG` 粗层层级） | 3.90 GB | 3.63 GiB |
| **峰值总计** | **全楼 4,245 万单元运行峰值显存（目标 `< 96 GB`）** | **61.55 GB** | **57.32 GiB** |

> 相比之下，Python/CuPy 版由于 `cupy` 内存池碎片及每次算子表达式产生临时数组，在全楼网格上需 `94.1 GiB` 而触发 OOM；C++ 版 `chtMultiRegionGpuFoam` 全部采用固定预分配数组与无临时数组的 `thrust::for_each_n` / `transform_reduce` 核函数，峰值仅 **57.32 GiB（61.55 GB）**，在 96 GB 显卡（如 RTX PRO 6000 / G4 或 A100 80GB）上留有 **> 34 GB** 的充足余量。

## 5. 编译与运行命令

1. **在 NAS 容器内一键编译（OMP 或 CUDA）**：
   ```bash
   # 编译 OpenMP 后端
   ./tools/nas.sh 'BACKEND=omp bash /w/tools/build_app.sh'

   # 编译 CUDA 后端（NAS RTX 3080 为 sm_86；Colab A100 设为 sm_80，L4/G4 设为 sm_89）
   ./tools/nas.sh 'BACKEND=cuda ARCH=sm_86 bash /w/tools/build_app.sh'
   ```
2. **运行全部验证套件（M1～M7）**：
   ```bash
   ./tools/nas.sh 'for m in m1 m2 m3 m4 m5 m6 m7; do BACKEND=cuda bash /w/verify/${m}.sh || exit 1; done'
   ```
3. **直接对任意 OpenFOAM 多区算例运行 `chtMultiRegionGpuFoam`**：
   ```bash
   source /usr/lib/openfoam/openfoam2606/etc/bashrc
   export LD_LIBRARY_PATH=/w/build/cuda:$LD_LIBRARY_PATH
   /w/build/cuda/chtMultiRegionGpuFoam -case /path/to/case
   ```

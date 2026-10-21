# chtMultiRegionGpuFoam

**OpenFOAM v2606 的 GPU 版多区共轭传热求解器。** 它读写和 `chtMultiRegionFoam` 相同的算例，整个时间步都在一块 NVIDIA 显卡上算。在验证算例上和 OpenFOAM 对照，温度差在 10⁻⁵ K 量级；在 139 万单元的湍流算例上比 8 进程 OpenFOAM 快约 15 倍。写它是为了做建筑热环境和暖通空调（HVAC）模拟：把整栋楼的空气和墙、楼板、玻璃放在一起算，几千万单元装进一张大显存卡。

> **English summary.** chtMultiRegionGpuFoam is a CUDA port of OpenFOAM v2606's `chtMultiRegionFoam` for transient, buoyant conjugate heat transfer. It is a regular OpenFOAM application: it is built with `wmake`, links the OpenFOAM libraries, and reads and writes standard multi-region cases. The whole PIMPLE time step runs on one GPU: momentum, pressure, a single energy matrix for all regions, species and k-ε. This differs from plugins that offload only the linear solves. On the verification cases the results agree with OpenFOAM to about 1e-5 K. On a 1.4 M-cell turbulent conjugate cavity it is about 15× faster than chtMultiRegionFoam on 8 CPU cores. It was written for whole-building thermal and HVAC simulation, with solar shading, long-wave radiation and closed-loop HVAC controllers. It runs on a single GPU and supports a defined subset of OpenFOAM models, listed below; anything outside that subset stops with an error. The documentation is in Chinese.

## 它算什么

每个时间步，下面这些量一起算。方程和求解顺序与 chtMultiRegionFoam 相同，算法按 v2606 源码逐项核对过。

- **流体区（空气）的流动**：速度 U、压力 p_rgh / p，浮升力驱动（Boussinesq 或常密度），PIMPLE 外迭代。
- **所有区的温度**：流体区和固体区（墙、楼板、玻璃、保温层……）拼成一个能量矩阵一起解。区间界面的串联导热，包括接触热阻层，直接写进矩阵。
- **湿空气**：干空气和水蒸气两组分，混合物物性随组分变化。
- **湍流**：RNG k-ε 加壁面函数，或层流。
- **建筑物理**（本求解器加的，OpenFOAM 里没有）：
  - 逐时气象插值；
  - 按太阳位置用正交阴影图（ShadowMap）光线追踪，求外表面日照和遮挡；
  - 室外和室内长短波辐射的分配（`gpuCht::exteriorSurface` 边界、`constant/gpuChtProperties`）；
  - 室内得热和散湿（fvOptions）；
  - 调用外部 Python 控制器的 HVAC 盘管和新风闭环。

## 适合什么场景

- **建筑和房间的瞬态热环境**：自然对流、送回风、日照和围护结构蓄热一起算，例如设计日逐时模拟、空调启停过程、室内温湿度分布。作者用它做一栋多层建筑的全楼共轭传热：45 个区，约 4245 万单元，其中空气区 2272 万。
- **已经在用 chtMultiRegionFoam、但嫌 CPU 太慢的低速浮升力共轭传热问题**：方腔对流、设备和机柜散热、带固体导热的室内气流等。算例要在下面的“支持范围”内，规模在百万到几千万单元。
- **硬件**：一张显存足够的 NVIDIA 卡。1109 万单元算例实测约 1.65 GiB/百万单元，包括 AMG 等求解器结构，第一步建好后不再增长。按此估算，20 GB 卡约放 1000 万单元，96 GB 卡（RTX PRO 6000、Colab G4 等）约放 4000～5000 万单元。

不适合的情况：

- 可压缩高速流、燃烧、多相、OpenFOAM 的辐射模型（fvDOM 等）、稳态求解（chtMultiRegionSimpleFoam）：都没有做。
- 多卡和 MPI 并行（`-parallel`）：只能单卡串行。
- 几万单元以下的小算例：GPU 核函数的启动开销占大头，往往不如 CPU 快。

## 优势

1. **它就是一个 OpenFOAM 求解器。**
   - 用 `wmake` 编译，链接 OpenFOAM 库。
   - 直接用 chtMultiRegionFoam 的算例：`regionProperties`、各区 `polyMesh`、`0/`、`system/`、fvSchemes、fvSolution 都不用改。
   - 结果写成标准时间目录，ParaView、`postProcess` 和重启照常可用。
   - functionObjects（probes、fieldMinMax、volFieldValue、sample……）和 `solverInfo` 每步拿到的都是当前场。
2. **整个时间步在 GPU 上算。** 网格、场和矩阵常驻显存，动量、压力、能量、组分和湍流方程的组装与求解都在显卡上完成，只在写出或 functionObjects 需要时把场拷回主机。只把线性方程交给 GPU 的插件，每次求解都要转换矩阵格式、在主机和显卡之间搬数据，其余方程也还在 CPU 上算。
3. **结果和 OpenFOAM 一致。** 算法按 v2606 源码逐项对过，例如松弛查哪个字典项、Final 迭代、多变量对流限制器、壁面函数、adjustTimeStep 的步长增长公式。七级验证算例与 chtMultiRegionFoam 对照，温度最大差为 10⁻¹¹～10⁻⁵ K，自适应步长序列逐位相同（见“精度”一节）。
4. **快。** 139 万单元的三维湍流共轭方腔，RTX 3080 上每步 0.28 s；同一台机器上 OpenFOAM 用 8 进程，每步 4.4 s。
5. **省显存。** 几何量直接取自 OpenFOAM 的对象，不在显卡上重算最小二乘矩阵。所有数组都预先分配、反复复用，计算中不产生临时数组。
6. **结果可复现。** 按单元求和一律是“每个单元收集自己的面”，不用原子加，归约次序也固定。所以同一个程序在同一张卡上重复运行，结果逐位相同。
7. **不会悄悄算错。** 不支持的边界条件、离散格式或求解器名字一律报错退出，不会换成别的做法接着算。
8. **也能编成 CPU 版。** 同一份源码用 `-omp` 编成 OpenMP 多线程版，没有 NVIDIA 卡也能运行和调试。

### 和现有 OpenFOAM GPU 方案的区别

| 方案 | 做法 | 情况 |
| --- | --- | --- |
| OGL（Ginkgo）、openfoam-amgx（AmgX）、petsc4Foam（PETSc） | 只把压力等线性方程交给 GPU，组装和其余方程仍在 CPU 上 | 作者 2026-10 在 RTX PRO 6000（96 GB）上测 210 万单元的 icoFoam 方腔，三个都比 24 进程 CPU GAMG（17.9 s）慢，最好的是 petsc4Foam 8 进程（33 s） |
| OpenFOAM 官方 GPU 分支（`feature-gpu`，NVHPC stdpar） | 用标准并行算法把循环放到 GPU 上 | 需要 HMM 或 ATS 统一内存；在没有 HMM 的机器（如 Colab G4）上，一运行就非法访存 |
| RapidCFD | 整个求解器搬上 GPU | 基于 OpenFOAM 2.3，没有多区共轭传热要用的耦合温度边界 |
| **本项目** | 整个时间步在 GPU 上，单卡 | 只覆盖 chtMultiRegionFoam 的一部分功能（见“支持范围”） |

## 性能

测试机是 NVIDIA RTX 3080（20 GB）和 Intel i9-13900HX。数据是 2026-10-20 的实测，每步做 3 次外迭代，两边线性求解的容差相同。

| 算例 | 单元数 | chtMultiRegionGpuFoam | OpenFOAM chtMultiRegionFoam |
| --- | ---: | ---: | ---: |
| 三维湍流共轭方腔（`verify/bench_case.sh OUT 2 60`），RNG k-ε，5 步 | 139 万（空气 86 万） | **1.42 s** | 21.8 s（8 进程 scotch，容器限 10 核） |
| 同上加密（`bench_case.sh OUT 4 120`），3 步 | 1109 万（空气 691 万） | **6.37 s**，显存 18.3 GB | — |

- 1109 万单元算例每次外迭代约 0.7 s。其中两次压力修正约 0.31 s，动量 0.13 s，湍流 0.12 s，能量 0.08 s。
- 在这个规模上瓶颈是显存带宽。小算例则主要是核函数启动的开销。
- OpenFOAM 一栏用的不是这颗 CPU 的最快设置，只说明量级。
- 测速算例是封闭方腔，只用来测速，物理上不收敛到稳定解，见[开发记录](docs/开发记录.md) 6.5。

## 精度（与 OpenFOAM v2606 chtMultiRegionFoam 对照）

同一算例分别用两个求解器算，逐场比较最大差。用的是 CUDA 版，在 RTX 3080 上跑，数据为 2026-10-20 的结果。

| 编号 | 算例 | 最大差 |
| --- | --- | --- |
| M0 | 库自检（不用 OpenFOAM）：双材料棒解析解（另有界面带面换热的一项）、48³ 立方体 AMG、ShadowMap 阴影面积 | 与解析解差 2.8e-14 K；阴影面积 4.000 m²（解析值 4.000） |
| M1 | 非正交斜六面体双固体区、锯齿界面、`externalWallHeatFluxTemperature`（含辐射） | 固体 T 1.1e-11 K |
| M2 | 共轭方腔（空气 Boussinesq 加 4 块固体壁），热面阶跃 +5 K，算到 60 s | 空气 T 4.2e-5 K，U 3.1e-7 m/s。作为参照，OpenFOAM 自己 15 次和 60 次外迭代的差是 5.3e-5 K。线性求解收紧到 relTol 0 时为 4.4e-6 K |
| M3 | 湿空气两组分、混合物物性变化、多变量 `limitedLinear`，算到 60 s | 水蒸气参考焓 Href 为 0 的两个变体（水蒸气 Cp 1006 和 1860）：空气 T 4.2e-5 K，U 2e-7 m/s，水蒸气质量分数 1.7e-7（场值约 0.007）。Href 2501000 的变体差别偏大，见“已知问题” |
| M4 | RNG k-ε 湍流共轭方腔加四种壁面函数，算到 2 s | 空气 T 2.9e-5 K，U 2.2e-6 m/s |
| M5 | 全楼用到的边界和 fvOptions：`flowRateInlet/Outlet`、`pressureInletOutletVelocity`、`prghTotalPressure`、`scalarSemiImplicitSource`、`fixedTemperatureConstraint`，算到 60 s | 空气 T 6.4e-5 K |
| M6 | adjustTimeStep（maxCo / maxDi） | 库朗数、扩散数和 deltaT 序列与 OpenFOAM 逐位相同 |
| — | 139 万单元湍流方腔，算到 0.25 s | U 5.5e-5 m/s（场量级 0.0093 m/s）。T 的最大差 1.2e-3 K 出现在一个三面交角单元，均方根 2.3e-5 K。固体 T 3e-8 K |

差别来自两方面：一是多色 Gauss-Seidel / DILU 按颜色次序扫描单元，而 OpenFOAM 按编号次序扫描；二是线性求解在容差内停止。把容差收紧，差别随之变小（见 M2）。

## 安装

先 `source` OpenFOAM v2606 的 `etc/bashrc`，然后在本目录下运行：

```bash
./Allwmake                 # CUDA 版，按本机显卡编译（nvcc 要在 PATH 里）
ARCH=sm_120 ./Allwmake     # 给别的显卡编译：A100 sm_80，RTX 30 系 sm_86，L4 / RTX 40 系 sm_89，
                           # H100 sm_90，RTX PRO 6000 / Colab G4 sm_120；ARCH=all 一次编进全部
./Allwmake -omp            # CPU 版（OpenMP 多线程，不需要 CUDA）
```

- `libgpucht.so` 装到 `$FOAM_USER_LIBBIN`，`chtMultiRegionGpuFoam` 装到 `$FOAM_USER_APPBIN`，两者都在 OpenFOAM 的路径上。`./Allwclean` 用来清理。
- 开发和测试用的环境：OpenFOAM v2606（openfoam.com）、g++ 13、CUDA 12.8。CPU 版要用的 Thrust/CUB 头文件（NVIDIA CCCL 3.0.3）随仓库放在 `third_party/`。
- 仓库里只有源码，需要自己编译。

## 运行

算例就是标准的 chtMultiRegionFoam 多区算例，不用改：

```bash
chtMultiRegionGpuFoam -case <算例>          # 单卡；用 CUDA_VISIBLE_DEVICES 选卡
chtMultiRegionGpuFoam -noFunctionObjects   # 其余 OpenFOAM 通用选项照常可用
```

结果写成标准时间目录，paraFoam、foamToVTK、postProcess 都能直接读。重启时从时间目录接着算（`startFrom latestTime`）。

## 和 OpenFOAM 一起用时的行为

- **functionObjects**：场常驻显存。controlDict 里有 functionObjects 时，每个时间步结束都把场拷回 OpenFOAM，OpenFOAM 在下一步开头执行它们。probes、fieldMinMax、volFieldValue、surfaceFieldValue、sample、fieldAverage 等按场工作的都能用。没有 functionObjects 时，只在写出时刻拷回。
- **残差**：各方程的初始残差、最终残差和迭代数写进 `mesh.data()`，和 OpenFOAM 自己的求解器一样，`solverInfo` 能记录。能量方程是全区一个矩阵，所以每个区报的 `h` 残差相同。
- **日志**：
  - `Solving for ...` 行的格式和 OpenFOAM 相同。
  - 每步开头打印各流体区的库朗数和各固体区的扩散数。
  - `Min/max T`、`Min/max rho` 和 OpenFOAM 一样含边界面值。
  - 第一步结束时打印显存占用。
  - 结束时打印各阶段（动量、组分、能量、压力、湍流、拷回、写出、functionObjects）的墙钟时间。
- **nOuterCorrectors 为 1 时不能松弛 p_rgh**：和 OpenFOAM 一样会报错。
- **运行中用不了的 functionObjects**：需要 OpenFOAM 热物性或湍流模型对象的那些（`wallHeatFlux`、`yPlus`、`Qdot` 等）用不了，因为本求解器不构造这些对象，会报找不到。可以在写出后事后计算，例如 `chtMultiRegionFoam -postProcess -func wallHeatFlux`。

## 支持范围

下表按代码核对。不在表里的写法一律报错退出，不会去猜。

| 类别 | 支持 |
| --- | --- |
| 区 | `regionProperties` 的 fluid / solid；不支持流体-流体界面；只能串行（不支持 `-parallel`、cyclic 和 processor 补丁） |
| 时间 | 瞬态，`ddtSchemes` 只能用 Euler；`adjustTimeStep`（maxCo / maxDi / maxDeltaT）；没有稳态版 |
| 流体热物性 | `heRhoThermo`，`const` 输运，`hConst`，`sensibleEnthalpy`；`pureMixture` 或 `multiComponentMixture`；状态方程 `Boussinesq` / `rhoConst`（多组分 Boussinesq 要加 `libs (boussinesqMixtureThermo)`） |
| 湍流 | `laminar`（Stokes）、`RAS RNGkEpsilon`；壁面函数 `kqRWallFunction`、`epsilonWallFunction`（无 lowReCorrection）、`nutkWallFunction`、`alphatJayatillekeWallFunction` / `alphatWallFunction` |
| 辐射、燃烧 | `none`。建筑用的太阳和长波辐射在 `gpuCht::exteriorSurface` 和 `constant/gpuChtProperties` 里 |
| 离散格式 | grad：`leastSquares`、`cellLimited leastSquares k`；div：`[bounded] Gauss upwind / linearUpwind grad(x) / linear / limitedLinear k`（`div(phi,Yi_h)` 不能加 bounded，与 v2606 相同）；laplacian：`Gauss linear corrected / uncorrected / limited [corrected] ψ` |
| 线性求解器 | 名字照 OpenFOAM 的写，内部映射到本库的实现：<br>• p_rgh 一律用 AMG 预条件 PCG（日志里显示 `AMGPCG`）。凡是 AMG 预条件，V 循环都用单精度，外层 Krylov 迭代用双精度。<br>• `GAMG`（非压力方程）→ AMG 预条件 BiCGStab；`PCG`/`PPCG`/`PPCR` → PCG；`PBiCG`/`PBiCGStab` → BiCGStab。<br>• preconditioner 认 `GAMG`、`DIC`/`FDIC`/`DILU`（多色 DILU）、`diagonal`、`none`。<br>• `smoothSolver` 认 `GaussSeidel`/`nonBlockingGaussSeidel`/`symGaussSeidel`（多色 Gauss-Seidel）、`DIC`/`DILU`，以及 `nSweeps`。<br>• 收敛判据（tolerance / relTol / maxIter / minIter）和残差归一化与 OpenFOAM 相同。 |
| PIMPLE | nOuterCorrectors、nCorrectors、nNonOrthogonalCorrectors、momentumPredictor、frozenFlow、outerCorrectorResidualControl（relTol 只能为 0）、pRefCell / pRefPoint；不支持 consistent（SIMPLEC） |
| T 边界 | fixedValue / uniformFixedValue / timeVaryingMappedFixedValue，zeroGradient / symmetry(Plane)，fixedGradient，mixed，inletOutlet / uniformInletOutlet，externalWallHeatFluxTemperature（power / flux / coefficient，无 qr，无 thermalInertia），区间界面 turbulentTemperatureCoupledBaffleMixed / RadCoupledMixed（qr none，可带 thicknessLayers / kappaLayers），gpuCht::exteriorSurface |
| U 边界 | noSlip、fixedValue、movingWallVelocity、uniformFixedValue、flowRateInletVelocity / flowRateOutletVelocity（无 extrapolateProfile）、zeroGradient、inletOutlet、pressureInletOutletVelocity（无 tangentialVelocity）、slip / symmetry(Plane)；区间界面上只能是壁面 |
| p_rgh 边界 | fixedFluxPressure、zeroGradient / symmetry、fixedValue / uniformFixedValue、prghPressure、prghTotalPressure、totalPressure（只认默认的 U / phi / rho，psi none） |
| fvOptions | scalarSemiImplicitSource（h 或某个组分；all / cellZone；absolute / specific）、fixedTemperatureConstraint（uniform，cellZone） |

## 原理

- **两层结构**：
  - `app/chtMultiRegionGpuFoam` 是 OpenFOAM 程序。它读算例、建各区网格和边界条件、拼出全局网格，再把几何量、物性和边界系数交给计算库，写出时把场取回来。
  - `lib/` 是计算库 `libgpucht.so`。它不含 OpenFOAM 头文件，只用标准 C++17 和 Thrust。同一份源码可以用 nvcc 编成 CUDA 版，也可以用 g++ -fopenmp 编成 CPU 版。
- **几何量全部取自 OpenFOAM**：`weights()`、`nonOrthDeltaCoeffs()`、`nonOrthCorrectionVectors()`、最小二乘向量、`patch.deltaCoeffs()` 直接上传。所以离散和 OpenFOAM 一致，也省去在显卡上重算这些量所需的显存。
- **全区一个能量矩阵**：所有流体区和固体区拼成一个 LDU 矩阵，以 T 为未知量解 he 方程。界面的串联导热导纳直接作为非对角项进矩阵，松弛和 setValues 按 OpenFOAM 对耦合边界系数的处理方式计入。
- **确定性的 GPU 核函数**：每个单元从“单元→面”邻接表收集自己面上的通量，没有原子操作。面上的量每个面只算一次，存进面数组。
- **线性求解器**：
  - 压力方程用聚合 AMG（边优先级配对，最细层 CSR，粗层 CSR，单精度 V 循环）预条件的柔性 PCG。
  - 其余方程用多色 DILU / Gauss-Seidel。单元着色用 Jones-Plassmann，结果确定，整个求解在颜色行序里做。
  - 工作向量常驻复用，求解过程中不申请显存。
- **建筑模块**：气象插值、太阳位置、正交阴影图光线追踪、室外和室内长短波辐射分配都在 `lib/building.cpp`。HVAC 控制器经内嵌的 libpython3 调用，零拷贝读写设备场的主机副本。

## 目录

```
Allwmake / Allwclean         安装与清理
app/chtMultiRegionGpuFoam/   OpenFOAM 主程序（wmake）：主循环、多区网格拼装、能量和流体的边界条件、fvOptions、HVAC 接口
lib/                         计算库 libgpucht.so（make BACKEND=cuda|omp）
  par.h vec.h                后端层（Thrust）、向量和张量工具
  api.h                      主程序与计算库的接口（只用标准 C++ 类型）
  mesh.* ldu.* linsolve.*    设备网格、LDU 矩阵、线性求解器（PCG、BiCGStab、AMG、多色 DILU / GS）
  energy.* flow.*            全区能量方程；流体区 PIMPLE、组分、RNG k-ε
  building.* engine.cpp      建筑物理；各模块调度
test/test_lib.cpp            不依赖 OpenFOAM 的库自检（M0）
verify/                      与 OpenFOAM 对照的验证、测速和诊断脚本
tools/                       编译脚本 build_app.sh；作者用的远程容器脚本 nas.sh
third_party/                 NVIDIA CCCL 3.0.3（CPU 版用的 Thrust / CUB 头文件）
docs/开发记录.md              按时间写的开发记录：每次改动、测速、验证结果和遇到的问题
```

## 开发与验证

- `test/test_lib.cpp`（M0）不依赖 OpenFOAM，能直接编译运行：`make -C lib BACKEND=cuda ARCH=sm_86 test && build/cuda/test_lib`。CPU 版用 `BACKEND=omp`，要先按 `Allwmake` 里的做法解开 `third_party/cccl-3.0.3.tar.gz`。
- `verify/` 里的 M1～M8、测速和诊断脚本是按作者的环境写的。它们在容器里运行，本仓库挂在 `/w`；还要用到作者另一个未公开的 Python 版求解器（挂在 `/py`）里的算例生成脚本和参考算例，所以在别的机器上不能直接运行，只能作为验证做法的参考。逐场比较用的 `verify/compare_of.py` 也是调用 Python 版里的同名脚本。不依赖这些、可以单独使用的有 `verify/ordering.py`（网格编号局部性）和 `verify/cmp_logs.py`（比较两次验证日志里的误差）。
- `tools/nas.sh` 把源码同步到作者的远程 GPU 机器，并在隔离容器里执行命令。主机和目录从不入库的 `tools/nas.env` 读取，格式见 `tools/nas.env.example`。

## 现状和已知问题

- 全楼算例（4245 万单元，45 区）2026-10-20 在 Colab G4（96 GB）上试跑：第一步后显存 71.1 GiB，之后的外迭代不再增长，每次外迭代约 2.0 s；但 10 s 和 1 s 步长都在第一步发散，原因是网格里一处 78～86° 的严重非正交过渡区和大量毫米级薄片单元，要先改网格，见[开发记录](docs/开发记录.md) 6.7。
- 建筑模块的输入格式（`constant/gpuChtProperties` 的 weather / solar / interiorRadiation / hvac 子字典，以及 `gpuCht::exteriorSurface` 的参数）目前只能看代码（`app/chtMultiRegionGpuFoam/energyCoupling.C`、`fluidCoupling.C`），还没有单独的文档。HVAC 控制器本身（Python）不在本仓库。
- 多色 DILU 比 OpenFOAM 按编号次序的 DILU 多用 2～4 倍迭代，这是多色方法的通病。
- 网格质量差（几毫米厚的薄片单元、20:1 的尺寸突变）时，大步长会发散，OpenFOAM 也一样。详见[开发记录](docs/开发记录.md) 6.4。
- 湿空气算例中，水蒸气参考焓 Href 2501000 的变体在 t = 10 s 时与 OpenFOAM 差 0.063 K，到 60 s 降到 5e-4 K，原因还没查。
- 不支持多卡和 `-parallel`。

## 许可证

[GPL-3.0-or-later](LICENSE)。本程序链接 OpenFOAM 库，算法依照 OpenFOAM v2606（GPL-3.0-or-later）的 chtMultiRegionFoam 源码实现，所以采用同一许可证。`third_party/cccl-3.0.3.tar.gz` 是 NVIDIA CCCL，许可证为 Apache-2.0 WITH LLVM-exception，见压缩包内的 LICENSE。

This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the OpenFOAM software via www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks.

# CLI 货架合成数据生成

CLI 与网页共用 Three.js 场景、拍摄路径、SKU 陈列和导出代码。可以不启动网页服务，直接生成完整 RGB 图片、扫过区域的正交 GT 及配套 JSON；不需要 Python、模型权重或外部数据集。

## 环境与入口

需要 Node.js 22.18 或更新版本，以及本机安装的 Google Chrome / Chromium。首次使用，在项目根目录执行：

```sh
cd /Users/haoyu/Documents/Projects/tinyStitch
npm ci
npm run generate -- --help
```

CLI 会临时启动仅监听 `127.0.0.1` 的渲染服务，用独立的无头浏览器渲染图片，完成后关闭服务与浏览器。无需提前运行 `npm start` 或 `npm run build`，也不会操作已经打开的浏览器标签页。

## 单组生成

```sh
npm run generate -- \
  --out ./output/shelf-42 \
  --seed 42 \
  --length 6 \
  --frames 12 \
  --bay-mode auto \
  --bay-width 1 \
  --layers 3,4,5 \
  --capture handheld \
  --jitter 15 \
  --empty 0.25 \
  --facings 2:5 \
  --depth 3 \
  --zip
```

此示例生成总长 6 模拟米、6 个等宽子货架，层板循环为 3 / 4 / 5 / 3 / 4 / 5 层，输出 12 张 720×960 JPEG，以及一张正面正交 GT 和覆盖掩码。默认采用手持仿真，偏航、俯仰各在 ±15° 范围连续随机扰动；另有 25% 空位概率、同款连续 2–5 列，每列纵深最多容纳 3 件，并同时保存 ZIP。

`--out` 必须指向一个不存在的新目录，已有目录不会覆盖。

## 批量生成与随机范围

```sh
npm run generate -- \
  --out ./output/batch-100 \
  --seed 100 \
  --count 10 \
  --length 3:8 \
  --frames 8:16 \
  --bay-mode auto \
  --layers 3,4,5 \
  --capture handheld \
  --empty 0.18
```

生成 seed 100–109 的 10 个场景。每个场景按其种子确定长度和图片张数，并根据实际长度计算子货架数量，分别写入 `seed-100/` 至 `seed-109/`。所有随机量均由种子驱动；修改空位设置不会改变拍摄路径。

## 子货架自动伸缩

默认 `bayMode=auto`、目标段宽 `bayWidth=1` 模拟米。数量为 `ceil(总长度 / 目标段宽)`，再将总长度等宽分配；例如 3 米生成 3 段，6 米生成 6 段，6.2 米生成 7 段，每段约 0.886 米。最多 24 段，每段层板可独立设置，未配置的段循环使用 `bayLayers` 模板。

如果要固定为三个子货架，可以使用：

```sh
npm run generate -- --out ./output/manual-3 --length 6 --layers 3,4,5
```

单独提供 `--layers` 会切换为手动数量，保持旧命令的含义。若要将该列表作为自动模式的层板模板，同时加 `--bay-mode auto`；参数书写顺序不影响结果。旧 JSON 若只有 `bayLayers`、没有 `bayMode`，也按手动数量处理。

## 复用网页配置

网页点击“下载 CLI 生成配置”，得到 `shelf-config.json`：

```sh
npm run generate -- \
  --config ~/Downloads/shelf-config.json \
  --out ./output/from-web \
  --zip
```

完整序列 ZIP 中的 `generation/config.json` 也可以直接复用。CLI 自己生成的配置同样可以再次使用：

```sh
npm run generate -- \
  --config ./output/shelf-42/generation/config.json \
  --out ./output/shelf-42-replay
```

命令行显式参数优先于配置，配置优先于默认值。例如，保留陈列与视角设置，只改变种子和图片张数：

```sh
npm run generate -- \
  --config ./output/shelf-42/generation/config.json \
  --out ./output/seed-43 \
  --seed 43 --frames 16
```

配置格式：

```json
{
  "schemaVersion": 1,
  "seed": 42,
  "options": {
    "shelfLength": 6,
    "frames": 12,
    "bayMode": "auto",
    "bayWidth": 1,
    "bayLayers": [3, 4, 5],
    "view": { "mode": "handheld", "yaw": 0, "pitch": 0, "jitter": 15 },
    "stock": { "emptyRate": 0.25, "facingsMin": 2, "facingsMax": 5, "depthCopies": 3 }
  }
}
```

随机范围在 JSON 中写成 `{ "min": 3, "max": 8 }`，可用于 `shelfLength` 和 `frames`。

## 参数表

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--out DIR` | 必填 | 新的输出目录，不覆盖已有目录 |
| `--config FILE` | 无 | 网页或 CLI 导出的配置 JSON |
| `--seed N` | 42 | 0–4294967295 的整数 |
| `--count N` | 1 | 批量场景数 1–1000，种子逐个递增，不能溢出 |
| `--length N` / `MIN:MAX` | 2:8 | 总长度 1–12 模拟米 |
| `--frames N` / `MIN:MAX` | 9 | 图片数量 2–48 的整数 |
| `--bay-mode MODE` | auto | `auto` 按长度计算数量；`manual` 按层板列表长度确定数量 |
| `--bay-width N` | 1 | 自动模式的目标段宽，0.5–3 模拟米；最终等宽分配 |
| `--layers N,N,…` | 3,4,5 | 最多 24 项，每项 1–8 层，包含底层；单独指定切换手动，配合 `--bay-mode auto` 则循环作为层板模板 |
| `--capture MODE` | handheld | `handheld` 仿真拍摄或 `fixed` 固定路线 |
| `--yaw N` | 0 | 偏航角 −25° 至 25° |
| `--pitch N` | 0 | 俯仰角 −20° 至 20°，正值向下看 |
| `--jitter N` | 15 | 偏航、俯仰各自的扰动最大幅度 0–15°；默认中心为 0°，随机范围均为 −15° 至 15° |
| `--empty N` | 0.18 | 空货位概率 0–1；1 表示全部缺货 |
| `--facings N` / `MIN:MAX` | 2:5 | 同款连续列数 1–12，子货架末端可能截短 |
| `--depth N` | 3 | 每列纵深容量 1–4；总容量还包含净空允许的上下堆叠层数，实际余货数随机为 0–总容量 |
| `--zip` | 关闭 | 同时保存每组 `sequence.zip` |
| `--browser FILE` | 自动定位 Chrome | 指定 Chrome / Chromium 可执行文件 |
| `--help` | — | 显示帮助 |

固定路线可关闭角度扰动以获得完全平稳的拍摄：

```sh
npm run generate -- --out ./output/stable --capture fixed --jitter 0
```

仿真模式采用连续平滑的姿态变化：相机水平摆动不超过约 2 cm，高度不超过 2.5 cm，距离不超过 3.5 cm，轻微滚转不超过 0.7°；偏航、俯仰由 `jitter` 控制。另有轻微行进速度变化。仿真模式下 `--jitter 0` 仅关闭角度摆动，位置与采样时间仍有变化。

## 商品陈列规则

- 每层选取小规模 SKU 组合，按同款连续多列成块陈列。
- 按商品实际宽度依次排布，横向只留约 4 毫米间隙，不再使用固定等距列中心。货架两侧预留约 2.5 厘米避开立柱，末端不足以放下一件商品的余量保留。缺货时保留该商品原有宽度的空货位，不把两侧商品收拢来填空；修改 `empty` 不改变计划货位的位置、SKU 或数量。
- 同 SKU 使用相同包装贴图、颜色和物理尺寸；不会给每件复制商品随机换包装。
- 商品混合盒装（`box`）、带肩和瓶盖的瓶装（`bottle`）、金属沿罐装（`can`）与带盖广口罐（`jar`）。瓶罐标签贴合圆柱表面；RGB 与正交 GT 使用同一外形。
- SKU 宽度更分散：盒装为货位宽度的 40%–88%，圆形容器还受纵深容量限制，宽、深相等；所有外形的最大宽度保持原来的货位宽度 88% 上限。瓶身、瓶肩、瓶盖和罐盖均包含在导出尺寸内，不挤出货位。
- 每列可以纵深摆放多件，剩余数量不同，部分余货会向后缩进。
- 层间净空允许时，盒装、罐装和广口罐可上下堆叠；瓶装因瓶肩、窄盖保持单层。每个纵深位置先从底部向上摆放，再开始另一摞，不生成悬空商品，也不穿过上层板。不同子货架分别按自己的层高计算容量。
- 同时模拟整块 SKU 缺货与零散空货位。`empty` 是概率，实际空位数随种子变化，不要求精确等于输入百分比。
- 层高独立设置，商品仅选择能够放入该层的 SKU，避免穿过层板；末端陈列块可少于最小列数。
- 货架两面都生成陈列；统计中的货位和商品数包含两面。拍摄序列只从选定一面拍摄。

这是规则驱动的合成场景，不等同于真实门店分布或照片级渲染。

## 输出内容

单组目录：

```text
shelf-42/
  rgb/0000.jpg … 0011.jpg
  gt/
    orthographic.png
    coverage.png
    metadata.json
    camera_poses.json
    camera_poses.csv
  intrinsics.json
  timestamps.json
  capture.json
  generation/
    config.json
    scene.json
  manifest.json
  sequence.zip                # 使用 --zip 时
```

批量目录外层保存 `manifest.json`，其下每个 `seed-N/` 包含该组数据。

| 文件 | 内容 |
| --- | --- |
| `rgb/` | 按拍摄顺序编号的 720×960 JPEG |
| `gt/orthographic.png` | 扫过区域的正面正交 GT，未覆盖区域透明 |
| `gt/coverage.png` | 与正交 GT 同尺寸的二值覆盖掩码，白色覆盖、黑色未覆盖 |
| `gt/metadata.json` | 参考平面、裁剪范围、像素比例、局部坐标到像素的变换，以及逐帧覆盖多边形 |
| `gt/camera_poses.json` | 每帧相机真实世界 6D 位姿，附四元数、双向变换矩阵和完整坐标约定 |
| `gt/camera_poses.csv` | 同一位姿的平面表格，含 XYZ、旋转角（弧度和度）及四元数 |
| `intrinsics.json` | 图像宽高和内参矩阵 K，垂直 FOV 54° |
| `timestamps.json` | 秒单位采样时间，与 RGB 顺序对应；fixed 间隔 0.5 秒，handheld 间隔 0.45–0.55 秒 |
| `capture.json` | 渲染器、坐标约定与数据字段说明 |
| `generation/config.json` | 可直接复用的种子和完整生成配置 |
| `generation/scene.json` | 货架、子货架、层板高度、逐帧位姿、SKU、货位和商品数据 |
| `manifest.json` | 是否全部完成、种子、各组图片数量、子货架数量、GT 尺寸、实际空位数与货位数 |
| `sequence.zip` | 与网页 ZIP 一致的图片及配套 JSON，不包含批量 manifest |

`scene.json` 的关键字段：

- `shelf.bays[]`：子货架位置、宽度、层板数量和各层高度。
- `shelf.merchandising.catalog[]`：SKU ID、品类名、外形 `shape`、是否允许堆叠 `stackable`、包装颜色和宽高深（整体包围尺寸）。
- `shelf.merchandising.slots[]`：所有计划货位，包括空位；含 `skuId`、`x` 横向中心、`width` 占位宽度、`stock` 实际数量、`depthCapacity` 纵深容量、`stackCapacity` 堆叠层数上限、`capacity = depthCapacity × stackCapacity` 总容量，以及 `bay`、`layer`、`col` 和 `side`。`stock === 0` 表示空货位，其 `x/width` 仍保留。
- `shelf.merchandising.products[]`：实际存在的每一件商品，含 `slotId`、`skuId`、`shape`、`depthIndex` 纵深位置、`stackIndex` 堆叠层号、位置和尺寸。空货位不会生成虚假的商品。
- `poses[]`：逐帧 `x/y/z`、`lookX/lookY/lookZ`、`yawDegrees/pitchDegrees/rollDegrees` 和 `timestamp`。

货架和商品位置单位为模拟米。子货架与商品使用货架局部坐标，`bay/layer/col/depthIndex/stackIndex` 从 0 开始，`stackIndex=0` 表示直接放在层板上；`side` 为 −1 或 +1。位姿使用 Three.js 世界坐标，Y 向上，相机朝局部 −Z，先 `lookAt` 再绕相机局部 +Z 应用滚转。

同一种子和参数可以复现场景、库存、路径及时间戳。不同浏览器版本或 GPU 的最终像素可能存在细微差别。

## 相机真实世界 6D 位姿 GT

每次生成都自动输出 `gt/camera_poses.json` 和 `gt/camera_poses.csv`，无需额外参数。网页可单独点击“下载相机位姿 GT”获取 JSON，完整 ZIP 包含 JSON 和 CSV；预览拍摄序列时，下方显示当前帧的位置与旋转角。

GT 直接在每张 RGB 渲染完成时读取该相机的 `matrixWorld`，包括实际位置、偏航、俯仰和滚转扰动；使用整段统一的场景世界坐标，不以首帧归零，不从图片估计，也不把输入的货架相对视角直接当作世界旋转。这是合成场景中的真实位姿，不是地理坐标或真实门店的测量坐标。

JSON 顶层记录约定，`frames[]` 与 RGB 顺序、时间戳一一对应：

| 字段 | 含义 |
| --- | --- |
| `frame` / `image` / `timestamp_s` | 从 0 开始的帧号、对应 RGB 路径、秒单位采样时间 |
| `pose6d_m_rad` | `[x, y, z, rx, ry, rz]`，前三项为米，后三项为弧度 |
| `position_m` | 相机光心在世界坐标中的 XYZ |
| `euler_xyz_rad` / `euler_xyz_deg` | 相机到世界的内禀 Euler XYZ 旋转角，分别使用弧度、度 |
| `quaternion_xyzw` | 相机到世界的单位四元数，按 `[qx, qy, qz, qw]` 排列 |
| `T_c2w` / `T_w2c` | 原生 Three.js 相机坐标到世界、世界到相机的 4×4 变换 |
| `opencv.T_c2w` / `opencv.T_w2c` | 同一世界坐标下，OpenCV 相机坐标约定的双向变换 |

世界系为右手系，原点在店铺地面中心；X 沿店铺宽度，Y 向上，Z 沿店铺深度。一个场景单位对应一米。原生相机系 X 向右、Y 向上、−Z 为观看方向。

`rx/ry/rz` 满足 `R_c2w = Rx(rx) × Ry(ry) × Rz(rz)`，为内禀 XYZ 顺序，**不同于**生成配置中相对货架法线的 `yaw/pitch/roll`。Euler 角存在等价表达、分支跳变及万向锁；进行几何计算时优先使用四元数或矩阵，不能把角度跳变直接解释成相机突然转动。

矩阵保存为嵌套的按行数组，对列向量左乘：

```text
p_world  = T_c2w × p_camera
p_camera = T_w2c × p_world
```

OpenCV 相机系为 X 向右、Y 向下、Z 向前，世界系保持不变：

```text
T_c2w_cv = T_c2w × diag(1, -1, -1, 1)
```

将世界点通过 `opencv.T_w2c` 转到相机系，再使用 `intrinsics.json` 中的 K 投影即可对应原 RGB。K 使用从 0 开始的像素中心索引；若从渲染器 NDC 转换，则像素坐标为 `u = (ndc_x + 1) × 720 / 2 − 0.5`、`v = (1 − ndc_y) × 960 / 2 − 0.5`。

CSV 的列名明确写出单位：

```text
frame,image,timestamp_s,x_m,y_m,z_m,rx_rad,ry_rad,rz_rad,rx_deg,ry_deg,rz_deg,qx,qy,qz,qw
```

同一份数据无需复制到 `scene.json`，其中原有 `poses[]` 继续保存生成路径与控制视角；相机真正旋转与变换请读取上述 GT 文件。

## 正交 GT 的范围与坐标

每组必生成一张正面正交图，使用与 RGB 完全相同的货架、商品和包装。它直接从场景渲染，不由 RGB 拼接生成，也不是俯视图。商品保留原始三维深度；层板和立柱在正交视角下保持水平和垂直。

覆盖定义为：每个实际相机视锥与货架外侧正面参考平面（局部 `z = side × depth / 2`）相交，并裁到货架宽、高范围，再取全部视角的并集。GT 按并集外接矩形裁出，矩形内部未被扫到的部分也保留透明；不是只取中间帧，也不自动补齐整面货架。空货位仍属于可拍摄区域，不应与未覆盖区域混淆。

该掩码表示参考平面的几何视野覆盖，不是逐商品表面可见性或遮挡掩码。斜拍中商品侧面和遮挡关系与正面正交图可能不同。

输出比例为 `960 / 货架高度` 像素 / 模拟米，当前高度 2 米即 480 像素 / 米，宽高随覆盖范围变化。为保证 X/Y 同比例，边界向外对称扩展不足一个像素，再由掩码标记真实覆盖。

`gt/metadata.json` 中 `shelfXYToImage` 为 3×3 矩阵：

```text
[u, v, 1]ᵀ = shelfXYToImage × [shelf_local_x, shelf_local_y, 1]ᵀ
```

图像边缘坐标起点为 `(0,0)`，像素中心为 `(列+0.5, 行+0.5)`；图像向右对应 `side × 货架局部 +X`，向下对应局部 `−Y`，与拍摄面正视方向一致。`shelfToWorldColumnMajor` 保存 Three.js 列主序 4×4 世界变换。若要映射商品位置，可使用 `scene.json` 中实际商品中心与尺寸的局部 X/Y；该正交映射无需商品深度参与缩放。

网页在“拍摄序列 / 正交 GT / 覆盖掩码”之间切换，可单独下载当前图片；完整 ZIP 和 CLI 输出始终包含上述三个 GT 文件。

## 浏览器路径与错误处理

默认寻找本机 Chrome。若使用 Chromium 或自定义安装位置：

```sh
npm run generate -- \
  --out ./output/custom-chrome \
  --browser "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
```

也可以通过 `CHROME_PATH` 指定。`--browser` 优先于该环境变量。

参数错误在创建输出前拒绝。已存在的输出目录不会覆盖。渲染途中失败时返回非零退出码，临时目录保留 `FAILED.json`；只有所有场景成功后，输出目录才会发布，`manifest.json` 才标记 `complete`。

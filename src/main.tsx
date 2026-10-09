import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Camera, Layers, Download, RefreshCw, Dices, AlertCircle, Image as ImageIcon } from "lucide-react";
import { StoreRenderer } from "./sim/renderer";
import { generateStore } from "./sim/store";
import { makeSweep, type Sweep } from "./simulator";
import { makeZip, sequenceFiles, saveFile, imageBytes } from "./archive";
import { DEFAULT_OPTIONS } from "./defaults";
import { resolveBayLayout, MAX_BAYS } from "./sim/structure";
import { resolveShelfLength } from "./sweep";
import "./style.css";

type Setting = {
  mode: "random" | "custom";
  value: number;
  min: number;
  max: number;
};
const initialLength: Setting = { mode: "random", value: 3, min: 2, max: 8 };
const initialFrames: Setting = { mode: "custom", value: 9, min: 8, max: 16 };
const initialBayLayers = DEFAULT_OPTIONS.bayLayers;
const initialView = DEFAULT_OPTIONS.view;
function settingValue(setting: Setting) {
  return setting.mode === "custom"
    ? setting.value
    : { min: setting.min, max: setting.max };
}
function GenerationSetting({
  title,
  unit,
  min,
  max,
  step,
  value,
  onChange,
  disabled,
}: {
  title: string;
  unit?: string;
  min: number;
  max: number;
  step: number;
  value: Setting;
  onChange: (value: Setting) => void;
  disabled: boolean;
}) {
  return (
    <div className="generation-setting">
      <div className="setting-title">
        <span>
          {title}
          {unit && <small>{unit}</small>}
        </span>
        <select
          aria-label={`${title}模式`}
          disabled={disabled}
          value={value.mode}
          onChange={(e) =>
            onChange({ ...value, mode: e.target.value as Setting["mode"] })
          }
        >
          <option value="random">随机</option>
          <option value="custom">自定义</option>
        </select>
      </div>
      {value.mode === "custom" ? (
        <input
          aria-label={title}
          type="number"
          min={min}
          max={max}
          step={step}
          value={value.value}
          disabled={disabled}
          onChange={(e) =>
            onChange({ ...value, value: Number(e.target.value) })
          }
        />
      ) : (
        <div className="range-inputs">
          <input
            aria-label={`${title}下限`}
            type="number"
            min={min}
            max={max}
            step={step}
            value={value.min}
            disabled={disabled}
            onChange={(e) =>
              onChange({ ...value, min: Number(e.target.value) })
            }
          />
          <span>–</span>
          <input
            aria-label={`${title}上限`}
            type="number"
            min={min}
            max={max}
            step={step}
            value={value.max}
            disabled={disabled}
            onChange={(e) =>
              onChange({ ...value, max: Number(e.target.value) })
            }
          />
        </div>
      )}
    </div>
  );
}
function App() {
  const [lengthSetting, setLengthSetting] = useState<Setting>(initialLength);
  const [frameSetting, setFrameSetting] = useState<Setting>(initialFrames);
  const [bayLayers, setBayLayers] = useState(initialBayLayers);
  const [bayMode, setBayMode] = useState<"auto" | "manual">("auto");
  const [bayWidth, setBayWidth] = useState(DEFAULT_OPTIONS.bayWidth);
  const [manualBayCount, setManualBayCount] = useState(3);
  const [viewAngles, setViewAngles] = useState<{ mode: "fixed" | "handheld"; yaw: number; pitch: number; jitter: number }>(initialView);
  const [stock, setStock] = useState(DEFAULT_OPTIONS.stock);
  const [seed, setSeed] = useState(42), [sweep, setSweep] = useState<Sweep | null>(null);
  const [selected, setSelected] = useState(0), [error, setError] = useState("");
  const [preview, setPreview] = useState<"rgb" | "gt" | "mask">("rgb");
  const [busy, setBusy] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null), renderer = useRef<StoreRenderer | null>(null);
  useEffect(() => {
    localStorage.removeItem("tinyStitch.rgb-grid.lastJob");
    if (location.pathname !== "/" || location.search) history.replaceState(null, "", "/");
    let view: StoreRenderer | undefined;
    try {
      view = new StoreRenderer(canvas.current!, generateStore(42, 8), 720, 960);
      renderer.current = view;
      setSweep(makeSweep(view, 42, { ...DEFAULT_OPTIONS, shelfLength: settingValue(initialLength), frames: settingValue(initialFrames) }));
    } catch (e) { setError(`三维渲染启动失败：${(e as Error).message}`); }
    return () => { view?.dispose(); renderer.current = null; };
  }, []);
  const generate = (newSeed = seed) => {
    try {
      if (!renderer.current) throw Error("三维渲染器尚未就绪");
      setSweep(makeSweep(renderer.current, newSeed, { shelfLength: settingValue(lengthSetting), frames: settingValue(frameSetting), bayMode, bayWidth,
        bayLayers: bayMode === "auto" ? bayLayers : Array.from({ length: manualBayCount }, (_, i) => bayLayers[i % bayLayers.length]), view: viewAngles, stock }));
      setSeed(newSeed); setSelected(0); setError("");
    } catch (e) { setError((e as Error).message); }
  };
  const download = () => {
    if (!sweep) return;
    setBusy(true); setError("");
    try { saveFile(new Blob([makeZip(sequenceFiles(sweep))], { type: "application/zip" }), `shelf-${sweep.store.seed}.zip`); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const photos = sweep?.frames || [];
  const currentPose = sweep?.cameraGroundTruth.frames[selected];
  const previewImage = preview === "rgb" ? photos[selected] : preview === "gt" ? sweep?.groundTruth.image : sweep?.groundTruth.coverageMask;
  const previewName = preview === "rgb" ? `rgb/${String(selected).padStart(4, "0")}.jpg` : preview === "gt" ? "gt/orthographic.png" : "gt/coverage.png";
  const previewAlt = preview === "rgb" ? `拍摄图片 ${selected + 1}` : preview === "gt" ? "扫过区域的正交 GT" : "扫过区域的覆盖掩码";
  let activeBayLayers: number[] = [];
  try {
    activeBayLayers = resolveBayLayout(resolveShelfLength(seed, settingValue(lengthSetting)), {
      bayMode, bayWidth, bayLayers: bayMode === "auto" ? bayLayers : Array.from({ length: manualBayCount }, (_, i) => bayLayers[i % bayLayers.length]),
    }).layers;
  } catch { /* Invalid draft values are explained when generation is requested. */ }
  return <div className="app">
    <header><a className="brand" href="/"><span className="brand-icon"><Layers size={22} /></span>
      <div><strong>tinyStitch <span>/ 合成数据</span></strong><small>三维货架拍摄序列生成器</small></div></a>
      <span className="connection online"><i />浏览器内生成</span></header>
    <main>
      <div className="intro"><div><div className="eyebrow">SHELF DATA GENERATOR</div>
        <h1>生成一组货架拍摄图片。</h1><p>独立设置子货架层板数量，选择拍摄视角，下载有序 RGB 序列与扫过区域的正交 GT。</p></div></div>
      {error && <div className="error" role="alert"><AlertCircle size={18} /><span>{error}</span></div>}
      <div className="workspace">
        <section className="panel source"><div className="panel-title"><span><Camera size={18} />生成设置</span><small>{photos.length} 张图片</small></div>
          <div className="controls"><label>场景种子<input aria-label="场景种子" type="number" min="0" max="4294967295" value={seed} onChange={e => setSeed(Number(e.target.value))} disabled={busy} /></label>
            <button onClick={() => generate()} disabled={busy}><RefreshCw size={15} />生成拍摄</button>
            <button className="icon-button" aria-label="随机生成" title="使用新种子随机生成" disabled={busy} onClick={() => generate(crypto.getRandomValues(new Uint32Array(1))[0])}><Dices size={16} /></button></div>
          <div className="generation-options">
            <GenerationSetting title="货架长度" unit="模拟米" min={1} max={12} step={0.1} value={lengthSetting} onChange={setLengthSetting} disabled={busy} />
            <GenerationSetting title="图片张数" min={2} max={48} step={1} value={frameSetting} onChange={setFrameSetting} disabled={busy} />
            <p>长度 1–12，图片 2–48 张。点击“生成拍摄”应用设置；相同种子与参数可复现。</p>
          </div>
          <div className="configuration-section">
            <h2>子货架结构</h2>
            <label className="inline-setting">子货架数量模式
              <select aria-label="子货架数量模式" value={bayMode} disabled={busy} onChange={e => setBayMode(e.target.value as "auto" | "manual")}>
                <option value="auto">随长度自动伸缩</option><option value="manual">手动数量</option>
              </select>
            </label>
            {bayMode === "auto" ? <label className="generation-setting">目标子货架宽度（模拟米）<input aria-label="目标子货架宽度" type="number" min={0.5} max={3} step={0.1} value={bayWidth} disabled={busy} onChange={e => setBayWidth(Number(e.target.value))} /></label> :
              <label className="inline-setting">子货架数量<select aria-label="子货架数量" value={manualBayCount} disabled={busy} onChange={e => setManualBayCount(Number(e.target.value))}>
                {Array.from({ length: MAX_BAYS }, (_, i) => <option key={i} value={i + 1}>{i + 1} 个</option>)}
              </select></label>}
            <span className="setting-title">按当前种子与长度：{activeBayLayers.length || "—"} 个子货架</span>
            <div className="bay-settings">{activeBayLayers.map((layers, i) => <label key={i}>子货架 {i + 1}
              <select aria-label={`子货架 ${i + 1} 层板数量`} value={layers} disabled={busy} onChange={e => setBayLayers(current => Array.from({ length: Math.max(current.length, i + 1) }, (_, j) => j === i ? Number(e.target.value) : current[j % current.length]))}>
                {Array.from({ length: 8 }, (_, j) => <option key={j} value={j + 1}>{j + 1} 层</option>)}
              </select>
            </label>)}</div>
            <p>默认每段约 1 模拟米，数量向上取整后等宽分配；随机长度按当前种子计算。新增段循环沿用层板配置，每段可独立修改，层板数量包含底层。</p>
          </div>
          <div className="configuration-section">
            <h2>商品陈列</h2>
            <div className="angle-settings">
              <label>空位比例（%）<input aria-label="空位比例" type="number" min={0} max={100} step={1} value={Math.round(stock.emptyRate * 100)} disabled={busy} onChange={e => setStock(current => ({ ...current, emptyRate: Number(e.target.value) / 100 }))} /></label>
              <label>每列纵深容量<input aria-label="每列纵深容量" type="number" min={1} max={4} step={1} value={stock.depthCopies} disabled={busy} onChange={e => setStock(current => ({ ...current, depthCopies: Number(e.target.value) }))} /></label>
            </div>
            <div className="generation-setting"><span className="setting-title">同款连续列数范围</span>
              <div className="range-inputs">{(["facingsMin", "facingsMax"] as const).map((key, i) => <input key={key} aria-label={`同款连续列数${i ? "上限" : "下限"}`} type="number" min={1} max={12} step={1} disabled={busy} value={stock[key]} onChange={e => setStock(current => ({ ...current, [key]: Number(e.target.value) }))} />)}</div>
            </div>
            <p>混合盒装、瓶装、罐装与广口罐，按实际宽度紧凑排列，商品间仅留约 4 毫米。缺货时保留原货位，不收拢填空。最大宽度沿用原上限；层间净空足够时，盒装、罐装和广口罐从底部堆叠，瓶装保持单层。</p>
          </div>
          <div className="configuration-section">
            <h2>拍摄视角</h2>
            <label className="inline-setting">拍摄模式<select aria-label="拍摄模式" disabled={busy} value={viewAngles.mode} onChange={e => setViewAngles(current => ({ ...current, mode: e.target.value as "fixed" | "handheld" }))}>
              <option value="fixed">固定路线</option><option value="handheld">仿真拍摄 · 轻微手持</option>
            </select></label>
            <div className="view-presets" role="group" aria-label="视角预设">
              {([{ title: "正面", yaw: 0, pitch: 0 }, { title: "左侧斜拍", yaw: 18, pitch: 0 }, { title: "右侧斜拍", yaw: -18, pitch: 0 }, { title: "俯拍", yaw: 0, pitch: 15 }]).map(preset =>
                <button key={preset.title} disabled={busy} aria-pressed={viewAngles.mode === "fixed" && viewAngles.yaw === preset.yaw && viewAngles.pitch === preset.pitch} onClick={() => setViewAngles(current => ({ ...current, mode: "fixed", yaw: preset.yaw, pitch: preset.pitch }))}>{preset.title}</button>)}
              <button disabled={busy} aria-pressed={viewAngles.mode === "handheld"} onClick={() => setViewAngles({ ...DEFAULT_OPTIONS.view })}>仿真拍摄</button>
            </div>
            <div className="angle-settings">
              {([{ key: "yaw", title: "偏航角", min: -25, max: 25 }, { key: "pitch", title: "俯仰角", min: -20, max: 20 }, { key: "jitter", title: "逐帧角度扰动", min: 0, max: 15 }] as const).map(field =>
                <label key={field.key}>{field.title}（°）<input aria-label={field.title} type="number" min={field.min} max={field.max} step={0.5} value={viewAngles[field.key]} disabled={busy} onChange={e => setViewAngles(current => ({ ...current, [field.key]: Number(e.target.value) }))} /></label>)}
            </div>
            <p>默认仿真拍摄：偏航、俯仰各在 −15°～15° 范围内连续随机扰动，并带轻微位置变化、滚转和非匀速采样。偏航、俯仰输入为中心角，角度扰动为最大幅度。</p>
            <button className="apply-settings" onClick={() => generate()} disabled={busy}><RefreshCw size={15} />应用设置并生成</button>
          </div>
          <canvas ref={canvas} className="hidden-renderer" />
          {sweep && <div className="generator-summary"><dl>
            <div><dt>货架长度</dt><dd>{sweep.shelf.width.toFixed(2)} 模拟米</dd></div>
            <div><dt>图片尺寸</dt><dd>720 × 960</dd></div>
            <div><dt>图片张数</dt><dd>{photos.length} 张</dd></div>
            <div><dt>子货架数量 / 单段宽</dt><dd>{sweep.shelf.bays?.length} 个 / {sweep.shelf.bays?.[0].width.toFixed(2)} 米</dd></div>
            <div><dt>各子货架层板</dt><dd>{sweep.shelf.bays?.map(b => b.layers).join(" / ")} 层</dd></div>
            <div><dt>偏航 / 俯仰</dt><dd>{sweep.options.view?.yaw}° / {sweep.options.view?.pitch}°</dd></div>
            <div><dt>拍摄模式</dt><dd>{sweep.options.view?.mode === "handheld" ? "仿真拍摄" : "固定路线"}</dd></div>
            <div><dt>全货架空位 / 货位</dt><dd>{sweep.shelf.merchandising?.slots.filter(s => s.stock === 0).length} / {sweep.shelf.merchandising?.slots.length}</dd></div>
            <div><dt>相邻估计重叠</dt><dd>{(sweep.estimatedOverlap * 100).toFixed(0)}%</dd></div>
          </dl><p>镜头沿货架正面横向移动，使用设定视角与逐帧扰动。图片较少时自动后退；重叠率为正面视角下的近似值。</p></div>}
          <div className="generator-download"><button className="primary" onClick={download} disabled={!sweep || busy}><Download size={16} />下载完整序列 ZIP</button>
            <p>包含 RGB、正交 GT、覆盖掩码、坐标映射、相机真实世界 6D 位姿 GT（JSON / CSV）、内参、时间戳、SKU 陈列与空位数据，以及可供 CLI 复用的配置。</p>
            <button disabled={!sweep || busy} onClick={() => { if (sweep) saveFile(new Blob([JSON.stringify(sweep.cameraGroundTruth, null, 2)], { type: "application/json" }), "camera_poses.json"); }}><Download size={14} />下载相机位姿 GT</button>
            <button disabled={!sweep || busy} onClick={() => { if (sweep) saveFile(new Blob([JSON.stringify({ schemaVersion: 1, seed: sweep.store.seed, options: sweep.options }, null, 2)], { type: "application/json" }), "shelf-config.json"); }}><Download size={14} />下载 CLI 生成配置</button>
          </div>
        </section>
        <section className="panel output"><div className="panel-title"><span><ImageIcon size={18} />图片预览</span><small>{preview === "rgb" ? (photos.length ? `${selected + 1} / ${photos.length}` : "等待生成") : sweep ? `${sweep.groundTruth.metadata.width} × ${sweep.groundTruth.metadata.height}` : "等待生成"}</small></div>
          <div className="preview-tabs" role="group" aria-label="预览内容">{([{ key: "rgb", title: "拍摄序列" }, { key: "gt", title: "正交 GT" }, { key: "mask", title: "覆盖掩码" }] as const).map(tab =>
            <button key={tab.key} aria-pressed={preview === tab.key} onClick={() => setPreview(tab.key)}>{tab.title}</button>)}</div>
          <div className={`photo-stage generator-preview ${preview !== "rgb" ? "atlas-preview" : ""}`}>{previewImage ? <img src={previewImage} alt={previewAlt} /> : <div className="empty"><Camera size={32} /><p>生成后在这里查看图片</p></div>}</div>
          {preview === "rgb" ? <div className="thumbs">{photos.map((src, i) => <button key={i} aria-label={`查看第 ${i + 1} 张`} className={i === selected ? "active" : ""} onClick={() => setSelected(i)}><img src={src} alt={`序列 ${i + 1}`} loading="lazy" /><span>{i + 1}</span></button>)}</div> :
            <p className="gt-description">正面正交渲染，按全部视角在货架正面参考平面上的覆盖并集裁出；未扫到的区域透明。掩码白色为覆盖、黑色为未覆盖，不表示商品遮挡或缺货。{sweep && ` 比例 ${sweep.groundTruth.metadata.pixelsPerMeter.toFixed(0)} 像素 / 模拟米。`}</p>}
          {previewImage && <div className="generator-frame"><span>{previewName}</span><button onClick={() => saveFile(new Blob([imageBytes(previewImage)], { type: preview === "rgb" ? "image/jpeg" : "image/png" }), previewName.split("/").at(-1)!)}><Download size={14} />下载当前图片</button></div>}
          {preview === "rgb" && currentPose && <section className="pose-preview" aria-label="相机真实世界位姿 GT">
            <h2>当前帧 · 相机真实世界位姿 GT</h2>
            <dl><div><dt>X / Y / Z（米）</dt><dd>{currentPose.position_m.map(v => v.toFixed(4)).join(" / ")}</dd></div>
              <div><dt>Rx / Ry / Rz（°）</dt><dd>{currentPose.euler_xyz_deg.map(v => v.toFixed(3)).join(" / ")}</dd></div>
              <div><dt>时间戳（秒）</dt><dd>{currentPose.timestamp_s.toFixed(6)}</dd></div></dl>
            <p>场景世界坐标，Y 向上；旋转为内禀 Euler XYZ。完整 GT 包含米 / 弧度 6D 坐标、四元数和双向变换矩阵。</p>
          </section>}
        </section>
      </div>
      <footer><span>Three.js 三维货架 · 有序 RGB 序列</span><span>无需后端或模型权重</span></footer>
    </main>
  </div>;
}
createRoot(document.getElementById("root")!).render(<App />);

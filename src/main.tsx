import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Camera,
  Upload,
  Layers,
  Download,
  ArrowRight,
  RefreshCw,
  Dices,
  CheckCircle,
  AlertCircle,
  Square,
  Image as ImageIcon,
} from "lucide-react";
import { StoreRenderer } from "../../src/sim/renderer";
import { generateStore } from "../../src/sim/store";
import { makeSweep, type Sweep } from "./simulator";
import { createSweepPlan, type SweepOptions } from "./sweep";
import "./style.css";

type Input = { id: string; count: number; names: string[]; preview: string[] };
type Report = {
  size: number[];
  seconds: number;
  input_count: number;
  input_digest: string;
  alignment: { after_median_px: number };
  pairs: {
    i: number;
    j: number;
    accepted: boolean;
    matches: number;
    inliers: number;
    median_error_px?: number;
    reason?: string;
  }[];
  warnings: string[];
};
type Job = {
  id: string;
  status: string;
  progress: number;
  message: string;
  error?: string;
  result?: { report: Report; png: string; jpg: string; json: string };
};
async function api(path: string, options?: RequestInit) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let text = await res.text();
    try {
      text = JSON.parse(text).detail || text;
    } catch {}
    throw Error(typeof text === "string" ? text : JSON.stringify(text));
  }
  return res.json();
}
type Setting = {
  mode: "random" | "custom";
  value: number;
  min: number;
  max: number;
};
const initialLength: Setting = { mode: "random", value: 3, min: 2, max: 8 };
const initialFrames: Setting = { mode: "custom", value: 9, min: 8, max: 16 };
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
  const [lengthSetting, setLengthSetting] = useState<Setting>(initialLength),
    [frameSetting, setFrameSetting] = useState<Setting>(initialFrames);
  const [seed, setSeed] = useState(42),
    [sweep, setSweep] = useState<Sweep | null>(null),
    [input, setInput] = useState<Input | null>(null);
  const [selected, setSelected] = useState(0),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [job, setJob] = useState<Job | null>(null),
    [online, setOnline] = useState(false),
    [method, setMethod] = useState("jepa"),
    [health, setHealth] = useState<any>(null);
  const canvas = useRef<HTMLCanvasElement>(null),
    renderer = useRef<StoreRenderer | null>(null),
    poll = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    const view = new StoreRenderer(
      canvas.current!,
      generateStore(42, 8),
      720,
      960,
    );
    renderer.current = view;
    try {
      setSweep(
        makeSweep(view, 42, {
          shelfLength: settingValue(initialLength),
          frames: settingValue(initialFrames),
        }),
      );
    } catch (e) {
      setError(String(e));
    }
    return () => {
      view.dispose();
      if (poll.current) clearTimeout(poll.current);
    };
  }, []);
  useEffect(() => {
    let alive = true;
    const check = () =>
      api("/api/health")
        .then((value) => {
          if (alive) {
            setOnline(true);
            setHealth(value);
          }
        })
        .catch(() => {
          if (alive) setOnline(false);
        });
    check();
    const timer = setInterval(check, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  const running = !!job && ["queued", "running"].includes(job.status);
  const generate = (newSeed = seed) => {
    try {
      setSweep(
        makeSweep(renderer.current!, newSeed, {
          shelfLength: settingValue(lengthSetting),
          frames: settingValue(frameSetting),
        }),
      );
      setSeed(newSeed);
      setInput(null);
      setJob(null);
      setSelected(0);
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const storeCapture = async () => {
    if (input) return input;
    if (!sweep) throw Error("没有图片");
    const stored: Input = await api("/api/captures", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames: sweep.frames }),
    });
    setInput(stored);
    return stored;
  };
  const upload = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      files.forEach((f) => body.append("files", f));
      const stored = await api("/api/uploads", { method: "POST", body });
      setInput(stored);
      setSweep(null);
      setJob(null);
      setSelected(0);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const watch = async (id: string) => {
    try {
      const current: Job = await api(`/api/jobs/${id}`);
      setJob(current);
      if (["queued", "running"].includes(current.status))
        poll.current = setTimeout(() => watch(id), 500);
      else if (current.status === "failed")
        setError(current.error || "拼接失败");
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const stitch = async () => {
    setBusy(true);
    setError("");
    setJob(null);
    try {
      const stored = await storeCapture();
      const current = await api("/api/stitch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rgb_id: stored.id, method }),
      });
      setJob(current);
      watch(current.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const downloadSequence = async () => {
    setBusy(true);
    try {
      const stored = await storeCapture();
      const a = document.createElement("a");
      a.href = `/api/inputs/${stored.id}/download`;
      a.download = "shelf-sequence.zip";
      a.click();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const photos = input?.preview || sweep?.frames || [];
  const report = job?.result?.report;
  // Headless generator uses the exact same renderer and sweep as the visible app.
  useEffect(() => {
    (window as any).tinyStitch = {
      generate: (newSeed: number, options: number | SweepOptions = 9) =>
        makeSweep(renderer.current!, newSeed, options),
      plan: createSweepPlan,
    };
    return () => {
      delete (window as any).tinyStitch;
    };
  }, []);
  return (
    <div className="app">
      <header>
        <a className="brand" href="http://127.0.0.1:5173">
          <span className="brand-icon">
            <Layers size={22} />
          </span>
          <div>
            <strong>
              tinyLayout <span>/ tinyStitch</span>
            </strong>
            <small>货架序列图片拼接实验</small>
          </div>
        </a>
        <span className={`connection ${online ? "online" : ""}`}>
          <i />
          {online ? "拼接服务就绪" : "等待后端 · 8010"}
        </span>
      </header>
      <main>
        <div className="intro">
          <div>
            <div className="eyebrow">SHELF MOSAIC LAB</div>
            <h1>把连续拍摄，拼成一张货架图。</h1>
            <p>单个货架的一侧正面 · JEPA 表示匹配 · 仅使用 RGB 图片</p>
          </div>
          <div className="steps">
            <span>01 拍摄序列</span>
            <ArrowRight size={14} />
            <span>02 自动对齐</span>
            <ArrowRight size={14} />
            <span>03 拼图导出</span>
          </div>
        </div>
        <div className="workspace">
          <section className="panel source">
            <div className="panel-title">
              <span>
                <Camera size={18} />
                输入序列
              </span>
              <small>{photos.length} 张照片</small>
            </div>
            <div className="controls">
              <label>
                场景种子
                <input
                  aria-label="场景种子"
                  type="number"
                  min="0"
                  max="4294967295"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  disabled={running || busy}
                />
              </label>
              <button onClick={() => generate()} disabled={busy || running}>
                <RefreshCw size={15} />
                生成拍摄
              </button>
              <button
                className="icon-button"
                aria-label="随机生成"
                title="使用新种子随机生成"
                disabled={busy || running}
                onClick={() =>
                  generate(crypto.getRandomValues(new Uint32Array(1))[0])
                }
              >
                <Dices size={16} />
              </button>
            </div>
            <div className="generation-options">
              <GenerationSetting
                title="货架长度"
                unit="模拟米"
                min={1}
                max={12}
                step={0.1}
                value={lengthSetting}
                onChange={setLengthSetting}
                disabled={busy || running}
              />
              <GenerationSetting
                title="图片张数"
                min={2}
                max={48}
                step={1}
                value={frameSetting}
                onChange={setFrameSetting}
                disabled={busy || running}
              />
              <p>
                长度 1–12，图片 2–48
                张。点击“生成拍摄”应用设置；骰子按钮换种子随机生成。
              </p>
            </div>
            <div className="method-control">
              <label>
                拼接方法
                <select
                  aria-label="拼接方法"
                  value={method}
                  disabled={busy || running}
                  onChange={(e) => {
                    setMethod(e.target.value);
                    setJob(null);
                    setError("");
                  }}
                >
                  <option value="jepa">JEPA 学习特征（主流程）</option>
                  <option value="sift">SIFT 传统特征（对照）</option>
                </select>
              </label>
              <p>
                {method === "jepa"
                  ? health?.jepa_ready
                    ? `已训练 ${health.model.trained_steps} 步 · ${health.model.parameters.toLocaleString()} 参数`
                    : `缺少已训练 JEPA 权重，预测已禁用`
                  : "明确使用传统特征对照，不调用 JEPA"}
              </p>
              {health?.training && (
                <p>
                  训练进度 {health.training.step} /{" "}
                  {health.training.total_steps} ·{" "}
                  {health.training.device.toUpperCase()}
                </p>
              )}
            </div>
            <canvas ref={canvas} className="hidden-renderer" />
            <div className="photo-stage">
              {photos[selected] && (
                <img src={photos[selected]} alt={`拍摄图片 ${selected + 1}`} />
              )}
              <span className="frame-number">
                {selected + 1} / {photos.length}
              </span>
            </div>
            <div className="thumbs">
              {photos.map((src, i) => (
                <button
                  key={src.slice(-60) + i}
                  aria-label={`查看第 ${i + 1} 张`}
                  className={i === selected ? "active" : ""}
                  onClick={() => setSelected(i)}
                >
                  <img src={src} alt={`序列 ${i + 1}`} loading="lazy" />
                  <span>{i + 1}</span>
                </button>
              ))}
            </div>
            {sweep && (
              <p className="source-note">
                场景 {sweep.store.seed} · 实际货架长度{" "}
                {sweep.shelf.width.toFixed(2)} 模拟米 · {sweep.frames.length}{" "}
                张照片。沿正面横向移动，相邻画面估计重叠{" "}
                {(sweep.estimatedOverlap * 100).toFixed(0)}
                %。图片较少时镜头自动后退，商品会更小。
              </p>
            )}
            {input && !sweep && (
              <p className="source-note">
                已按文件名数字顺序导入。请保持同一货架、同一侧，照片之间保留
                50%–70% 重叠。
              </p>
            )}
            <div className="actions">
              <label
                className={`button secondary ${busy || running ? "disabled" : ""}`}
              >
                <Upload size={16} />
                导入拍摄图片
                <input
                  aria-label="导入拍摄图片"
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  multiple
                  disabled={busy || running}
                  onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    e.target.value = "";
                    upload(files);
                  }}
                />
              </label>
              <button
                className="icon-button"
                title="导出输入序列"
                aria-label="导出输入序列"
                onClick={downloadSequence}
                disabled={!photos.length || busy || running || !online}
              >
                <Download size={17} />
              </button>
            </div>
            <button
              className="primary stitch-button"
              onClick={stitch}
              disabled={
                !online ||
                !photos.length ||
                busy ||
                running ||
                (method === "jepa" && !health?.jepa_ready)
              }
            >
              <Layers size={18} />
              {busy ? "准备图片…" : running ? "正在自动拼接…" : "生成完整拼图"}
              <ArrowRight size={17} />
            </button>
            <p className="hint">
              不需要手动选点。支持 JPG / PNG / WebP，2–48 张；HEIC 请先转换为
              JPEG。
            </p>
          </section>
          <section className="panel output">
            <div className="panel-title">
              <span>
                <ImageIcon size={18} />
                拼接结果
              </span>
              <small>图像像素坐标</small>
            </div>
            {error && (
              <div role="alert" className="error">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}
            {running && (
              <div className="progress">
                <div>
                  <span>{job?.message}</span>
                  <button
                    onClick={() =>
                      api(`/api/jobs/${job!.id}/cancel`, { method: "POST" })
                    }
                  >
                    <Square size={12} />
                    取消
                  </button>
                </div>
                <progress value={job!.progress} max="1" />
              </div>
            )}
            {job?.status === "cancelled" && (
              <p className="hint">任务已取消，可以重新拼接。</p>
            )}
            <div
              className={`panorama-stage ${job?.result ? "has-result" : ""}`}
            >
              {job?.result ? (
                <img src={job.result.png} alt="自动拼接的完整货架图片" />
              ) : (
                <div className="empty">
                  <div>
                    <Layers size={34} />
                  </div>
                  <h2>等待第一张拼图</h2>
                  <p>
                    使用左侧模拟序列，或导入自己的连续照片。
                    <br />
                    程序自动对齐、校正曝光并融合接缝。
                  </p>
                </div>
              )}
            </div>
            {report && (
              <>
                <div className="result-stats">
                  <div>
                    <small>输出尺寸</small>
                    <strong>{report.size.join(" × ")}</strong>
                  </div>
                  <div>
                    <small>计算耗时</small>
                    <strong>{report.seconds.toFixed(2)} s</strong>
                  </div>
                  <div>
                    <small>匹配残差中位数</small>
                    <strong>
                      {report.alignment.after_median_px.toFixed(2)} px
                    </strong>
                  </div>
                  <div>
                    <small>有效连接</small>
                    <strong>
                      {report.pairs.filter((p) => p.accepted).length} /{" "}
                      {report.pairs.length}
                    </strong>
                  </div>
                </div>
                <div className="result-actions">
                  <span>
                    <CheckCircle size={16} />
                    已使用全部 {report.input_count} 张输入照片
                  </span>
                  <div>
                    <a
                      className="button"
                      href={job!.result!.png}
                      download="shelf-panorama.png"
                    >
                      <Download size={15} />
                      PNG
                    </a>
                    <a
                      className="button"
                      href={job!.result!.jpg}
                      download="shelf-panorama.jpg"
                    >
                      JPG
                    </a>
                    <a
                      className="button"
                      href={job!.result!.json}
                      download="stitch-report.json"
                    >
                      报告 JSON
                    </a>
                  </div>
                </div>
                <details>
                  <summary>查看匹配诊断</summary>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>图片连接</th>
                          <th>匹配数</th>
                          <th>内点数</th>
                          <th>状态</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.pairs.map((p) => (
                          <tr key={`${p.i}-${p.j}`}>
                            <td>
                              {p.i + 1} → {p.j + 1}
                            </td>
                            <td>{p.matches}</td>
                            <td>{p.inliers}</td>
                            <td title={p.reason}>
                              {p.accepted ? "通过" : p.reason}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="digest">输入 SHA-256：{report.input_digest}</p>
                </details>
                <div className="notes">
                  {report.warnings.map((w) => (
                    <p key={w}>{w}</p>
                  ))}
                </div>
              </>
            )}
            {!report && (
              <div className="notes">
                <p>
                  “完整”表示融合这组照片的已拍摄区域。未拍到的货架和商品不会被补画。商品凸起或镜头大幅转动可能产生重影。
                </p>
              </div>
            )}
          </section>
        </div>
        <footer>
          JEPA 潜在表示 · RANSAC 平面变换 · 全序列优化 · 窄缝融合{" "}
          <span>
            跨视角 JEPA 实验模型 · MPS / CPU ·{" "}
            <a href="/api/experiment-report" target="_blank" rel="noreferrer">
              实际实验报告
            </a>
          </span>
        </footer>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);

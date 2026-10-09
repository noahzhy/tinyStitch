import type { Sweep } from "./simulator";
import { cameraPoseCSV } from "./sim/camera-pose.ts";

export type ArchiveFile = { name: string; bytes: Uint8Array };
export const imageBytes = (data: string) => Uint8Array.from(atob(data.split(",")[1]), c => c.charCodeAt(0));
const encoder = new TextEncoder();
const crcTable = Uint32Array.from({ length: 256 }, (_, n) => {
  let value = n;
  for (let i = 0; i < 8; i++) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  return value >>> 0;
});
export function crc32(bytes: Uint8Array) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crcTable[(crc ^ byte) & 255] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

// Standard, uncompressed ZIP. JPEG inputs already contain compressed pixels.
export function makeZip(files: ArchiveFile[]) {
  const chunks: Uint8Array[] = [], directory: Uint8Array[] = [];
  let offset = 0, directorySize = 0;
  for (const file of files) {
    const name = encoder.encode(file.name), checksum = crc32(file.bytes);
    const header = new Uint8Array(30 + name.length), h = new DataView(header.buffer);
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x800, true);
    h.setUint16(12, 0x21, true); h.setUint32(14, checksum, true);
    h.setUint32(18, file.bytes.length, true); h.setUint32(22, file.bytes.length, true);
    h.setUint16(26, name.length, true); header.set(name, 30);
    const entry = new Uint8Array(46 + name.length), d = new DataView(entry.buffer);
    d.setUint32(0, 0x02014b50, true); d.setUint16(4, 20, true); d.setUint16(6, 20, true);
    d.setUint16(8, 0x800, true); d.setUint16(14, 0x21, true); d.setUint32(16, checksum, true);
    d.setUint32(20, file.bytes.length, true); d.setUint32(24, file.bytes.length, true);
    d.setUint16(28, name.length, true); d.setUint32(42, offset, true); entry.set(name, 46);
    chunks.push(header, file.bytes); directory.push(entry);
    offset += header.length + file.bytes.length; directorySize += entry.length;
  }
  const end = new Uint8Array(22), e = new DataView(end.buffer);
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
  e.setUint32(12, directorySize, true); e.setUint32(16, offset, true);
  const result = new Uint8Array(offset + directorySize + end.length);
  let at = 0;
  for (const part of [...chunks, ...directory, end]) { result.set(part, at); at += part.length; }
  return result;
}

export function sequenceFiles(sweep: Sweep): ArchiveFile[] {
  const { frames, groundTruth, cameraGroundTruth, ...scene } = sweep;
  if (cameraGroundTruth.frames.length !== frames.length) throw Error("相机位姿 GT 与 RGB 帧数不一致");
  const width = 720, height = 960, focal = height / (2 * Math.tan(54 * Math.PI / 360));
  const json = (name: string, value: unknown) => ({ name, bytes: encoder.encode(JSON.stringify(value, null, 2)) });
  const image = (name: string, data: string) => ({ name, bytes: imageBytes(data) });
  return [
    ...frames.map((frame, i) => ({ name: `rgb/${String(i).padStart(4, "0")}.jpg`,
      bytes: imageBytes(frame) })),
    image("gt/orthographic.png", groundTruth.image),
    image("gt/coverage.png", groundTruth.coverageMask),
    json("gt/metadata.json", groundTruth.metadata),
    json("gt/camera_poses.json", cameraGroundTruth),
    { name: "gt/camera_poses.csv", bytes: encoder.encode(cameraPoseCSV(cameraGroundTruth)) },
    json("generation/scene.json", scene),
    json("generation/config.json", { schemaVersion: 1, seed: sweep.store.seed, options: sweep.options }),
    json("intrinsics.json", { width, height, K: [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]] }),
    json("timestamps.json", sweep.poses.map(p => p.timestamp)),
    json("capture.json", { renderer: "Three.js", image_size: [width, height], vertical_fov_degrees: 54,
      timestamps: "Simulated sampling: fixed mode 0.5s; handheld mode seeded 0.45–0.55s intervals", poses: "Three.js world coordinates; camera looks along local -Z; rollDegrees rotates around camera local +Z after lookAt; see generation/scene.json",
      view_angles: "Degrees relative to shelf normal; positive yaw aims toward shelf local +X; positive pitch looks down. Per-frame values include seeded jitter.",
      bays: "Ordered along shelf local +X; x and layerHeights are in shelf-local coordinates. Each bay has independent board heights.",
      merchandising: "shelf.merchandising contains the SKU catalog, all slots including empty slots, stock counts and individual product boxes in shelf-local coordinates.",
      camera_pose_ground_truth: "gt/camera_poses.json and gt/camera_poses.csv contain actual per-RGB-frame world position and intrinsic XYZ Euler rotation, quaternion and camera/world transforms; see JSON convention metadata. Rotations are not shelf-relative input view angles.",
      ground_truth: "gt/orthographic.png is the frontal orthographic rendering cropped and masked to reference-plane coverage of all RGB views. gt/coverage.png is the binary coverage mask. gt/metadata.json records scale, bounds, coordinate mapping and per-frame footprints." }),
  ];
}

export function saveFile(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = name; document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

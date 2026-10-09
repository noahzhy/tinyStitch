import { StoreRenderer } from "./sim/renderer";
import { generateStore } from "./sim/store";
import { makeSweep } from "./simulator";
import { sequenceFiles } from "./archive";
import type { SweepOptions } from "./sweep";

declare global {
  interface Window {
    generateShelfCLI: (seed: number, options: SweepOptions) => { name: string; base64: string }[];
  }
}

let renderer: StoreRenderer | undefined;
// A dedicated rendering entry point, sharing the exact web geometry and export pipeline.
window.generateShelfCLI = (seed, options) => {
  renderer ??= new StoreRenderer(document.querySelector("canvas")!, generateStore(seed), 720, 960);
  return sequenceFiles(makeSweep(renderer, seed, options)).map(file => {
    let binary = "";
    for (let at = 0; at < file.bytes.length; at += 32768)
      binary += String.fromCharCode(...file.bytes.subarray(at, at + 32768));
    return { name: file.name, base64: btoa(binary) };
  });
};

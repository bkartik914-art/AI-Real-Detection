"use client";

import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, FormEvent } from "react";

type PredictionLabel = "AI Generated" | "Real" | "Uncertain";

type PredictionResponse = {
  filename: string | null;
  label: PredictionLabel;
  raw_label: "ai" | "real" | "uncertain";
  upstream_label: "ai" | "real";
  confidence: number;
  ai_probability: number;
  real_probability: number;
  thresholds: {
    ai_threshold: number;
    uncertainty_margin: number;
  };
  model: {
    provider: string;
    backend: string;
    model_dir: string | null;
    device: string;
    loaded: boolean;
  };
};

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);

const acceptedTypes = [
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/webp",
  "image/bmp",
  "image/tiff",
];

const percentFormatter = new Intl.NumberFormat("en", {
  maximumFractionDigits: 1,
  minimumFractionDigits: 0,
  style: "percent",
});

function formatPercent(value: number) {
  return percentFormatter.format(value);
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function readError(response: Response) {
  const contentType = response.headers.get("content-type") ?? "";

  if (contentType.includes("application/json")) {
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        return body.detail;
      }

      return JSON.stringify(body);
    } catch {
      return `Request failed with status ${response.status}`;
    }
  }

  const text = await response.text();
  return text || `Request failed with status ${response.status}`;
}

function verdictStyles(label: PredictionLabel) {
  if (label === "AI Generated") {
    return {
      badge: "border-red-200 bg-red-50 text-red-700",
      glow: "shadow-[0_18px_70px_rgba(239,68,68,0.18)]",
      ring: "ring-red-100",
    };
  }

  if (label === "Real") {
    return {
      badge: "border-emerald-200 bg-emerald-50 text-emerald-700",
      glow: "shadow-[0_18px_70px_rgba(16,185,129,0.18)]",
      ring: "ring-emerald-100",
    };
  }

  return {
    badge: "border-amber-200 bg-amber-50 text-amber-700",
    glow: "shadow-[0_18px_70px_rgba(245,158,11,0.18)]",
    ring: "ring-amber-100",
  };
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  function updatePreviewUrl(nextPreviewUrl: string | null) {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }

    previewUrlRef.current = nextPreviewUrl;
    setPreviewUrl(nextPreviewUrl);
  }

  function setNextFile(nextFile: File | null) {
    if (!nextFile) {
      return;
    }

    if (nextFile.type && !nextFile.type.startsWith("image/")) {
      setError("Choose a valid image file.");
      return;
    }

    setFile(nextFile);
    updatePreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setError(null);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setNextFile(event.target.files?.[0] ?? null);
  }

  function handleDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    setNextFile(event.dataTransfer.files?.[0] ?? null);
  }

  function clearSelection() {
    setFile(null);
    updatePreviewUrl(null);
    setResult(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file) {
      setError("Select an image before running detection.");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        body: formData,
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(await readError(response));
      }

      const prediction = (await response.json()) as PredictionResponse;
      setResult(prediction);
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : "Detection failed.";
      setError(
        message === "Failed to fetch"
          ? "Could not reach the backend. Start FastAPI on port 8000 or set NEXT_PUBLIC_API_URL."
          : message,
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  const styles = result ? verdictStyles(result.label) : null;
  const confidence = result
    ? result.raw_label === "real"
      ? result.real_probability
      : result.raw_label === "ai"
        ? result.ai_probability
        : result.confidence
    : 0;
  const aiBarWidth = result ? `${Math.round(result.ai_probability * 100)}%` : "0%";
  const realBarWidth = result ? `${Math.round(result.real_probability * 100)}%` : "0%";

  return (
    <main className="min-h-screen bg-[#f6f7f9] text-neutral-950">
      <section className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-5 py-6 sm:px-8 lg:px-10">
        <header className="flex flex-col gap-4 border-b border-neutral-200 pb-6 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm font-medium text-neutral-500">AI vs Real Detection</p>
            <h1 className="mt-2 max-w-2xl text-3xl font-semibold leading-tight">
              Upload an image and check whether it is AI generated.
            </h1>
          </div>
          <div className="rounded-full border border-neutral-200 bg-white px-4 py-2 text-sm text-neutral-600 shadow-sm">
            Backend: <span className="font-medium text-neutral-900">{API_BASE_URL}</span>
          </div>
        </header>

        <div className="grid flex-1 gap-6 py-6 lg:grid-cols-[minmax(0,1.08fr)_minmax(340px,0.92fr)]">
          <form
            className="flex min-h-[620px] flex-col rounded-lg border border-neutral-200 bg-white p-4 shadow-sm sm:p-5"
            onSubmit={handleSubmit}
          >
            <label
              className={[
                "relative flex flex-1 cursor-pointer flex-col overflow-hidden rounded-md border border-dashed transition",
                isDragging
                  ? "border-neutral-900 bg-neutral-50"
                  : "border-neutral-300 bg-neutral-50 hover:border-neutral-500",
              ].join(" ")}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                ref={inputRef}
                accept={acceptedTypes.join(",")}
                className="sr-only"
                onChange={handleFileChange}
                type="file"
              />

              {previewUrl ? (
                <div
                  aria-label={file ? `Preview of ${file.name}` : "Selected image preview"}
                  className="min-h-[430px] flex-1 bg-neutral-100 bg-contain bg-center bg-no-repeat"
                  role="img"
                  style={{ backgroundImage: `url(${previewUrl})` }}
                />
              ) : (
                <div className="flex min-h-[430px] flex-1 flex-col items-center justify-center px-6 text-center">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full border border-neutral-200 bg-white">
                    <svg
                      aria-hidden="true"
                      className="h-7 w-7 text-neutral-700"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <path
                        d="M12 16V4m0 0 4 4m-4-4-4 4M5 16v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2"
                        stroke="currentColor"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="1.8"
                      />
                    </svg>
                  </div>
                  <p className="mt-5 text-lg font-semibold text-neutral-950">
                    Drop an image here or click to browse.
                  </p>
                  <p className="mt-2 max-w-sm text-sm leading-6 text-neutral-500">
                    JPG, PNG, WEBP, BMP, and TIFF are supported by the backend.
                  </p>
                </div>
              )}
            </label>

            <div className="mt-5 flex flex-col gap-4 border-t border-neutral-100 pt-5 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-h-11 text-sm text-neutral-500">
                {file ? (
                  <>
                    <p className="max-w-xl truncate font-medium text-neutral-900">{file.name}</p>
                    <p>
                      {file.type || "image"} - {formatBytes(file.size)}
                    </p>
                  </>
                ) : (
                  <p>Select an image to enable analysis.</p>
                )}
              </div>
              <div className="flex gap-3">
                {file ? (
                  <button
                    className="h-11 rounded-md border border-neutral-200 px-4 text-sm font-medium text-neutral-700 transition hover:bg-neutral-50"
                    onClick={clearSelection}
                    type="button"
                  >
                    Clear
                  </button>
                ) : null}
                <button
                  className="h-11 rounded-md bg-neutral-950 px-5 text-sm font-semibold text-white transition hover:bg-neutral-800 disabled:cursor-not-allowed disabled:bg-neutral-300 disabled:text-neutral-500"
                  disabled={!file || isSubmitting}
                  type="submit"
                >
                  {isSubmitting ? "Analyzing..." : "Analyze image"}
                </button>
              </div>
            </div>
          </form>

          <aside className="flex min-h-[620px] flex-col gap-4">
            <div
              className={[
                "flex flex-1 flex-col rounded-lg border border-neutral-200 bg-white p-5 shadow-sm ring-1 ring-transparent transition sm:p-6",
                styles?.glow ?? "",
                styles?.ring ?? "",
              ].join(" ")}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-medium text-neutral-500">Result</p>
                  <h2 className="mt-2 text-2xl font-semibold">
                    {result ? result.label : "Awaiting image"}
                  </h2>
                </div>
                {result ? (
                  <span
                    className={[
                      "rounded-full border px-3 py-1 text-sm font-semibold",
                      styles?.badge ?? "",
                    ].join(" ")}
                  >
                    {formatPercent(confidence)}
                  </span>
                ) : null}
              </div>

              {error ? (
                <div className="mt-5 rounded-md border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-700">
                  {error}
                </div>
              ) : null}

              {result ? (
                <div className="mt-8 space-y-7">
                  <div>
                    <div className="mb-3 flex items-center justify-between text-sm">
                      <span className="font-medium text-neutral-700">AI probability</span>
                      <span className="font-semibold text-neutral-950">
                        {formatPercent(result.ai_probability)}
                      </span>
                    </div>
                    <div className="h-3 overflow-hidden rounded-full bg-neutral-100">
                      <div className="h-full bg-red-500" style={{ width: aiBarWidth }} />
                    </div>
                  </div>

                  <div>
                    <div className="mb-3 flex items-center justify-between text-sm">
                      <span className="font-medium text-neutral-700">Real probability</span>
                      <span className="font-semibold text-neutral-950">
                        {formatPercent(result.real_probability)}
                      </span>
                    </div>
                    <div className="h-3 overflow-hidden rounded-full bg-neutral-100">
                      <div className="h-full bg-emerald-500" style={{ width: realBarWidth }} />
                    </div>
                  </div>

                  <div className="grid gap-x-8 gap-y-5 border-t border-neutral-100 pt-6 text-sm sm:grid-cols-2">
                    <div>
                      <p className="text-neutral-500">Raw model label</p>
                      <p className="mt-1 font-semibold capitalize text-neutral-950">
                        {result.upstream_label}
                      </p>
                    </div>
                    <div>
                      <p className="text-neutral-500">Model device</p>
                      <p className="mt-1 font-semibold text-neutral-950">{result.model.device}</p>
                    </div>
                    <div>
                      <p className="text-neutral-500">AI threshold</p>
                      <p className="mt-1 font-semibold text-neutral-950">
                        {formatPercent(result.thresholds.ai_threshold)}
                      </p>
                    </div>
                    <div>
                      <p className="text-neutral-500">Uncertainty margin</p>
                      <p className="mt-1 font-semibold text-neutral-950">
                        {formatPercent(result.thresholds.uncertainty_margin)}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-1 flex-col justify-center py-14 text-neutral-500">
                  <div className="h-2 w-full rounded-full bg-neutral-100" />
                  <div className="mt-4 h-2 w-4/5 rounded-full bg-neutral-100" />
                  <div className="mt-4 h-2 w-2/3 rounded-full bg-neutral-100" />
                  <p className="mt-8 max-w-sm text-sm leading-6">No analysis yet.</p>
                </div>
              )}
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}

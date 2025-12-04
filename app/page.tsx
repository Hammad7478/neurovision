"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Image from "next/image";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface PredictionResult {
  prediction: string;
  probabilities: {
    glioma: number;
    meningioma: number;
    pituitary: number;
    notumor: number;
  };
  tumor_detected: boolean;
  gradcam_path?: string;
}

interface TrainingStatus {
  model?: string;
  modelExists: boolean;
  isTraining: boolean;
  progress: number;
  message: string;
  error: string | null;
}

interface ModelMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  macro_f1?: number;
  per_class_f1?: Record<string, number>;
  per_class_metrics?: Record<
    string,
    { precision: number; recall: number; f1_score: number; support: number }
  >;
  train_accuracy?: number[];
  val_accuracy?: number[];
  train_loss?: number[];
  val_loss?: number[];
  confusion_matrix?: number[][];
  classification_report?: Record<string, unknown>;
}

export default function Home() {
  type ModelKey = "resnet50" | "mobilenetv2";

  const modelOptions: { value: ModelKey; label: string; helper: string }[] = [
    { value: "resnet50", label: "ResNet-50 (high accuracy)", helper: "Heavier, best overall accuracy" },
    { value: "mobilenetv2", label: "MobileNetV2 (lightweight)", helper: "Faster, good for constrained devices" },
  ];

  const [selectedModel, setSelectedModel] = useState<ModelKey>("resnet50");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [checkingTraining, setCheckingTraining] = useState(true);

  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [metricsModel, setMetricsModel] = useState<ModelKey>("resnet50");
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [showMetrics, setShowMetrics] = useState(false);

  const loadMetrics = useCallback(async (model: ModelKey) => {
    try {
      setMetricsLoading(true);
      setMetricsError(null);
      const response = await fetch(`/api/model-metrics?model=${model}`, { cache: "no-store" });
      if (!response.ok) {
        if (response.status === 404) {
          // Metrics not available yet (model not trained)
          setMetrics(null);
          setMetricsModel(model);
          setMetricsError(null);
          return;
        }
        const data = await response.json().catch(() => ({}));
        throw new Error(data?.error || "Failed to load model metrics.");
      }
      const data: ModelMetrics = await response.json();
      setMetrics(data);
      setMetricsModel(model);
    } catch (err) {
      console.error("Failed to load model metrics:", err);
      setMetricsError(
        err instanceof Error ? err.message : "Unknown error loading metrics."
      );
      setMetrics(null);
    } finally {
      setMetricsLoading(false);
    }
  }, []);

  useEffect(() => {
    const checkTrainingStatus = async () => {
      try {
        setCheckingTraining(true);
        const response = await fetch(`/api/train-status?model=${selectedModel}`);
        const status: TrainingStatus = await response.json();
        setTrainingStatus(status);
        setCheckingTraining(false);
      } catch (err) {
        console.error("Failed to check training status:", err);
        setCheckingTraining(false);
      }
    };

    checkTrainingStatus();
    const interval = setInterval(checkTrainingStatus, 10000);
    return () => clearInterval(interval);
  }, [selectedModel]);

  useEffect(() => {
    loadMetrics(selectedModel);
  }, [loadMetrics, selectedModel]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select an image file");
      return;
    }

    setFile(selectedFile);
    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(selectedFile);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (!droppedFile) return;

    if (!droppedFile.type.startsWith("image/")) {
      setError("Please drop an image file");
      return;
    }

    setFile(droppedFile);
    setError(null);
    setResult(null);

    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(droppedFile);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleModelChange = (value: ModelKey) => {
    setSelectedModel(value);
    setTrainingStatus(null);
    setMetrics(null);
    setMetricsError(null);
    setResult(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model", selectedModel);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.status === 503 && data.training) {
        setTrainingStatus(data.status);
        setError(null);
        setLoading(false);
        return;
      }

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResult(data);
      await loadMetrics(selectedModel);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const formatProbability = (prob: number) => (prob * 100).toFixed(1);

  const getProbabilityColor = (prob: number) => {
    if (prob > 0.7) {
      return "bg-gradient-to-r from-[#5BC0BE] to-[#6FFFE9]";
    }
    if (prob > 0.4) {
      return "bg-gradient-to-r from-[#3A506B] to-[#5BC0BE]";
    }
    return "bg-gradient-to-r from-[#F97316]/90 to-[#EF4444]/90";
  };

  const formatMetricPercentage = (value?: number) =>
    value === undefined || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(1)}%`;

  const selectedModelLabel = useMemo(
    () =>
      modelOptions.find((opt) => opt.value === selectedModel)?.label ??
      selectedModel,
    [modelOptions, selectedModel]
  );

  const activeMetrics = metricsModel === selectedModel ? metrics : null;

  const summaryCards = useMemo(
    () =>
      activeMetrics
        ? [
            {
              label: "Accuracy",
              value: activeMetrics.accuracy,
              description: "Overall proportion of MRI scans classified correctly.",
            },
            {
              label: "Precision",
              value: activeMetrics.precision,
              description: "How often predicted tumors were actually correct.",
            },
            {
              label: "Recall",
              value: activeMetrics.recall,
              description: "How many tumors present in MRIs the model found.",
            },
            {
              label: "F1 Score",
              value: activeMetrics.f1_score ?? activeMetrics.macro_f1,
              description: "Balance of precision and recall across all classes.",
            },
          ]
        : [],
    [activeMetrics]
  );

  const classLabels = useMemo(
    () => ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
    []
  );

  const confusionMatrix = activeMetrics?.confusion_matrix ?? [];

  const predictedClassIndex = useMemo(() => {
    if (!result) return -1;
    const prediction = result.prediction?.toLowerCase();
    return classLabels.findIndex(
      (label) => label.toLowerCase() === prediction
    );
  }, [result, classLabels]);

  const maxConfusionValue = useMemo(() => {
    let max = 0;
    for (const row of confusionMatrix) {
      for (const value of row) {
        if (typeof value === "number" && value > max) {
          max = value;
        }
      }
    }
    return max;
  }, [confusionMatrix]);

  const accuracyChartData = useMemo(() => {
    const train = activeMetrics?.train_accuracy ?? [];
    const val = activeMetrics?.val_accuracy ?? [];
    const length = Math.max(train.length, val.length);
    if (length === 0) return null;

    const labels = Array.from({ length }, (_, idx) => `Epoch ${idx + 1}`);
    const toPoint = (arr: number[], idx: number) =>
      arr[idx] !== undefined ? Number(arr[idx]) : null;

    return {
      labels,
      datasets: [
        {
          label: "Training Accuracy",
          data: labels.map((_, idx) => toPoint(train, idx)),
          borderColor: "#5BC0BE",
          backgroundColor: "rgba(91, 192, 190, 0.25)",
          pointBackgroundColor: "#6FFFE9",
          pointBorderColor: "#0B132B",
          pointRadius: 4,
          borderWidth: 3,
          tension: 0.4,
        },
        {
          label: "Validation Accuracy",
          data: labels.map((_, idx) => toPoint(val, idx)),
          borderColor: "#F2A541",
          backgroundColor: "rgba(242, 165, 65, 0.2)",
          pointBackgroundColor: "#F2A541",
          pointBorderColor: "#0B132B",
          pointRadius: 4,
          borderWidth: 3,
          tension: 0.4,
        },
      ],
    };
  }, [activeMetrics]);

  const lossChartData = useMemo(() => {
    const train = activeMetrics?.train_loss ?? [];
    const val = activeMetrics?.val_loss ?? [];
    const length = Math.max(train.length, val.length);
    if (length === 0) return null;

    const labels = Array.from({ length }, (_, idx) => `Epoch ${idx + 1}`);
    const toPoint = (arr: number[], idx: number) =>
      arr[idx] !== undefined ? Number(arr[idx]) : null;

    return {
      labels,
      datasets: [
        {
          label: "Training Loss",
          data: labels.map((_, idx) => toPoint(train, idx)),
          borderColor: "#E15A97",
          backgroundColor: "rgba(225, 90, 151, 0.25)",
          pointBackgroundColor: "#E15A97",
          pointBorderColor: "#0B132B",
          pointRadius: 4,
          borderWidth: 3,
          tension: 0.4,
        },
        {
          label: "Validation Loss",
          data: labels.map((_, idx) => toPoint(val, idx)),
          borderColor: "#3D8BEB",
          backgroundColor: "rgba(61, 139, 235, 0.22)",
          pointBackgroundColor: "#3D8BEB",
          pointBorderColor: "#0B132B",
          pointRadius: 4,
          borderWidth: 3,
          tension: 0.4,
        },
      ],
    };
  }, [activeMetrics]);

  const lineChartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "#9AA9D8",
            font: {
              family: "var(--font-poppins, 'DM Sans', system-ui, sans-serif)",
              weight: 500,
            },
          },
        },
        tooltip: {
          backgroundColor: "rgba(20, 32, 53, 0.92)",
          borderColor: "#5BC0BE",
          borderWidth: 1,
          titleColor: "#E6F1FF",
          bodyColor: "#D5E5FF",
        },
      },
      scales: {
        x: {
          ticks: { color: "#7C8FB8", font: { family: "var(--font-poppins)" } },
          grid: { color: "rgba(76, 103, 148, 0.25)" },
        },
        y: {
          ticks: { color: "#7C8FB8", font: { family: "var(--font-poppins)" } },
          grid: { color: "rgba(76, 103, 148, 0.25)" },
        },
      },
    }),
    []
  );

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoading(false);

    const fileInput = document.getElementById("file-upload") as HTMLInputElement;
    if (fileInput) {
      fileInput.value = "";
    }
  };

  const handleToggleMetrics = useCallback(() => {
    setShowMetrics((prev) => {
      const next = !prev;
      if (!prev && !metricsLoading) {
        loadMetrics(selectedModel);
      }
      return next;
    });
  }, [loadMetrics, metricsLoading, selectedModel]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#0B132B] text-[#E6F1FF]">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(111,255,233,0.18),transparent_55%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_bottom,_rgba(91,192,190,0.17),transparent_55%)]" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#0B132B]/65 to-[#0B132B]" />
      </div>

      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-10 px-4 py-12">
        {/* Header */}
        <header className="text-center space-y-3">
          <p className="text-[0.65rem] uppercase tracking-[0.65em] text-[#6FFFE9]/70">
            AI-Assisted Diagnostics
          </p>
          <h1 className="text-4xl sm:text-5xl font-semibold uppercase tracking-[0.38em] text-[#6FFFE9]">
            NeuroVision
          </h1>
          <p className="mx-auto max-w-2xl text-sm sm:text-base text-[#9AA9D8]">
            Upload a brain MRI scan to detect and classify tumor types in seconds. Powered by transfer learning and
            interpretable Grad-CAM visualisations.
          </p>
          <div className="mx-auto mt-4 h-[2px] w-48 rounded-full bg-gradient-to-r from-[#5BC0BE] via-[#6FFFE9] to-transparent animate-pulse" />
        </header>

        {/* Upload Section */}
        <div className="rounded-2xl border border-white/10 bg-[#1C2541]/80 p-8 shadow-[0_30px_80px_-35px_rgba(9,20,45,0.9)] backdrop-blur-2xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {modelOptions.map((opt) => {
                const isActive = selectedModel === opt.value;
                return (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => handleModelChange(opt.value)}
                    className={`flex flex-col items-start rounded-xl border px-4 py-3 text-left transition-all duration-300 ${
                      isActive
                        ? "border-[#6FFFE9]/80 bg-[#0f1b36] shadow-[0_12px_35px_-18px_rgba(111,255,233,0.45)]"
                        : "border-white/10 bg-white/5 hover:border-[#6FFFE9]/50 hover:-translate-y-0.5"
                    }`}
                  >
                    <span className="text-sm font-semibold text-[#E6F1FF]">
                      {opt.label}
                    </span>
                    <span className="text-xs text-[#9AA9D8]">{opt.helper}</span>
                    {isActive && (
                      <span className="mt-2 rounded-md bg-[#6FFFE9]/15 px-2 py-1 text-[0.7rem] font-semibold uppercase tracking-[0.2em] text-[#6FFFE9]">
                        Active
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className={`group relative overflow-hidden rounded-2xl border border-dashed border-white/15 bg-white/5 p-10 text-center transition-all duration-300 hover:border-[#6FFFE9] hover:bg-white/10 ${
                file ? "border-[#5BC0BE]" : ""
              }`}
            >
              <input
                type="file"
                id="file-upload"
                accept="image/jpeg,image/jpg,image/png"
                onChange={handleFileChange}
                className="hidden"
              />
              <label
                htmlFor="file-upload"
                className="flex cursor-pointer flex-col items-center gap-3 text-[#C0CEFF]"
              >
                {preview ? (
                  <div className="relative mx-auto mb-4 h-64 w-64 drop-shadow-[0_45px_65px_rgba(20,40,75,0.45)]">
                    <Image
                      src={preview}
                      alt="Preview"
                      fill
                      className="rounded-2xl border border-white/10 object-contain"
                    />
                  </div>
                ) : (
                  <>
                    <svg
                      className="mx-auto mb-4 h-16 w-16 text-[#6FFFE9]"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                    <p className="mb-1 text-sm font-medium tracking-wide text-[#E3ECFF]">
                      Drag &amp; drop or click to upload
                    </p>
                    <p className="text-xs text-[#9AA9D8]">
                      Accepts PNG or JPG. Max file size 10 MB.
                    </p>
                  </>
                )}
              </label>
              {file && (
                <p className="mt-3 text-sm text-[#9AA9D8]">
                  Selected: {file.name}
                </p>
              )}
            </div>

            <div className="flex gap-3">
              <button
                type="submit"
                disabled={!file || loading || (trainingStatus !== null && !trainingStatus.modelExists)}
                className={`flex-1 rounded-xl px-6 py-3 font-semibold transition-all duration-300 ${
                  !file || loading || (trainingStatus !== null && !trainingStatus.modelExists)
                    ? "cursor-not-allowed bg-slate-500/30 text-slate-400"
                    : "cursor-pointer bg-gradient-to-r from-[#4E7A95] via-[#5BA2B4] to-[#4A8BA2] text-[#E6F1FF] shadow-[0_10px_24px_rgba(76,128,146,0.35)] hover:shadow-[0_16px_32px_rgba(90,170,180,0.42)]"
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg
                      className="h-5 w-5 animate-spin"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                    <span>Classifying...</span>
                  </span>
                ) : trainingStatus !== null && !trainingStatus.modelExists ? (
                  "Waiting for model..."
                ) : (
                  "Classify MRI"
                )}
              </button>
              {(file || result) && (
                <button
                  type="button"
                  onClick={handleReset}
                  disabled={loading}
                  className="rounded-xl border border-[#5BC0BE]/60 px-6 py-3 font-semibold text-[#6FFFE9] transition-all duration-300 hover:bg-[#5BC0BE]/15 hover:text-[#E6F1FF] disabled:cursor-not-allowed disabled:border-slate-600 disabled:text-slate-500"
                >
                  Reset
                </button>
              )}
            </div>
          </form>

          {trainingStatus?.isTraining && (
            <div className="mt-5 rounded-2xl border border-[#5BC0BE]/30 bg-[#14213C]/75 p-4 text-[#C6E7FF] shadow-inner">
              <div className="flex items-center space-x-3">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-[#6FFFE9] border-t-transparent"></div>
                <div>
                  <p className="font-medium tracking-wide">
                    Training {selectedModelLabel} in progress...
                  </p>
                  <p className="mt-1 text-sm text-[#9AA9D8]">
                    {trainingStatus.message ||
                      "Training will stop automatically when validation accuracy reaches 95%. This typically takes 5-15 minutes. The page will automatically refresh when training completes."}
                  </p>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-5 rounded-2xl border border-red-500/40 bg-[#3A203C]/75 p-4 text-red-100 shadow-inner">
              {error}
            </div>
          )}

          {trainingStatus && !trainingStatus.isTraining && trainingStatus.modelExists && !result && (
            <div className="mt-5 rounded-2xl border border-[#6FFFE9]/30 bg-[#16324F]/75 p-4 text-[#E1FBFF] shadow-inner">
              <p className="font-semibold tracking-wide text-[#6FFFE9]">{selectedModelLabel} is ready!</p>
              <p className="mt-1 text-sm text-[#A8C6E7]">
                Training completed. You can now upload an image to get predictions.
              </p>
            </div>
          )}
        </div>

        {/* Prediction Results */}
        {result && (
          <div className="animate-[fadeIn_0.6s_ease-out] space-y-6 rounded-2xl border border-white/10 bg-[#1C2541]/85 p-6 shadow-[0_30px_80px_-40px_rgba(8,20,43,0.9)] backdrop-blur-xl sm:p-8">
            <h2 className="text-2xl font-semibold tracking-wide text-[#E6F1FF]">
              Prediction Results
            </h2>

            <div className="flex items-center space-x-4">
              <span className="text-lg font-medium text-[#A8C6E7]">
                Tumor Detected:
              </span>
              <span
                className={`rounded-xl px-4 py-2 text-sm font-semibold tracking-wide shadow-inner transition-transform duration-300 ${
                  result.tumor_detected
                    ? "bg-gradient-to-r from-[#F97316]/60 to-[#EF4444]/80 text-[#FFD5D5]"
                    : "bg-gradient-to-r from-[#5BC0BE]/40 to-[#6FFFE9]/70 text-[#0B132B]"
                }`}
              >
                {result.tumor_detected ? "Yes" : "No"}
              </span>
            </div>

            <div className="flex items-center space-x-4">
              <span className="text-lg font-medium text-[#A8C6E7]">
                Type:
              </span>
              <span className="rounded-xl bg-white/10 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-[#6FFFE9] shadow-inner">
                {result.prediction}
              </span>
            </div>

            <div>
              <h3 className="mb-4 text-lg font-medium text-[#E6F1FF]">
                Class Probabilities
              </h3>
              <div className="space-y-3">
                {Object.entries(result.probabilities).map(([className, prob]) => (
                  <div key={className} className="transition-transform duration-300 hover:-translate-y-0.5">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-sm font-medium capitalize text-[#A8C6E7]">
                        {className}
                      </span>
                      <span className="text-sm font-semibold text-[#E6F1FF]">
                        {formatProbability(prob)}%
                      </span>
                    </div>
                    <div className="h-3 w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className={`h-3 rounded-full transition-all duration-500 ${getProbabilityColor(prob)}`}
                        style={{ width: `${formatProbability(prob)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {result.gradcam_path && (
              <div>
                <h3 className="mb-4 text-lg font-medium text-[#E6F1FF]">
                  Grad-CAM Visualization
                </h3>
                <div className="relative mx-auto aspect-square w-full max-w-2xl rounded-2xl border border-white/10 bg-[#111A33]/70 p-4 shadow-inner">
                  <Image
                    src={`/api/serve-gradcam/${result.gradcam_path.split("/").pop() || result.gradcam_path}`}
                    alt="Grad-CAM Visualization"
                    fill
                    className="rounded-xl object-contain"
                    unoptimized
                  />
                </div>
                <p className="mt-2 text-center text-sm text-[#9AA9D8]">
                  Heatmap shows regions the model focused on for prediction
                </p>
              </div>
            )}
          </div>
        )}

        {/* Model Performance Section */}
        <div className="rounded-2xl border border-white/10 bg-[#1C2541]/85 p-6 shadow-[0_30px_80px_-40px_rgba(8,20,43,0.9)] backdrop-blur-xl sm:p-8">
          <button
            type="button"
            onClick={handleToggleMetrics}
            className="flex w-full items-center justify-between rounded-xl bg-white/5 px-4 py-3 text-left text-[#E6F1FF] ring-1 ring-white/5 transition-all duration-300 hover:-translate-y-0.5 hover:bg-white/8 hover:ring-[#5BC0BE]/40 cursor-pointer"
          >
            <div>
              <h2 className="text-2xl font-semibold tracking-wide text-[#E6F1FF]">
                {selectedModelLabel} Performance Metrics
              </h2>
              <p className="mt-1 text-sm text-[#9AA9D8]">
                Explore how the model performed during training and evaluation (metrics update live as you train the model).
              </p>
            </div>
            <svg
              className={`w-5 h-5 text-slate-500 dark:text-slate-300 transition-transform ${
                showMetrics ? "rotate-180" : ""
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showMetrics && (
            <div className="mt-6 space-y-6">
              {metricsLoading ? (
                <div className="flex items-center justify-center gap-3 py-8 text-slate-500 dark:text-slate-300">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 dark:border-blue-400"></div>
                  Loading latest metrics...
                </div>
              ) : metricsError ? (
                <div className="rounded-lg border border-red-200 dark:border-red-700 bg-red-50 dark:bg-red-900/30 px-4 py-3 text-sm text-red-700 dark:text-red-300">
                  {metricsError}
                </div>
              ) : activeMetrics ? (
                <>
                  {summaryCards.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                      {summaryCards.map((card) => (
                        <div
                          key={card.label}
                      title={card.description}
                      className="rounded-2xl border border-white/10 bg-[#131E36]/80 p-4 shadow-[0_25px_45px_-30px_rgba(10,20,40,0.8)] transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_35px_65px_-25px_rgba(111,255,233,0.35)]"
                        >
                      <p className="text-xs uppercase tracking-[0.45em] text-[#6FFFE9]/80">
                            {card.label}
                          </p>
                      <p className="mt-2 text-3xl font-semibold text-[#E6F1FF]">
                            {formatMetricPercentage(card.value)}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}

                  {confusionMatrix.length > 0 && (
                <div className="rounded-2xl border border-white/10 bg-[#141F35]/85 p-5 shadow-[0_25px_60px_-30px_rgba(10,20,43,0.8)]">
                  <h3 className="mb-4 text-lg font-semibold text-[#E6F1FF]">
                        Confusion Matrix
                      </h3>
                      <div className="overflow-x-auto">
                    <table className="min-w-full border-collapse text-sm">
                          <thead>
                            <tr>
                          <th className="p-2 text-xs uppercase tracking-[0.35em] text-[#6FFFE9]/70 text-left">
                                True \ Pred
                              </th>
                              {classLabels.map((label) => (
                                <th
                                  key={`cm-head-${label}`}
                              className="p-2 text-xs font-medium uppercase tracking-[0.3em] text-[#9AA9D8] text-center"
                                >
                                  {label}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {confusionMatrix.map((row, rowIdx) => {
                              const isPredictedRow = rowIdx === predictedClassIndex;
                              return (
                                <tr key={`cm-row-${rowIdx}`}>
                                  <th
                                  className={`p-2 text-sm text-left font-semibold tracking-wide ${
                                      isPredictedRow
                                      ? "text-[#6FFFE9]"
                                      : "text-[#A8C6E7]"
                                    }`}
                                  >
                                    {classLabels[rowIdx] ?? `Class ${rowIdx + 1}`}
                                  </th>
                                  {row.map((value, colIdx) => {
                                    const ratio = maxConfusionValue === 0 ? 0 : value / maxConfusionValue;
                                  const backgroundColor = `rgba(91, 192, 190, ${0.12 + ratio * 0.6})`;
                                    const textClass =
                                    ratio > 0.45 ? "text-[#0B132B]" : "text-[#E6F1FF]";
                                    const isPredictedCol = colIdx === predictedClassIndex;
                                    const highlightClass =
                                      isPredictedRow && isPredictedCol
                                      ? "ring-2 ring-[#6FFFE9]"
                                        : isPredictedRow || isPredictedCol
                                      ? "ring-1 ring-[#5BC0BE]/60"
                                        : "";
                                    return (
                                      <td
                                        key={`cm-cell-${rowIdx}-${colIdx}`}
                                      className={`p-2 text-center font-semibold transition-transform duration-200 ${textClass} ${highlightClass}`}
                                        style={{ backgroundColor }}
                                      >
                                        {value}
                                      </td>
                                    );
                                  })}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {(accuracyChartData || lossChartData) && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {accuracyChartData && (
                    <div className="h-72 rounded-2xl border border-white/10 bg-[#141F35]/85 p-5 shadow-[0_25px_60px_-30px_rgba(10,20,43,0.8)]">
                      <h3 className="mb-4 text-lg font-semibold text-[#E6F1FF]">
                            Accuracy Over Epochs
                          </h3>
                      <div className="h-56">
                            <Line data={accuracyChartData} options={lineChartOptions} />
                          </div>
                        </div>
                      )}
                      {lossChartData && (
                    <div className="h-72 rounded-2xl border border-white/10 bg-[#141F35]/85 p-5 shadow-[0_25px_60px_-30px_rgba(10,20,43,0.8)]">
                      <h3 className="mb-4 text-lg font-semibold text-[#E6F1FF]">
                            Loss Over Epochs
                          </h3>
                      <div className="h-56">
                            <Line data={lossChartData} options={lineChartOptions} />
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {activeMetrics?.per_class_f1 && (
                    <div>
                  <h3 className="mb-4 text-lg font-semibold text-[#E6F1FF]">
                        Per-Class F1 Scores
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        {Object.entries(activeMetrics.per_class_f1).map(([cls, value]) => (
                          <div
                            key={`per-class-${cls}`}
                        className="rounded-2xl border border-white/10 bg-[#131F32]/80 p-4 shadow-[0_20px_45px_-25px_rgba(10,30,57,0.8)] transition-transform duration-300 hover:-translate-y-1 hover:shadow-[0_30px_60px_-28px_rgba(91,192,190,0.45)]"
                          >
                        <p className="text-xs uppercase tracking-[0.3em] text-[#5BC0BE]/80">
                              {cls}
                            </p>
                        <p className="mt-3 text-2xl font-semibold text-[#E6F1FF]">
                              {formatMetricPercentage(value)}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-sm text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-6 bg-slate-50 dark:bg-slate-900/30">
                  Train the model to generate performance metrics.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}



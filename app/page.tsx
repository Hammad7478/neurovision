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
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [checkingTraining, setCheckingTraining] = useState(true);

  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [showMetrics, setShowMetrics] = useState(false);

  const loadMetrics = useCallback(async () => {
    try {
      setMetricsLoading(true);
      setMetricsError(null);
      const response = await fetch("/api/model-metrics", { cache: "no-store" });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data?.error || "Failed to load model metrics.");
      }
      const data: ModelMetrics = await response.json();
      setMetrics(data);
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
        const response = await fetch("/api/train-status");
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
  }, []);

  useEffect(() => {
    loadMetrics();
  }, [loadMetrics]);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

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
      await loadMetrics();
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const formatProbability = (prob: number) => (prob * 100).toFixed(1);

  const getProbabilityColor = (prob: number) => {
    if (prob > 0.7) return "bg-green-500";
    if (prob > 0.4) return "bg-yellow-500";
    return "bg-red-500";
  };

  const formatMetricPercentage = (value?: number) =>
    value === undefined || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(1)}%`;

  const summaryCards = useMemo(
    () =>
      metrics
        ? [
            { label: "Accuracy", value: metrics.accuracy },
            { label: "Precision", value: metrics.precision },
            { label: "Recall", value: metrics.recall },
            { label: "F1 Score", value: metrics.f1_score ?? metrics.macro_f1 },
          ]
        : [],
    [metrics]
  );

  const classLabels = useMemo(
    () => ["Glioma", "Meningioma", "Pituitary", "No Tumor"],
    []
  );

  const confusionMatrix = metrics?.confusion_matrix ?? [];

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
    const train = metrics?.train_accuracy ?? [];
    const val = metrics?.val_accuracy ?? [];
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
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.15)",
          tension: 0.3,
        },
        {
          label: "Validation Accuracy",
          data: labels.map((_, idx) => toPoint(val, idx)),
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.15)",
          tension: 0.3,
        },
      ],
    };
  }, [metrics]);

  const lossChartData = useMemo(() => {
    const train = metrics?.train_loss ?? [];
    const val = metrics?.val_loss ?? [];
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
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.15)",
          tension: 0.3,
        },
        {
          label: "Validation Loss",
          data: labels.map((_, idx) => toPoint(val, idx)),
          borderColor: "#ec4899",
          backgroundColor: "rgba(236, 72, 153, 0.15)",
          tension: 0.3,
        },
      ],
    };
  }, [metrics]);

  const lineChartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "#64748b",
            font: {
              family: "var(--font-geist-sans, Inter, system-ui, sans-serif)",
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148, 163, 184, 0.2)" },
        },
        y: {
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148, 163, 184, 0.2)" },
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
        loadMetrics();
      }
      return next;
    });
  }, [loadMetrics, metricsLoading]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
            NeuroVision
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-300">
            Brain MRI Tumor Classification using Deep Learning
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6 mb-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                file
                  ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                  : "border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500"
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
                className="cursor-pointer flex flex-col items-center"
              >
                {preview ? (
                  <div className="relative w-64 h-64 mx-auto mb-4">
                    <Image
                      src={preview}
                      alt="Preview"
                      fill
                      className="object-contain rounded-lg"
                    />
                  </div>
                ) : (
                  <>
                    <svg
                      className="w-16 h-16 mx-auto mb-4 text-slate-400"
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
                    <p className="text-slate-600 dark:text-slate-300 mb-2">
                      Click to upload or drag and drop
                    </p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      PNG, JPG (Max 10MB)
                    </p>
                  </>
                )}
              </label>
              {file && (
                <p className="text-sm text-slate-600 dark:text-slate-300 mt-2">
                  Selected: {file.name}
                </p>
              )}
            </div>

            <div className="flex gap-3">
              <button
                type="submit"
                disabled={!file || loading || (trainingStatus !== null && !trainingStatus.modelExists)}
                className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all ${
                  !file || loading || (trainingStatus !== null && !trainingStatus.modelExists)
                    ? "bg-slate-300 dark:bg-slate-700 text-slate-500 dark:text-slate-400 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 text-white hover:shadow-lg"
                }`}
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg
                      className="w-5 h-5 animate-spin"
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
                  className="px-6 py-3 rounded-lg font-semibold transition-colors bg-slate-200 hover:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Reset
                </button>
              )}
            </div>
          </form>

          {trainingStatus?.isTraining && (
            <div className="mt-4 p-4 bg-blue-100 dark:bg-blue-900/30 border border-blue-400 dark:border-blue-600 text-blue-700 dark:text-blue-300 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-700 dark:border-blue-300"></div>
                <div>
                  <p className="font-semibold">Training model in progress...</p>
                  <p className="text-sm mt-1">
                    {trainingStatus.message ||
                      "Training will stop automatically when validation accuracy reaches 95%. This typically takes 5-15 minutes. The page will automatically refresh when training completes."}
                  </p>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-600 text-red-700 dark:text-red-300 rounded-lg">
              {error}
            </div>
          )}

          {trainingStatus && !trainingStatus.isTraining && trainingStatus.modelExists && !result && (
            <div className="mt-4 p-4 bg-green-100 dark:bg-green-900/30 border border-green-400 dark:border-green-600 text-green-700 dark:text-green-300 rounded-lg">
              <p className="font-semibold">Model is ready!</p>
              <p className="text-sm mt-1">
                Training completed (stopped at 95% validation accuracy). You can now upload an image to get predictions.
              </p>
            </div>
          )}
        </div>

        {/* Prediction Results */}
        {result && (
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6 space-y-6 animate-fadeIn">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
              Prediction Results
            </h2>

            <div className="flex items-center space-x-4">
              <span className="text-lg font-semibold text-slate-700 dark:text-slate-300">
                Tumor Detected:
              </span>
              <span
                className={`px-4 py-2 rounded-lg font-bold ${
                  result.tumor_detected
                    ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                    : "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                }`}
              >
                {result.tumor_detected ? "Yes" : "No"}
              </span>
            </div>

            <div className="flex items-center space-x-4">
              <span className="text-lg font-semibold text-slate-700 dark:text-slate-300">
                Type:
              </span>
              <span className="px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-lg font-semibold capitalize">
                {result.prediction}
              </span>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-4">
                Class Probabilities
              </h3>
              <div className="space-y-3">
                {Object.entries(result.probabilities).map(([className, prob]) => (
                  <div key={className}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm font-medium text-slate-600 dark:text-slate-400 capitalize">
                        {className}
                      </span>
                      <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        {formatProbability(prob)}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3">
                      <div
                        className={`h-3 rounded-full transition-all ${getProbabilityColor(prob)}`}
                        style={{ width: `${formatProbability(prob)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {result.gradcam_path && (
              <div>
                <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-300 mb-4">
                  Grad-CAM Visualization
                </h3>
                <div className="relative w-full max-w-2xl mx-auto aspect-square">
                  <Image
                    src={`/api/serve-gradcam/${result.gradcam_path.split("/").pop() || result.gradcam_path}`}
                    alt="Grad-CAM Visualization"
                    fill
                    className="object-contain rounded-lg"
                    unoptimized
                  />
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-2 text-center">
                  Heatmap shows regions the model focused on for prediction
                </p>
              </div>
            )}
          </div>
        )}

        {/* Model Performance Section */}
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6 mt-6">
          <button
            type="button"
            onClick={handleToggleMetrics}
            className="flex w-full items-center justify-between text-left"
          >
            <div>
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
                Model Performance Metrics
              </h2>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                Explore how the model performed during training and evaluation.
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
              ) : metrics ? (
                <>
                  {summaryCards.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                      {summaryCards.map((card) => (
                        <div
                          key={card.label}
                          className="bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 rounded-lg p-4 shadow-sm"
                        >
                          <p className="text-sm uppercase tracking-wide text-slate-500 dark:text-slate-400">
                            {card.label}
                          </p>
                          <p className="text-2xl font-semibold text-slate-900 dark:text-white mt-1">
                            {formatMetricPercentage(card.value)}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}

                  {confusionMatrix.length > 0 && (
                    <div className="bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 rounded-lg p-4 shadow-sm">
                      <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-4">
                        Confusion Matrix
                      </h3>
                      <div className="overflow-x-auto">
                        <table className="min-w-full border-collapse">
                          <thead>
                            <tr>
                              <th className="p-2 text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400 text-left">
                                True \ Pred
                              </th>
                              {classLabels.map((label) => (
                                <th
                                  key={`cm-head-${label}`}
                                  className="p-2 text-xs font-medium text-slate-500 dark:text-slate-400 text-center"
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
                                    className={`p-2 text-sm text-left font-semibold ${
                                      isPredictedRow
                                        ? "text-blue-600 dark:text-blue-300"
                                        : "text-slate-600 dark:text-slate-300"
                                    }`}
                                  >
                                    {classLabels[rowIdx] ?? `Class ${rowIdx + 1}`}
                                  </th>
                                  {row.map((value, colIdx) => {
                                    const ratio = maxConfusionValue === 0 ? 0 : value / maxConfusionValue;
                                    const backgroundColor = `rgba(59, 130, 246, ${0.15 + ratio * 0.65})`;
                                    const textClass =
                                      ratio > 0.45 ? "text-white" : "text-slate-900 dark:text-slate-100";
                                    const isPredictedCol = colIdx === predictedClassIndex;
                                    const highlightClass =
                                      isPredictedRow && isPredictedCol
                                        ? "ring-2 ring-blue-500"
                                        : isPredictedRow || isPredictedCol
                                        ? "ring-1 ring-blue-300/70"
                                        : "";
                                    return (
                                      <td
                                        key={`cm-cell-${rowIdx}-${colIdx}`}
                                        className={`p-2 text-center font-semibold ${textClass} ${highlightClass}`}
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
                        <div className="bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 rounded-lg p-4 shadow-sm h-72">
                          <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-4">
                            Accuracy Over Epochs
                          </h3>
                          <div className="h-56">
                            <Line data={accuracyChartData} options={lineChartOptions} />
                          </div>
                        </div>
                      )}
                      {lossChartData && (
                        <div className="bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 rounded-lg p-4 shadow-sm h-72">
                          <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-4">
                            Loss Over Epochs
                          </h3>
                          <div className="h-56">
                            <Line data={lossChartData} options={lineChartOptions} />
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {metrics.per_class_f1 && (
                    <div>
                      <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-4">
                        Per-Class F1 Scores
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        {Object.entries(metrics.per_class_f1).map(([cls, value]) => (
                          <div
                            key={`per-class-${cls}`}
                            className="bg-white dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 rounded-lg p-4 shadow-sm"
                          >
                            <p className="text-sm text-slate-500 dark:text-slate-400 capitalize">
                              {cls}
                            </p>
                            <p className="text-xl font-semibold text-slate-900 dark:text-white mt-1">
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



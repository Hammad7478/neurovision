"use client";

import { useState, useEffect } from "react";
import Image from "next/image";

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

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [checkingTraining, setCheckingTraining] = useState(true);

  // Check training status on mount and periodically
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
    
    // Poll every 10 seconds if training is in progress or model doesn't exist
    const interval = setInterval(() => {
      checkTrainingStatus();
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      // Validate file type
      if (!selectedFile.type.startsWith("image/")) {
        setError("Please select an image file");
        return;
      }

      setFile(selectedFile);
      setError(null);
      setResult(null);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      setFile(droppedFile);
      setError(null);
      setResult(null);

      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(droppedFile);
    } else {
      setError("Please drop an image file");
    }
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

      // Handle automatic training response
      if (response.status === 503 && data.training) {
        setTrainingStatus(data.status);
        setError(null); // Clear error to show training message instead
        setLoading(false);
        return;
      }

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const formatProbability = (prob: number) => {
    return (prob * 100).toFixed(1);
  };

  const getProbabilityColor = (prob: number) => {
    if (prob > 0.7) return "bg-green-500";
    if (prob > 0.4) return "bg-yellow-500";
    return "bg-red-500";
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoading(false);
    // Reset file input
    const fileInput = document.getElementById("file-upload") as HTMLInputElement;
    if (fileInput) {
      fileInput.value = "";
    }
  };


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
            {/* File Dropzone */}
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

            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                type="submit"
                disabled={!file || loading || (trainingStatus !== null && !trainingStatus.modelExists)}
                className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all relative overflow-hidden ${
                  !file || loading || (trainingStatus !== null && !trainingStatus.modelExists)
                    ? "bg-slate-300 dark:bg-slate-700 text-slate-500 dark:text-slate-400 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 text-white hover:shadow-lg"
                } ${loading ? "animate-pulse" : ""}`}
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
                    <span className="flex items-center">
                      Classifying
                      <span className="inline-flex gap-1 ml-1">
                        <span className="animate-[bounce_1s_ease-in-out_infinite]">.</span>
                        <span className="animate-[bounce_1s_ease-in-out_0.2s_infinite]">.</span>
                        <span className="animate-[bounce_1s_ease-in-out_0.4s_infinite]">.</span>
                      </span>
                    </span>
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

          {/* Training Status */}
          {trainingStatus?.isTraining && (
            <div className="mt-4 p-4 bg-blue-100 dark:bg-blue-900/30 border border-blue-400 dark:border-blue-600 text-blue-700 dark:text-blue-300 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-700 dark:border-blue-300"></div>
                <div>
                  <p className="font-semibold">Training model in progress...</p>
                  <p className="text-sm mt-1">
                    {trainingStatus.message || "Training will stop automatically when validation accuracy reaches 90%. This typically takes 5-15 minutes. The page will automatically refresh when training completes."}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-600 text-red-700 dark:text-red-300 rounded-lg">
              {error}
            </div>
          )}

          {/* Training Complete Message */}
          {trainingStatus && !trainingStatus.isTraining && trainingStatus.modelExists && !result && (
            <div className="mt-4 p-4 bg-green-100 dark:bg-green-900/30 border border-green-400 dark:border-green-600 text-green-700 dark:text-green-300 rounded-lg">
              <p className="font-semibold">Model is ready!</p>
              <p className="text-sm mt-1">
                Training completed (stopped at 90% validation accuracy). You can now upload an image to get predictions.
              </p>
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6 space-y-6 animate-fadeIn">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
              Prediction Results
            </h2>

            {/* Tumor Detection */}
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

            {/* Predicted Type */}
            <div className="flex items-center space-x-4">
              <span className="text-lg font-semibold text-slate-700 dark:text-slate-300">
                Type:
              </span>
              <span className="px-4 py-2 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-lg font-semibold capitalize">
                {result.prediction}
              </span>
            </div>

            {/* Probabilities */}
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
                        className={`h-3 rounded-full transition-all ${getProbabilityColor(
                          prob
                        )}`}
                        style={{ width: `${formatProbability(prob)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Grad-CAM Visualization */}
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
      </div>
    </div>
  );
}

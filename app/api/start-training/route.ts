import { NextResponse } from "next/server";
import { spawn } from "child_process";
import { join } from "path";
import { existsSync } from "fs";
import { getTrainingStatus, setTrainingStatus } from "../../../lib/training-status";

export const runtime = "nodejs";
export const maxDuration = 300; // 5 minutes for starting training

export async function POST() {
  try {
    // Check if model already exists
    const modelPath = join(process.cwd(), "model", "model.h5");
    if (existsSync(modelPath)) {
      return NextResponse.json({
        success: true,
        message: "Model already exists",
      });
    }

    // Check if training is already in progress
    const status = getTrainingStatus();
    if (status.isTraining) {
      return NextResponse.json({
        success: true,
        message: "Training already in progress",
        status,
      });
    }

    // Start training in background
    const trainScriptPath = join(process.cwd(), "ml", "train_model.py");
    const pythonCmd = process.platform === "win32" ? "python" : "python3";

    setTrainingStatus({
      isTraining: true,
      progress: 0,
      message: "Training started. Will stop automatically at 95% validation accuracy...",
      error: null,
    });

    // Spawn training process (don't wait for it)
    const trainProcess = spawn(pythonCmd, [trainScriptPath], {
      detached: true,
      stdio: "ignore",
    });

    // Unref so parent process can exit independently
    trainProcess.unref();

    // Monitor training process
    trainProcess.on("error", (error) => {
      setTrainingStatus({
        isTraining: false,
        progress: 0,
        message: "Training failed to start",
        error: error.message,
      });
    });

    trainProcess.on("exit", (code) => {
      if (code === 0) {
        setTrainingStatus({
          isTraining: false,
          progress: 100,
          message: "Training completed successfully! (Stopped at 95% validation accuracy)",
          error: null,
        });
      } else {
        setTrainingStatus({
          isTraining: false,
          progress: 0,
          message: "Training failed",
          error: `Process exited with code ${code}`,
        });
      }
    });

    return NextResponse.json({
      success: true,
      message: "Training started in background",
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}


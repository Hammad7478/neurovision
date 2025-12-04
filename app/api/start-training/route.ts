import { NextResponse } from "next/server";
import { spawn } from "child_process";
import { join } from "path";
import { existsSync } from "fs";
import { getTrainingStatus, setTrainingStatus } from "../../../lib/training-status";

export const runtime = "nodejs";
export const maxDuration = 300; // 5 minutes for starting training

type ModelKey = "resnet50" | "mobilenetv2";

const MODEL_CONFIG: Record<
  ModelKey,
  { modelPath: string; trainScriptPath: string; legacyPaths?: string[] }
> = {
  resnet50: {
    modelPath: join(process.cwd(), "model", "resnet50_model.h5"),
    legacyPaths: [join(process.cwd(), "model", "model.h5")],
    trainScriptPath: join(process.cwd(), "ml", "train_resnet50.py"),
  },
  mobilenetv2: {
    modelPath: join(process.cwd(), "model", "mobilenetv2_model.h5"),
    trainScriptPath: join(process.cwd(), "ml", "train_mobilenetv2.py"),
  },
};

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const model = (body?.model as ModelKey) ?? "resnet50";
    const config = MODEL_CONFIG[model];

    if (!config) {
      return NextResponse.json(
        { success: false, message: "Unsupported model selection." },
        { status: 400 }
      );
    }

    // Check if model already exists
    const existingPath = [config.modelPath, ...(config.legacyPaths ?? [])].find((p) =>
      existsSync(p)
    );
    if (existingPath) {
      return NextResponse.json({
        success: true,
        message: "Model already exists",
      });
    }

    // Check if training is already in progress
    const status = getTrainingStatus(model);
    if (status.isTraining) {
      return NextResponse.json({
        success: true,
        message: "Training already in progress",
        status,
      });
    }

    // Start training in background
    const pythonCmd = process.platform === "win32" ? "python" : "python3";

    setTrainingStatus(model, {
      isTraining: true,
      progress: 0,
      message: `Training ${model} started. Will stop automatically at 95% validation accuracy...`,
      error: null,
    });

    // Spawn training process (don't wait for it)
    const trainProcess = spawn(pythonCmd, [config.trainScriptPath], {
      detached: true,
      stdio: "ignore",
    });

    // Unref so parent process can exit independently
    trainProcess.unref();

    // Monitor training process
    trainProcess.on("error", (error) => {
      setTrainingStatus(model, {
        isTraining: false,
        progress: 0,
        message: "Training failed to start",
        error: error.message,
      });
    });

    trainProcess.on("exit", (code) => {
      if (code === 0) {
        setTrainingStatus(model, {
          isTraining: false,
          progress: 100,
          message: "Training completed successfully! (Stopped at 95% validation accuracy)",
          error: null,
        });
      } else {
        setTrainingStatus(model, {
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


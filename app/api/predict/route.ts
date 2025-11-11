import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";

export const runtime = "nodejs";
export const maxDuration = 30; // 30 seconds max for prediction

export async function POST(request: NextRequest) {
  try {
    // Parse form data
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json(
        { error: "No file provided" },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json(
        { error: "Invalid file type. Please upload a JPEG or PNG image." },
        { status: 400 }
      );
    }

    // Check if model exists
    const modelPath = join(process.cwd(), "model", "model.h5");
    if (!existsSync(modelPath)) {
      // Try to download model if MODEL_URL is set
      const modelUrl = process.env.MODEL_URL;
      const { getTrainingStatus, setTrainingStatus } = await import("../../../lib/training-status");
      const status = getTrainingStatus();
      
      if (modelUrl && !status.isTraining) {
        // Try downloading the model
        try {
          
          setTrainingStatus({
            isTraining: true,
            progress: 0,
            message: "Downloading pre-trained model...",
            error: null,
          });
          
          const downloadScriptPath = join(process.cwd(), "ml", "download_model.py");
          const pythonCmd = process.platform === "win32" ? "python" : "python3";
          
          const downloadProcess = spawn(pythonCmd, [downloadScriptPath], {
            env: { ...process.env, MODEL_URL: modelUrl },
            detached: false,
          });
          
          downloadProcess.on("exit", async (code) => {
            if (code === 0 && existsSync(modelPath)) {
              setTrainingStatus({
                isTraining: false,
                progress: 100,
                message: "Model downloaded successfully!",
                error: null,
              });
            } else {
              // Download failed, fall back to training
              await startTrainingFallback();
            }
          });
        } catch (downloadError) {
          console.error("Failed to download model:", downloadError);
          // Fall back to training
          await startTrainingFallback();
        }
      } else {
        // No MODEL_URL set, start training
        await startTrainingFallback();
      }
      
      async function startTrainingFallback() {
        const currentStatus = getTrainingStatus();
        
        if (!currentStatus.isTraining) {
          try {
            const trainScriptPath = join(process.cwd(), "ml", "train_model.py");
            const pythonCmd = process.platform === "win32" ? "python" : "python3";
            
            setTrainingStatus({
              isTraining: true,
              progress: 0,
              message: "Training started automatically. Will stop at 90% validation accuracy...",
              error: null,
            });
            
            const trainProcess = spawn(pythonCmd, [trainScriptPath], {
              detached: true,
              stdio: "ignore",
            });
            
            trainProcess.unref();
            
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
          } catch (trainError) {
            console.error("Failed to start training:", trainError);
          }
        }
      }
      
      return NextResponse.json(
        {
          error: modelUrl 
            ? "Model not found. Downloading pre-trained model..." 
            : "Model not found. Training has been started automatically. Training will stop when validation accuracy reaches 90% (typically 5-15 minutes).",
          training: true,
          status: getTrainingStatus(),
        },
        { status: 503 } // Service Unavailable
      );
    }

    // Create temp directory if it doesn't exist
    const tempDir = join(process.cwd(), "model", "tmp");
    await mkdir(tempDir, { recursive: true });

    // Save uploaded file to temp directory
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const timestamp = Date.now();
    const fileExtension = file.name.split(".").pop() || "jpg";
    const tempFilePath = join(tempDir, `upload_${timestamp}.${fileExtension}`);
    await writeFile(tempFilePath, buffer);

    // Prepare Python script path
    const predictScriptPath = join(process.cwd(), "ml", "predict.py");
    const modelPathArg = join(process.cwd(), "model", "model.h5");

    // Call Python predict script
    // Try python3 first, fallback to python
    const pythonCmd = process.platform === "win32" ? "python" : "python3";
    
    return new Promise((resolve) => {
      const pythonProcess = spawn(pythonCmd, [
        predictScriptPath,
        "--image",
        tempFilePath,
        "--model",
        modelPathArg,
        "--gradcam",
        "true",
      ]);

      let stdout = "";
      let stderr = "";

      pythonProcess.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      pythonProcess.on("close", (code) => {
        // Clean up temp file (optional, can be done asynchronously)
        // For now, we'll leave it for potential debugging

        if (code !== 0) {
          // Try to parse error from stderr
          let errorMessage = "Prediction failed";
          try {
            const errorJson = JSON.parse(stderr);
            errorMessage = errorJson.error || errorMessage;
          } catch {
            errorMessage = stderr || errorMessage;
          }

          resolve(
            NextResponse.json(
              { error: errorMessage },
              { status: 500 }
            )
          );
          return;
        }

        try {
          // Parse JSON output from Python script
          const result = JSON.parse(stdout.trim());
          resolve(NextResponse.json(result));
        } catch (parseError) {
          resolve(
            NextResponse.json(
              {
                error: "Failed to parse prediction results",
                details: stdout,
              },
              { status: 500 }
            )
          );
        }
      });

      pythonProcess.on("error", (error) => {
        resolve(
          NextResponse.json(
            {
              error: "Failed to execute prediction script",
              details: error.message,
            },
            { status: 500 }
          )
        );
      });
    });
  } catch (error) {
    console.error("Prediction error:", error);
    return NextResponse.json(
      {
        error: "Internal server error",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}


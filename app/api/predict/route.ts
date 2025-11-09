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
      return NextResponse.json(
        { error: "Model not found. Please train the model first." },
        { status: 500 }
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


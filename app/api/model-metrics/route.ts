import { NextResponse } from "next/server";
import { join } from "path";
import { promises as fs } from "fs";

export const runtime = "nodejs";

export async function GET() {
  try {
    const metricsPath = join(process.cwd(), "model", "metrics.json");
    const data = await fs.readFile(metricsPath, "utf-8");
    const metrics = JSON.parse(data);
    return NextResponse.json(metrics);
  } catch (error) {
    console.error("Failed to load model metrics:", error);
    return NextResponse.json(
      {
        error:
          "Model metrics are not available. Train the model to generate metrics.",
      },
      { status: 404 }
    );
  }
}



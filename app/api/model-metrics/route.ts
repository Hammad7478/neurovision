import { NextRequest, NextResponse } from "next/server";
import { join } from "path";
import { promises as fs } from "fs";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  try {
    const modelParam = (request.nextUrl.searchParams.get("model") || "resnet50").toLowerCase();
    const metricsFileMap: Record<string, string> = {
      resnet50: "resnet50_metrics.json",
      mobilenetv2: "mobilenetv2_metrics.json",
    };

    const metricsFile = metricsFileMap[modelParam] ?? metricsFileMap.resnet50;
    const metricsPath = join(process.cwd(), "model", metricsFile);
    const content = await fs.readFile(metricsPath, "utf-8");
    const metrics = JSON.parse(content);
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



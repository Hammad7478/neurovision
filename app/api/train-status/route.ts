import { NextRequest, NextResponse } from "next/server";
import { existsSync } from "fs";
import { join } from "path";
import { getTrainingStatus } from "../../../lib/training-status";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  const modelParam = (request.nextUrl.searchParams.get("model") || "resnet50").toLowerCase();

  const modelMap: Record<
    string,
    { primary: string; legacy?: string[] }
  > = {
    resnet50: {
      primary: join(process.cwd(), "model", "resnet50_model.h5"),
      legacy: [join(process.cwd(), "model", "model.h5")],
    },
    mobilenetv2: { primary: join(process.cwd(), "model", "mobilenetv2_model.h5") },
  };

  const selected = modelMap[modelParam] ?? modelMap["resnet50"];
  const modelKey = modelMap[modelParam] ? modelParam : "resnet50";
  const pathsToCheck = [selected.primary, ...(selected.legacy ?? [])];

  const modelExists = pathsToCheck.some((p) => existsSync(p));
  const status = getTrainingStatus(modelKey);

  return NextResponse.json({
    model: modelKey,
    modelExists,
    ...status,
  });
}


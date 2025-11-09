import { NextResponse } from "next/server";
import { existsSync } from "fs";
import { join } from "path";
import { getTrainingStatus } from "../../../lib/training-status";

export const runtime = "nodejs";

export async function GET() {
  const modelPath = join(process.cwd(), "model", "model.h5");
  const modelExists = existsSync(modelPath);
  const status = getTrainingStatus();

  return NextResponse.json({
    modelExists,
    ...status,
  });
}


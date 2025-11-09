import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";

export const runtime = "nodejs";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    // Get the file path from params
    const { path } = await params;
    
    // Security: prevent directory traversal
    if (path.some(segment => segment.includes("..") || segment.includes("/"))) {
      return NextResponse.json(
        { error: "Invalid file path" },
        { status: 400 }
      );
    }
    
    const filename = path.join("/");

    // Construct full path
    const filePath = join(process.cwd(), "model", "tmp", filename);

    // Check if file exists
    if (!existsSync(filePath)) {
      return NextResponse.json(
        { error: "File not found" },
        { status: 404 }
      );
    }

    // Read file
    const fileBuffer = await readFile(filePath);

    // Determine content type based on file extension
    const ext = filename.split(".").pop()?.toLowerCase();
    let contentType = "image/png";
    if (ext === "jpg" || ext === "jpeg") {
      contentType = "image/jpeg";
    }

    // Return image with appropriate headers
    return new NextResponse(fileBuffer, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (error) {
    console.error("Error serving Grad-CAM image:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}


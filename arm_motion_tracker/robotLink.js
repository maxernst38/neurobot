// Thin WebSocket client that forwards per-joint direction state to the
// main.py --input webcam control bridge. Kept separate from main.js so the
// tracking/UI code doesn't need to know about connection/retry details.

export class RobotLink {
  constructor(onStatusChange) {
    this.ws = null;
    this.onStatusChange = onStatusChange || (() => {});
    this.lastSendMs = 0;
    this.sendIntervalMs = 66; // ~15Hz, independent of the render/detection framerate
  }

  connect(url) {
    this.disconnect();
    this.onStatusChange("connecting");
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      this.onStatusChange("error");
      return;
    }
    ws.onopen = () => this.onStatusChange("connected");
    ws.onclose = () => this.onStatusChange("disconnected");
    ws.onerror = () => this.onStatusChange("error");
    this.ws = ws;
  }

  disconnect() {
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.close();
      this.ws = null;
    }
  }

  get isConnected() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  /** directions: {metricKey: "inc"|"dec"|"idle"|"none", ...}. Throttled and silently dropped when not connected. */
  send(directions, nowMs) {
    if (!this.isConnected) return;
    if (nowMs - this.lastSendMs < this.sendIntervalMs) return;
    this.lastSendMs = nowMs;
    this.ws.send(JSON.stringify(directions));
  }
}

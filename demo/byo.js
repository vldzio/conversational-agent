/*
 * Shared demo-UI helper (copied into each project's demo UI).
 *  - window.demoPost(path, body)  -> POST JSON to the demo API, throws Error(message) on failure
 *  - "LLM key" panel: visitors may bring their own OpenAI-compatible key (kept in sessionStorage only,
 *    sent only to this demo's API, bypasses the shared daily limit)
 *  - banner explaining what is live
 * Config: window.DEMO_CONFIG = { API_URL, TITLE, SOURCE_URL, NOTE }
 */
(function () {
  "use strict";
  const CFG = window.DEMO_CONFIG || {};
  const KEY = "demo.byo.key", URLK = "demo.byo.baseUrl";
  const PROVIDERS = [
    ["https://api.openai.com/v1", "OpenAI"],
    ["https://api.groq.com/openai/v1", "Groq"],
    ["https://openrouter.ai/api/v1", "OpenRouter"],
    ["https://generativelanguage.googleapis.com/v1beta/openai", "Google Gemini"],
    ["https://api.mistral.ai/v1", "Mistral"],
    ["https://api.cerebras.ai/v1", "Cerebras"],
  ];
  const store = {
    get(k) { try { return sessionStorage.getItem(k) || ""; } catch (_) { return ""; } },
    set(k, v) { try { v ? sessionStorage.setItem(k, v) : sessionStorage.removeItem(k); } catch (_) { /* private mode */ } },
  };
  const api = () => (CFG.API_URL || "").replace(/\/+$/, "");

  window.demoPost = async function (path, body, extraHeaders) {
    if (!api()) throw new Error("The demo API URL is not configured yet.");
    const headers = Object.assign({ "Content-Type": "application/json" }, extraHeaders || {});
    const key = store.get(KEY);
    if (key) { headers["X-LLM-Api-Key"] = key; headers["X-LLM-Base-Url"] = store.get(URLK) || PROVIDERS[0][0]; }
    let resp;
    try { resp = await fetch(api() + path, { method: "POST", headers, body: JSON.stringify(body || {}) }); }
    catch (e) { throw new Error("Could not reach the demo API. It may be waking up from sleep (about a minute) - please try again."); }
    const text = await resp.text();
    let data; try { data = JSON.parse(text); } catch (_) { data = { error: text.slice(0, 300) }; }
    if (!resp.ok) throw new Error(data.error || ("Request failed (" + resp.status + ")"));
    return data;
  };

  function authHeaders(extra) {
    const headers = Object.assign({ "Content-Type": "application/json" }, extra || {});
    const key = store.get(KEY);
    if (key) { headers["X-LLM-Api-Key"] = key; headers["X-LLM-Base-Url"] = store.get(URLK) || PROVIDERS[0][0]; }
    return headers;
  }

  // POST JSON and read the reply as Server-Sent Events; onEvent(name, data) runs as each event arrives.
  window.demoStream = async function (path, body, onEvent, signal) {
    if (!api()) throw new Error("The demo API URL is not configured yet.");
    let resp;
    try { resp = await fetch(api() + path, { method: "POST", headers: authHeaders(), body: JSON.stringify(body || {}), signal }); }
    catch (e) { if (e.name === "AbortError") throw e; throw new Error("Could not reach the demo API. It may be waking up from sleep (about a minute) - please try again."); }
    if (!resp.ok || !(resp.headers.get("content-type") || "").includes("text/event-stream")) {
      const text = await resp.text();
      let data; try { data = JSON.parse(text); } catch (_) { data = { error: text.slice(0, 300) }; }
      throw new Error(data.error || ("Request failed (" + resp.status + ")"));
    }
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        let name = "message", data = "";
        block.split("\n").forEach(function (line) {
          if (line.startsWith("event: ")) name = line.slice(7);
          else if (line.startsWith("data: ")) data += line.slice(6);
        });
        let parsed; try { parsed = JSON.parse(data); } catch (_) { continue; }
        if (name === "error") throw new Error(parsed.error || "The demo API reported an error.");
        onEvent(name, parsed);
      }
    }
  };

  // Fire-and-forget POST (used to prepare a turn early); failures are ignored by the caller.
  window.demoBackground = function (path, body, signal) {
    if (!api()) return Promise.resolve(null);
    return fetch(api() + path, { method: "POST", headers: authHeaders(), body: JSON.stringify(body || {}), signal })
      .then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  };

  // POST for the agent's voice: returns the raw Response so the caller can read the audio as it streams in.
  // The visitor's own LLM key is deliberately not sent: it does not pay for speech.
  window.demoAudio = function (path, body, signal) {
    if (!api()) return Promise.reject(new Error("The demo API URL is not configured yet."));
    return fetch(api() + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}), signal });
  };

  // Wake the API (free hosting sleeps when idle) and open the connection before the first turn.
  window.demoWarm = function () {
    if (!api()) return Promise.resolve(null);
    return fetch(api() + "/").then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; });
  };

  document.addEventListener("DOMContentLoaded", function () {
    const bar = document.createElement("div");
    bar.className = "demo-bar";
    const title = document.createElement("span");
    title.className = "demo-bar-text";
    title.textContent = (CFG.NOTE || "Live demo: AI output is generated on the fly through a free-tier LLM gateway.") + " Each visitor gets a small daily allowance.";
    bar.appendChild(title);
    if (CFG.SOURCE_URL) {
      const a = document.createElement("a"); a.href = CFG.SOURCE_URL; a.target = "_blank"; a.rel = "noopener"; a.textContent = "Source"; bar.appendChild(a);
    }
    const det = document.createElement("details"); det.className = "demo-key";
    const sum = document.createElement("summary"); sum.textContent = "LLM key"; det.appendChild(sum);
    const dot = document.createElement("span"); dot.className = "demo-dot"; sum.appendChild(dot);
    const panel = document.createElement("div"); panel.className = "demo-key-panel";
    panel.innerHTML = '<p>Optional: use your own OpenAI-compatible key instead of the shared free tier. Stored in this tab only; never logged.</p>' +
      '<select id="byo-provider">' + PROVIDERS.map(function (p) { return '<option value="' + p[0] + '">' + p[1] + "</option>"; }).join("") + "</select>" +
      '<input id="byo-key" type="password" autocomplete="off" placeholder="API key" />' +
      '<div><button id="byo-save">Save</button> <button id="byo-clear" class="secondary">Clear</button></div>';
    det.appendChild(panel); bar.appendChild(det);
    document.body.insertBefore(bar, document.body.firstChild);
    const keyEl = panel.querySelector("#byo-key"), provEl = panel.querySelector("#byo-provider");
    const refresh = function () { dot.textContent = store.get(KEY) ? " ●" : ""; };
    provEl.value = store.get(URLK) || PROVIDERS[0][0]; refresh();
    panel.querySelector("#byo-save").addEventListener("click", function () { store.set(KEY, keyEl.value.trim()); store.set(URLK, provEl.value); keyEl.value = ""; refresh(); det.open = false; });
    panel.querySelector("#byo-clear").addEventListener("click", function () { store.set(KEY, ""); keyEl.value = ""; refresh(); });
  });
})();

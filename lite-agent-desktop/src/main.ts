import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import DOMPurify from "dompurify";
import hljs from "highlight.js";
import { Marked, Renderer } from "marked";
import markedKatex from "marked-katex-extension";
import "highlight.js/styles/github-dark.css";
import "katex/dist/katex.min.css";
import "./style.css";

type Config = { base_url: string; model: string; api_key: string; qianfan_api_key: string; workspace: string; max_context_tokens: number; max_model_iterations: number };
type Thread = { id: string; created_at: string; updated_at: string; turns: any[]; metadata: any };
type ToolCall = { call_id: string; name: string; arguments: unknown };
type LiveTool = ToolCall & { status: "queued" | "running" | "completed" | "failed"; error?: string };
type LiveTurn = { threadId: string; userText: string; assistantParts: string[]; iteration: number; tools: LiveTool[]; stopping?: boolean; error?: string };
type DesktopTurnEvent = { type: string; text?: string; iteration?: number; calls?: ToolCall[]; call_id?: string; name?: string; error?: string; reason?: string; outcome?: string };

const state = {
  config: null as Config | null,
  threads: [] as Thread[],
  selected: "",
  workspaceDialog: false,
  pending: null as LiveTurn | null,
  openProcesses: new Set<string>(),
  threadErrors: new Map<string, string>(),
  diagnosticsDir: "",
};

let composing = false;
let compositionEndedAt = 0;
let liveRenderFrame = 0;

const toolNames: Record<string, string> = {
  exec_command: "执行命令",
  web_search: "搜索网页",
  get_current_time: "获取当前时间",
  get_goal: "读取任务目标",
  update_goal: "更新任务目标",
  ask_user: "请求用户输入",
};

const renderer = new Renderer();
renderer.code = ({ text, lang }) => {
  const language = (lang || "").trim().split(/\s+/)[0];
  if (language === "mermaid") return `<div class="mermaid">${escapeHtml(text)}</div>`;
  const highlighted = language && hljs.getLanguage(language)
    ? hljs.highlight(text, { language }).value
    : hljs.highlightAuto(text).value;
  return `<pre><code class="hljs${language ? ` language-${escapeHtml(language)}` : ""}">${highlighted}</code></pre>`;
};

const markdown = new Marked(
  { gfm: true, breaks: true, renderer },
  markedKatex({ throwOnError: false, nonStandard: true }),
);

let mermaidReady: Promise<typeof import("mermaid").default> | null = null;

const el = (id: string) => document.getElementById(id)!;
const escapeHtml = (value: unknown) => String(value ?? "").replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]!));

function renderMarkdown(value: string) {
  const html = markdown.parse(value || "") as string;
  return DOMPurify.sanitize(html, { USE_PROFILES: { html: true, svg: true, svgFilters: true, mathMl: true } });
}

async function refresh(renderApp = true) {
  state.threads = await invoke<Thread[]>("list_threads");
  if (renderApp) render();
}

async function boot() {
  render();
  try {
    state.config = await invoke<Config>("get_config");
    state.diagnosticsDir = await invoke<string>("get_diagnostics_dir");
    const runtimeError = await invoke<string | null>("get_runtime_error");
    await listen<DesktopTurnEvent>("turn-event", event => handleTurnEvent(event.payload));
    await refresh();
    if (runtimeError) showSettings(false, runtimeError);
    else if (!state.config.base_url || !state.config.model) showSettings(true);
  } catch (error) {
    el("app").innerHTML = `<div class="settings"><h1>启动失败</h1><pre>${escapeHtml(error)}</pre></div>`;
  }
}

function render() {
  el("app").innerHTML = `<div class="shell">
    <aside><div class="brand"><span class="brand-mark">L</span><span>lite-agent</span></div><button id="new">＋ 新建对话</button>
      <div class="thread-list">${state.threads.map(thread => `<button class="thread ${thread.id === state.selected ? "active" : ""}" data-id="${thread.id}"><span>${title(thread)}</span><small>${formatTime(thread.updated_at)}</small></button>`).join("")}</div>
      <button id="settings" class="bottom">⚙ 设置</button></aside>
    <main><header><div><strong>${state.selected ? title(currentThread()) : "欢迎使用 lite-agent"}</strong>${state.selected ? `<small>${escapeHtml(currentThread()?.metadata?.workspace || "")}</small>` : ""}</div><button id="workspace" class="header-button">工作区</button></header>
      <section id="conversation" class="content">${state.selected ? conversationHtml() : welcome()}</section>
      ${state.selected ? composerHtml() : ""}</main>
  </div>${state.workspaceDialog ? workspaceDialog() : ""}`;
  bindEvents();
  if (state.selected) {
    const content = el("conversation");
    requestAnimationFrame(() => { content.scrollTop = content.scrollHeight; hydrateRichContent(content); });
  }
}

function bindEvents() {
  el("new")?.addEventListener("click", openWorkspaceDialog);
  el("welcome-new")?.addEventListener("click", openWorkspaceDialog);
  el("workspace")?.addEventListener("click", openWorkspaceDialog);
  el("settings")?.addEventListener("click", () => showSettings(false));
  el("workspace-cancel")?.addEventListener("click", closeWorkspaceDialog);
  el("workspace-confirm")?.addEventListener("click", createThread);
  document.querySelectorAll<HTMLButtonElement>(".thread").forEach(button => {
    button.onclick = () => { state.selected = button.dataset.id!; render(); };
  });
  el("composer")?.addEventListener("submit", sendMessage);
  el("stop")?.addEventListener("click", stopTurn);
  el("prompt")?.addEventListener("compositionstart", () => { composing = true; });
  el("prompt")?.addEventListener("compositionend", () => { composing = false; compositionEndedAt = Date.now(); });
  el("prompt")?.addEventListener("keydown", event => {
    const keyboardEvent = event as KeyboardEvent;
    if (keyboardEvent.key !== "Enter") return;
    if (keyboardEvent.isComposing || composing || keyboardEvent.keyCode === 229 || Date.now() - compositionEndedAt < 120) return;
    if (keyboardEvent.metaKey || keyboardEvent.ctrlKey || keyboardEvent.shiftKey) return;
    keyboardEvent.preventDefault();
    (el("composer") as HTMLFormElement).requestSubmit();
  });
  bindProcessToggles(document);
}

function bindProcessToggles(root: ParentNode) {
  root.querySelectorAll<HTMLDetailsElement>("details[data-process-id]").forEach(details => {
    details.addEventListener("toggle", () => {
      const id = details.dataset.processId;
      if (!id) return;
      if (details.open) state.openProcesses.add(id);
      else state.openProcesses.delete(id);
    });
  });
}

function handleTurnEvent(event: DesktopTurnEvent) {
  const pending = state.pending;
  if (!pending) return;
  if (event.type === "model_iteration_started") {
    pending.iteration = event.iteration || 0;
    pending.assistantParts[pending.iteration] ??= "";
  } else if (event.type === "assistant_delta") {
    pending.assistantParts[pending.iteration] = (pending.assistantParts[pending.iteration] || "") + (event.text || "");
  } else if (event.type === "assistant_message") {
    pending.assistantParts[pending.iteration] = event.text || "";
  } else if (event.type === "tool_calls_requested") {
    for (const call of event.calls || []) if (!pending.tools.some(tool => tool.call_id === call.call_id)) pending.tools.push({ ...call, status: "queued" });
  } else if (event.type === "tool_started" || event.type === "tool_completed" || event.type === "tool_failed") {
    let tool = pending.tools.find(value => value.call_id === event.call_id);
    if (!tool && event.call_id && event.name) {
      tool = { call_id: event.call_id, name: event.name, arguments: {}, status: "queued" };
      pending.tools.push(tool);
    }
    if (tool) {
      tool.status = event.type === "tool_started" ? "running" : event.type === "tool_completed" ? "completed" : "failed";
      tool.error = event.error;
    }
  } else if (event.type === "turn_failed" || event.type === "turn_aborted") {
    pending.error = event.error || event.reason || "本轮已终止";
  }
  if (pending.threadId === state.selected) scheduleLiveTurnDom();
}

function scheduleLiveTurnDom() {
  if (liveRenderFrame) return;
  liveRenderFrame = requestAnimationFrame(() => {
    liveRenderFrame = 0;
    updateLiveTurnDom();
  });
}

function updateLiveTurnDom() {
  const content = document.getElementById("conversation");
  const pending = state.pending;
  if (!content || !pending || pending.threadId !== state.selected) return;
  const staysAtBottom = content.scrollHeight - content.scrollTop - content.clientHeight < 96;
  const assistant = pending.assistantParts.filter(Boolean).join("\n\n");
  const processSlot = document.getElementById("live-process-slot");
  if (processSlot) {
    processSlot.innerHTML = pending.tools.length ? liveToolGroupHtml(pending.tools) : "";
    bindProcessToggles(processSlot);
  }
  const assistantRow = document.getElementById("live-assistant-row");
  const assistantBody = document.getElementById("live-assistant-content");
  if (assistantRow && assistantBody) {
    assistantRow.hidden = !assistant;
    assistantBody.innerHTML = assistant ? `${renderMarkdown(assistant)}<span class="cursor"></span>` : "";
  }
  const placeholder = document.getElementById("live-placeholder");
  if (placeholder) placeholder.hidden = Boolean(assistant);
  const errorBox = document.getElementById("live-error");
  if (errorBox) {
    errorBox.hidden = !pending.error;
    errorBox.textContent = pending.error || "";
  }
  const stopButton = document.getElementById("stop") as HTMLButtonElement | null;
  if (stopButton) {
    stopButton.disabled = Boolean(pending.stopping);
    stopButton.title = pending.stopping ? "正在停止" : "停止生成";
  }
  if (staysAtBottom) requestAnimationFrame(() => { content.scrollTop = content.scrollHeight; });
}

function conversationHtml() {
  const persisted = persistedConversation(currentThread());
  const live = state.pending?.threadId === state.selected ? liveTurnHtml(state.pending) : "";
  const transientError = state.threadErrors.get(state.selected);
  return `<div class="conversation-column">${persisted}${live}${transientError ? `<div class="turn-error">${escapeHtml(transientError)}</div>` : ""}</div>`;
}

function persistedConversation(thread?: Thread) {
  return (thread?.turns || []).map((turn: any, index: number) => {
    const items = turn.items || [];
    const userItems = items.filter((item: any) => item.type === "user_input");
    const modelItems = items.filter((item: any) => item.type === "model_response");
    const outputs = new Map(items.filter((item: any) => item.type === "tool_output").map((item: any) => [item.call_id, item]));
    const calls = modelItems.flatMap((item: any) => item.function_calls || []);
    const assistant = modelItems.map((item: any) => item.text).filter(Boolean).join("\n\n");
    const failure = items.filter((item: any) => item.type === "turn_failed").at(-1)?.error;
    const abortion = items.filter((item: any) => item.type === "turn_aborted").at(-1)?.reason;
    const processId = `turn:${turn.id || index}`;
    const terminalError = failure || abortion;
    return `${userItems.map((item: any) => userMessage(item.text)).join("")}${calls.length ? toolGroupHtml(calls, outputs, processId) : ""}${assistant ? assistantMessage(assistant) : ""}${terminalError ? `<div class="turn-error">${escapeHtml(terminalError)}</div>` : ""}`;
  }).join("");
}

function liveTurnHtml(turn: LiveTurn) {
  const assistant = turn.assistantParts.filter(Boolean).join("\n\n");
  return `${userMessage(turn.userText, true)}<div id="live-process-slot">${turn.tools.length ? liveToolGroupHtml(turn.tools) : ""}</div><article id="live-assistant-row" class="message-row assistant-row" ${assistant ? "" : "hidden"}><div class="avatar">L</div><div id="live-assistant-content" class="message assistant-message markdown-body">${assistant ? `${renderMarkdown(assistant)}<span class="cursor"></span>` : ""}</div></article><div id="live-placeholder" class="assistant-placeholder" ${assistant ? "hidden" : ""}><span></span><span></span><span></span></div><div id="live-error" class="turn-error" ${turn.error ? "" : "hidden"}>${escapeHtml(turn.error || "")}</div>`;
}

function userMessage(text: string, pending = false) {
  return `<article class="message-row user-row ${pending ? "pending" : ""}"><div class="message user-message">${escapeHtml(text).replace(/\n/g, "<br>")}</div></article>`;
}

function assistantMessage(text: string, streaming = false) {
  return `<article class="message-row assistant-row"><div class="avatar">L</div><div class="message assistant-message markdown-body">${renderMarkdown(text)}${streaming ? `<span class="cursor"></span>` : ""}</div></article>`;
}

function toolGroupHtml(calls: any[], outputs: Map<any, any>, processId: string) {
  return processGroupHtml(processId, calls.length, calls.map(call => toolCard(call.name, call.arguments, outputs.get(call.call_id))).join(""));
}

function liveToolGroupHtml(tools: LiveTool[]) {
  return processGroupHtml(`live:${state.pending?.threadId || state.selected}`, tools.length, tools.map(tool => toolCard(tool.name, tool.arguments, undefined, tool.status, tool.error)).join(""));
}

function processGroupHtml(processId: string, count: number, body: string) {
  const open = state.openProcesses.has(processId) ? "open" : "";
  return `<details class="thinking-group" data-process-id="${escapeHtml(processId)}" ${open}><summary class="thinking-summary"><span>过程</span><small>${count} 个步骤</small><span class="process-chevron">›</span></summary><div class="thinking-body">${body}</div></details>`;
}

function toolCard(name: string, args: unknown, output?: any, status = "completed", error?: string) {
  const label = toolNames[name] || name;
  const stateLabel = status === "running" ? "正在" : status === "queued" ? "准备" : status === "failed" ? "失败" : "已完成";
  const body = output ? formatToolResult(output.result) : error || formatToolResult(args);
  return `<details class="tool-card"><summary><span class="tool-status ${status}"></span><span>${stateLabel}${escapeHtml(label)}</span><span class="chevron">›</span></summary><pre>${escapeHtml(body)}</pre></details>`;
}

function formatToolResult(value: any) {
  const raw = value?.output ?? value?.error ?? value?.reason ?? value;
  let text = typeof raw === "string" ? raw : JSON.stringify(raw ?? {}, null, 2);
  const lines = text.split("\n");
  if (lines.length > 36) text = `${lines.slice(0, 36).join("\n")}\n…（已截断 ${lines.length - 36} 行）`;
  if (text.length > 5000) text = `${text.slice(0, 5000)}\n…（输出已截断）`;
  return text;
}

function composerHtml() {
  const busy = state.pending?.threadId === state.selected;
  const action = busy ? `<button id="stop" type="button" class="stop-button" title="停止生成" aria-label="停止生成">■</button>` : `<button type="submit" aria-label="发送">↑</button>`;
  return `<form id="composer"><textarea id="prompt" ${busy ? "disabled" : ""} placeholder="给 lite-agent 发送消息"></textarea><div class="composer-footer"><span>Enter 发送 · ⌘ Enter 换行</span>${action}</div></form>`;
}

async function sendMessage(event: Event) {
  event.preventDefault();
  const input = el("prompt") as HTMLTextAreaElement;
  const userText = input.value.trim();
  if (!userText || state.pending) return;
  state.threadErrors.delete(state.selected);
  state.pending = { threadId: state.selected, userText, assistantParts: [""], iteration: 0, tools: [] };
  state.openProcesses.add(`live:${state.selected}`);
  input.value = "";
  render();
  try {
    await invoke("run_turn", { threadId: state.selected, userText });
    state.pending = null;
    await refresh();
  } catch (error) {
    const threadId = state.pending?.threadId || state.selected;
    state.threadErrors.set(threadId, `发送失败：${String(error)}`);
    state.pending = null;
    await refresh();
  }
}

async function stopTurn() {
  const pending = state.pending;
  if (!pending || pending.stopping) return;
  pending.stopping = true;
  updateLiveTurnDom();
  try {
    await invoke("abort_turn", { threadId: pending.threadId });
  } catch (error) {
    pending.stopping = false;
    pending.error = `停止失败：${String(error)}`;
    updateLiveTurnDom();
  }
}

async function hydrateRichContent(container: HTMLElement) {
  const diagrams = Array.from(container.querySelectorAll<HTMLElement>(".mermaid:not([data-processed])"));
  if (!diagrams.length) return;
  try {
    mermaidReady ??= import("mermaid").then(module => {
      module.default.initialize({ startOnLoad: false, theme: "dark", securityLevel: "strict", suppressErrorRendering: true });
      return module.default;
    });
    const mermaid = await mermaidReady;
    await mermaid.run({ nodes: diagrams });
  } catch (error) { console.warn("Mermaid 渲染失败", error); }
}

function currentThread() { return state.threads.find(thread => thread.id === state.selected); }
function title(thread?: Thread) { const item = thread?.turns?.[0]?.items?.find((value: any) => value.type === "user_input"); return escapeHtml(item?.text?.slice(0, 28) || "新对话"); }
function formatTime(timestamp: string) { const seconds = Number(timestamp); return Number.isFinite(seconds) ? new Date(seconds * 1000).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }) : timestamp; }
function welcome() { return `<div class="welcome"><div class="welcome-logo">L</div><h1>今天想做点什么？</h1><p>创建对话并选择工作区，lite-agent 会在工作区内协助你。</p><button id="welcome-new">开始新对话</button></div>`; }

function openWorkspaceDialog() { state.workspaceDialog = true; render(); (el("workspace-input") as HTMLInputElement).focus(); }
function closeWorkspaceDialog() { state.workspaceDialog = false; render(); }
function workspaceDialog() { return `<div class="modal-backdrop"><div class="modal"><h2>选择工作区</h2><p>该目录在当前对话中可读写，目录外只读。</p><input id="workspace-input" value="${escapeHtml(state.config?.workspace || "")}" placeholder="/Users/name/workspace"><p id="workspace-error" class="error"></p><div class="modal-actions"><button id="workspace-cancel" class="secondary">取消</button><button id="workspace-confirm">创建对话</button></div></div></div>`; }

async function createThread() {
  const input = el("workspace-input") as HTMLInputElement;
  const errorBox = el("workspace-error");
  const button = el("workspace-confirm") as HTMLButtonElement;
  const workspace = input.value.trim();
  if (!workspace) { errorBox.textContent = "请输入工作区路径"; return; }
  button.disabled = true; errorBox.textContent = "正在创建…";
  try { state.selected = await invoke<string>("create_thread", { workspace }); state.workspaceDialog = false; await refresh(); }
  catch (error) { errorBox.textContent = `创建失败：${String(error)}`; button.disabled = false; }
}

function showSettings(first: boolean, runtimeError = "") {
  const config = state.config || { base_url: "", model: "", api_key: "", qianfan_api_key: "", workspace: "", max_context_tokens: 262144, max_model_iterations: 1024 };
  el("app").innerHTML = `<div class="settings-page"><div class="settings"><h1>${first ? "欢迎使用 lite-agent" : "设置"}</h1><p>所有 API Key 仅保存在本机，不会写入应用安装包。</p>${runtimeError ? `<div class="turn-error">运行时初始化失败：${escapeHtml(runtimeError)}</div>` : ""}<section class="settings-section"><h2>语言模型</h2><p>请填写兼容 OpenAI Chat Completions 的服务配置。</p><label>LLM URL<input id="url" value="${escapeHtml(config.base_url)}"></label><label>模型<input id="model" value="${escapeHtml(config.model)}"></label><label>API Key<input id="key" type="password" autocomplete="off" value="${escapeHtml(config.api_key)}"></label><label>上下文预算（tokens）<input id="max-context-tokens" type="number" min="8192" max="2000000" step="1024" value="${config.max_context_tokens}"></label><p>默认 262,144；不要超过模型服务实际支持的上下文窗口。</p><label>单轮最大 Model Iterations<input id="max-model-iterations" type="number" min="1" max="65535" value="${config.max_model_iterations}"></label><p>默认 1,024；达到上限时会终止本轮并记录明确错误。</p></section><section class="settings-section"><h2>千帆百度搜索</h2><p>用于搜索中文网页和实时信息。未配置时不影响聊天，但无法使用网页搜索。</p><label>千帆 API Key<input id="qianfan-key" type="password" autocomplete="off" value="${escapeHtml(config.qianfan_api_key || "")}" placeholder="请输入百度智能云千帆 API Key"></label></section><section class="settings-section"><h2>工作区</h2><label>默认工作区<input id="ws" value="${escapeHtml(config.workspace)}"></label></section><section class="settings-section"><h2>诊断日志</h2><p>应用日志和每个 thread 的 JSONL trace 会实时写入：</p><code>${escapeHtml(state.diagnosticsDir || "保存设置后生成")}</code></section><button id="save">保存并继续</button><p id="save-error" class="error"></p></div></div>`;
  el("save").onclick = async () => {
    const button = el("save") as HTMLButtonElement; const errorBox = el("save-error"); button.disabled = true; errorBox.textContent = "保存中…";
    try { state.config = { base_url: (el("url") as HTMLInputElement).value.trim(), model: (el("model") as HTMLInputElement).value.trim(), api_key: (el("key") as HTMLInputElement).value, qianfan_api_key: (el("qianfan-key") as HTMLInputElement).value.trim(), workspace: (el("ws") as HTMLInputElement).value.trim(), max_context_tokens: Number.parseInt((el("max-context-tokens") as HTMLInputElement).value, 10), max_model_iterations: Number.parseInt((el("max-model-iterations") as HTMLInputElement).value, 10) }; if (!state.config.base_url || !state.config.model) throw new Error("LLM URL 和模型不能为空"); if (!Number.isInteger(state.config.max_context_tokens) || state.config.max_context_tokens < 8192 || state.config.max_context_tokens > 2000000) throw new Error("上下文预算必须在 8,192 到 2,000,000 之间"); if (!Number.isInteger(state.config.max_model_iterations) || state.config.max_model_iterations < 1 || state.config.max_model_iterations > 65535) throw new Error("Model Iterations 必须在 1 到 65,535 之间"); await invoke("save_config", { config: state.config }); await refresh(); }
    catch (error) { errorBox.textContent = `保存失败：${String(error)}`; button.disabled = false; }
  };
}

boot();

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./style.css";

type Config = { base_url: string; model: string; api_key: string; workspace: string };
type Thread = { id: string; created_at: string; updated_at: string; turns: any[]; metadata: any };
const state = { config: null as Config | null, threads: [] as Thread[], selected: "", events: [] as string[] };
const el = (id: string) => document.getElementById(id)!;

async function refresh() { state.threads = await invoke<Thread[]>("list_threads"); render(); }
async function boot() {
  state.config = await invoke<Config>("get_config");
  await listen<{kind:string,message:string}>("turn-event", e => { state.events.push(e.payload.message); render(); });
  await refresh();
  if (!state.config.base_url || !state.config.model) showSettings(true); else render();
}
function render() {
  el("app").innerHTML = `<div class="shell"><aside><div class="brand">lite-agent</div><button id="new">＋ 新建对话</button><div class="thread-list">${state.threads.map(t => `<button class="thread ${t.id===state.selected?"active":""}" data-id="${t.id}">${title(t)}<small>${t.updated_at}</small></button>`).join("")}</div><button id="settings" class="bottom">⚙ 设置</button></aside><main><header><span>${state.selected ? "对话" : "欢迎使用 lite-agent"}</span><button id="workspace">工作区</button></header><section class="content">${state.selected ? conversation() : welcome()}</section>${state.selected ? `<form id="composer"><textarea id="prompt" placeholder="输入消息…"></textarea><button>发送</button></form>` : ""}</main></div>`;
  el("new")?.addEventListener("click", () => newThread()); el("welcome-new")?.addEventListener("click", () => newThread()); el("settings")?.addEventListener("click", () => showSettings(false)); el("workspace")?.addEventListener("click", () => newThread());
  document.querySelectorAll<HTMLButtonElement>(".thread").forEach(b => b.onclick = () => { state.selected=b.dataset.id!; state.events=[]; render(); });
  el("composer")?.addEventListener("submit", async e => { e.preventDefault(); const prompt=(el("prompt") as HTMLTextAreaElement).value.trim(); if(!prompt)return; (el("prompt") as HTMLTextAreaElement).value=""; state.events=[]; render(); await invoke("run_turn", { threadId: state.selected, userText: prompt }); await refresh(); });
}
function title(t: Thread) { const item=t.turns?.[0]?.items?.find((x:any)=>x.type==="user_input"); return item?.text?.slice(0,28) || "新对话"; }
function conversation() { const t=state.threads.find(x=>x.id===state.selected); const turns=(t?.turns||[]).flatMap((x:any)=>x.items||[]); return `${turns.map((x:any)=>{ const label=x.source==="user"?"你":x.source==="model"?"助手":x.source==="tool"?"工具":"系统"; const body=esc(x.text || x.result?.output || x.result?.error || x.function_calls?.map((c:any)=>`调用 ${c.name}`).join("、") || ""); return x.source==="tool" || x.function_calls?.length ? `<details class="item ${x.source}" ${x.source==="model"?"":"open"}><summary><b>${label}</b> ${x.function_calls?.map((c:any)=>esc(c.name)).join("、") || "工具结果"}</summary><div>${body}</div></details>` : `<article class="item ${x.source}"><b>${label}</b><div>${body}</div></article>`; }).join("")}<div class="live">${state.events.map(esc).join("<br>")}</div>`; }
function welcome() { return `<div class="welcome"><h1>你好，我是 lite-agent</h1><p>选择左侧对话，或创建一个新的对话。每个对话都可以绑定独立工作区。</p><button id="welcome-new">开始新对话</button></div>`; }
async function newThread() { const workspace=prompt("请输入工作区绝对路径", state.config?.workspace || ""); if(!workspace)return; state.selected=await invoke<string>("create_thread", { workspace }); await refresh(); }
function showSettings(first: boolean) { const c=state.config!; el("app").innerHTML=`<div class="settings"><h1>${first?"欢迎使用 lite-agent":"设置"}</h1><p>请填写兼容 OpenAI Chat Completions 的服务配置。API Key 仅保存在本机。</p><label>LLM URL<input id="url" value="${esc(c.base_url)}" placeholder="https://api.openai.com/v1"></label><label>模型<input id="model" value="${esc(c.model)}"></label><label>API Key<input id="key" type="password" value="${esc(c.api_key)}"></label><label>默认工作区<input id="ws" value="${esc(c.workspace)}"></label><button id="save">保存并继续</button></div>`; el("save").onclick=async()=>{state.config={base_url:(el("url") as HTMLInputElement).value,model:(el("model") as HTMLInputElement).value,api_key:(el("key") as HTMLInputElement).value,workspace:(el("ws") as HTMLInputElement).value}; await invoke("save_config",{config:state.config}); await refresh();}; }
function esc(v: unknown) { return String(v??"").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]!)); }
boot();

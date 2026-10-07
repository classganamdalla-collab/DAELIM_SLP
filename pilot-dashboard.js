const filesInput = document.getElementById("files");
const statusEl = document.getElementById("status");
const metricsEl = document.getElementById("metrics");
const participantRows = document.getElementById("participantRows");
const taskRows = document.getElementById("taskRows");
const surveyRows = document.getElementById("surveyRows");
const surveyComments = document.getElementById("surveyComments");
const exportCsvBtn = document.getElementById("exportCsv");

let logs = [];
let surveys = [];

const mean = xs => xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0;
const median = xs => {
  if (!xs.length) return 0;
  const a=[...xs].sort((x,y)=>x-y);
  const m=Math.floor(a.length/2);
  return a.length%2 ? a[m] : (a[m-1]+a[m])/2;
};
const pct = x => `${(100*x).toFixed(1)}%`;
const sec = ms => (ms/1000).toFixed(2);
const ms1 = n => Number(n||0).toFixed(2);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  })[ch]);
}

function modeOf(t) {
  if (!t?.success) return "failed";
  if (t.manualFallbackUsed) return "fallback";
  if (Number(t.candidateRepairs || 0) > 0) return "repair";
  return "ai-only";
}

function modeKo(mode) {
  return ({
    "ai-only":"AI 단독",
    "repair":"후보 repair",
    "fallback":"직접입력 fallback",
    "failed":"실패",
  })[mode] || mode;
}

function allTasks() {
  return logs.flatMap(log => (log.tasks || []).map(t => ({
    ...t,
    participant: log.participant || "unknown",
    completionMode: t.completionMode || modeOf(t),
  })));
}

function participantStats() {
  const map = new Map();
  for (const log of logs) {
    const p = log.participant || "unknown";
    if (!map.has(p)) map.set(p, []);
    map.get(p).push(log);
  }
  return [...map.entries()].map(([participant, items]) => {
    const tasks = items.flatMap(x => x.tasks || []).map(t => ({...t, completionMode:t.completionMode || modeOf(t)}));
    const inferences = items.map(x => x.summary?.meanInferenceLatencyMs || 0).filter(Boolean);
    const success = tasks.filter(t => t.success).length;
    return {
      participant,
      tasks: tasks.length,
      successRate: tasks.length ? success/tasks.length : 0,
      aiOnlyRate: tasks.length ? tasks.filter(t=>t.completionMode==="ai-only").length/tasks.length : 0,
      repairRate: tasks.length ? tasks.filter(t=>t.completionMode==="repair").length/tasks.length : 0,
      fallbackRate: tasks.length ? tasks.filter(t=>t.completionMode==="fallback").length/tasks.length : 0,
      meanDuration: mean(tasks.map(t => t.durationMs || 0)),
      meanRetry: mean(tasks.map(t => t.retryCount || 0)),
      inference: mean(inferences),
    };
  });
}

function renderSurvey() {
  const items = new Map();
  for (const survey of surveys) {
    for (const item of survey.scores || []) {
      const key = Number(item.item);
      if (!items.has(key)) items.set(key, {question:item.question || `문항 ${key}`, scores:[]});
      items.get(key).scores.push(Number(item.score));
    }
  }

  surveyRows.innerHTML = [...items.entries()].sort((a,b)=>a[0]-b[0]).map(([item, data]) => {
    const xs=data.scores.filter(Number.isFinite);
    const range=xs.length ? `${Math.min(...xs)}–${Math.max(...xs)}` : "-";
    return `<tr><td>${item}. ${esc(data.question)}</td><td>${xs.length}</td><td>${mean(xs).toFixed(2)}</td><td>${median(xs).toFixed(2)}</td><td>${range}</td></tr>`;
  }).join("");

  const comments = surveys.flatMap(s => {
    const p=esc(s.participant || "unknown");
    const c=s.comments || {};
    return [
      c.difficult ? `<div><b>${p} · 불편</b> — ${esc(c.difficult)}</div>` : "",
      c.useful ? `<div><b>${p} · 유용</b> — ${esc(c.useful)}</div>` : "",
      c.improve ? `<div><b>${p} · 개선</b> — ${esc(c.improve)}</div>` : "",
    ].filter(Boolean);
  });
  surveyComments.innerHTML = comments.length ? comments.join("") : "불러온 자유응답이 없습니다.";
}

function render() {
  const tasks = allTasks();
  const participants = new Set([
    ...logs.map(x=>x.participant||"unknown"),
    ...surveys.map(x=>x.participant||"unknown"),
  ]);
  const success = tasks.filter(t=>t.success).length;
  const durations = tasks.map(t=>Number(t.durationMs||0));
  const surveyMeans = surveys.map(s=>Number(s.meanScore)).filter(Number.isFinite);

  const summary = {
    participants: participants.size,
    tasks: tasks.length,
    successRate: tasks.length ? success/tasks.length : 0,
    aiOnlyRate: tasks.length ? tasks.filter(t=>t.completionMode==="ai-only").length/tasks.length : 0,
    repairSuccessRate: tasks.length ? tasks.filter(t=>t.completionMode==="repair").length/tasks.length : 0,
    fallbackSuccessRate: tasks.length ? tasks.filter(t=>t.completionMode==="fallback").length/tasks.length : 0,
    meanDuration: mean(durations),
    medianDuration: median(durations),
    meanRetry: mean(tasks.map(t=>Number(t.retryCount||0))),
    surveyMean: mean(surveyMeans),
  };

  metricsEl.innerHTML = [
    ["참여자", summary.participants],
    ["완료 과제", summary.tasks],
    ["기능적 성공률", pct(summary.successRate)],
    ["AI 단독 성공률", pct(summary.aiOnlyRate)],
    ["후보 repair 성공", pct(summary.repairSuccessRate)],
    ["직접입력 성공", pct(summary.fallbackSuccessRate)],
    ["평균 수행시간", `${sec(summary.meanDuration)}초`],
    ["중앙 수행시간", `${sec(summary.medianDuration)}초`],
    ["평균 재시도", summary.meanRetry.toFixed(2)],
    ["사용성 평균", surveyMeans.length ? `${summary.surveyMean.toFixed(2)}/5` : "-"],
  ].map(([k,v])=>`<div class="metric"><span class="muted">${k}</span><b>${v}</b></div>`).join("");

  participantRows.innerHTML = participantStats().map(p=>`
    <tr>
      <td>${esc(p.participant)}</td><td>${p.tasks}</td><td>${pct(p.successRate)}</td>
      <td>${pct(p.aiOnlyRate)}</td><td>${pct(p.repairRate)}</td><td>${pct(p.fallbackRate)}</td>
      <td>${sec(p.meanDuration)}</td><td>${p.meanRetry.toFixed(2)}</td><td>${ms1(p.inference)}</td>
    </tr>`).join("");

  taskRows.innerHTML = tasks.map(t=>`
    <tr>
      <td>${esc(t.participant)}</td><td>${esc(t.id)}</td><td>${t.success?"성공":"실패"}</td>
      <td>${modeKo(t.completionMode)}</td><td>${sec(t.durationMs||0)}</td><td>${t.finalPredictions||0}</td><td>${t.retryCount||0}</td>
      <td>${t.candidateRepairs||0}</td><td>${t.manualFallbackUsed?"예":"아니오"}</td>
      <td>${ms1(t.meanInferenceLatencyMs)}</td>
    </tr>`).join("");

  renderSurvey();
  statusEl.textContent = ` ${logs.length}개 로그 · ${surveys.length}개 설문 · ${tasks.length}개 과제`;
}

filesInput.addEventListener("change", async e => {
  const nextLogs = [];
  const nextSurveys = [];
  for (const file of e.target.files || []) {
    try {
      const data = JSON.parse(await file.text());
      if (data.format === "ieum-pilot-log-v1" || data.format === "ieum-pilot-log-v2") nextLogs.push(data);
      else if (data.format === "ieum-pilot-survey-v1") nextSurveys.push(data);
      else throw new Error("지원하지 않는 JSON 형식");
    } catch (err) {
      console.error(file.name, err);
      alert(`${file.name}: ${err.message}`);
    }
  }
  logs = nextLogs;
  surveys = nextSurveys;
  render();
});

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replaceAll('"','""')}"` : s;
}

function downloadText(text, filename, type) {
  const blob = new Blob([text], {type});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

exportCsvBtn.addEventListener("click", () => {
  const rows = [[
    "record_type","participant","task_id_or_item","success_or_score","completion_mode","duration_ms",
    "final_predictions","retry_count","candidate_repairs","manual_fallback",
    "mean_inference_latency_ms","question"
  ]];

  for (const log of logs) {
    for (const raw of log.tasks || []) {
      const t={...raw,completionMode:raw.completionMode||modeOf(raw)};
      rows.push([
        "task",log.participant,t.id,t.success,t.completionMode,t.durationMs,t.finalPredictions,t.retryCount,
        t.candidateRepairs,t.manualFallbackUsed,t.meanInferenceLatencyMs,""
      ]);
    }
  }

  for (const survey of surveys) {
    for (const item of survey.scores || []) {
      rows.push([
        "survey",survey.participant,item.item,item.score,"","","","","","","",item.question
      ]);
    }
  }

  const text = rows.map(r=>r.map(csvCell).join(",")).join("\n");
  downloadText(text, `ieum_pilot_combined_${Date.now()}.csv`, "text/csv;charset=utf-8");
});

render();

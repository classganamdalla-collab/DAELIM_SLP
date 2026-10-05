const filesInput = document.getElementById("files");
const statusEl = document.getElementById("status");
const metricsEl = document.getElementById("metrics");
const participantRows = document.getElementById("participantRows");
const taskRows = document.getElementById("taskRows");
const exportCsvBtn = document.getElementById("exportCsv");

let logs = [];

const mean = xs => xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0;
const pct = x => `${(100*x).toFixed(1)}%`;
const sec = ms => (ms/1000).toFixed(2);
const ms1 = n => Number(n||0).toFixed(2);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  })[ch]);
}

function participantStats() {
  const map = new Map();
  for (const log of logs) {
    const p = log.participant || "unknown";
    if (!map.has(p)) map.set(p, []);
    map.get(p).push(log);
  }
  return [...map.entries()].map(([participant, items]) => {
    const tasks = items.flatMap(x => x.tasks || []);
    const inferences = items.map(x => x.summary?.meanInferenceLatencyMs || 0).filter(Boolean);
    const success = tasks.filter(t => t.success).length;
    return {
      participant,
      tasks: tasks.length,
      successRate: tasks.length ? success/tasks.length : 0,
      meanDuration: mean(tasks.map(t => t.durationMs || 0)),
      meanRetry: mean(tasks.map(t => t.retryCount || 0)),
      manual: tasks.filter(t => t.manualFallbackUsed).length,
      repairs: tasks.reduce((a,t)=>a+(t.candidateRepairs||0),0),
      inference: mean(inferences),
    };
  });
}

function render() {
  const tasks = logs.flatMap(log => (log.tasks || []).map(t => ({...t, participant:log.participant||"unknown"})));
  const participants = new Set(logs.map(x=>x.participant||"unknown"));
  const success = tasks.filter(t=>t.success).length;
  const summary = {
    participants: participants.size,
    tasks: tasks.length,
    successRate: tasks.length ? success/tasks.length : 0,
    meanDuration: mean(tasks.map(t=>t.durationMs||0)),
    meanRetry: mean(tasks.map(t=>t.retryCount||0)),
    manualRate: tasks.length ? tasks.filter(t=>t.manualFallbackUsed).length/tasks.length : 0,
    repairs: tasks.reduce((a,t)=>a+(t.candidateRepairs||0),0),
  };

  metricsEl.innerHTML = [
    ["참여자", summary.participants],
    ["완료 과제", summary.tasks],
    ["과제 성공률", pct(summary.successRate)],
    ["평균 수행시간", `${sec(summary.meanDuration)}초`],
    ["평균 재시도", summary.meanRetry.toFixed(2)],
    ["수동입력 사용률", pct(summary.manualRate)],
    ["후보 선택 횟수", summary.repairs],
  ].map(([k,v])=>`<div class="metric"><span class="muted">${k}</span><b>${v}</b></div>`).join("");

  participantRows.innerHTML = participantStats().map(p=>`
    <tr>
      <td>${esc(p.participant)}</td><td>${p.tasks}</td><td>${pct(p.successRate)}</td>
      <td>${sec(p.meanDuration)}</td><td>${p.meanRetry.toFixed(2)}</td><td>${p.manual}</td>
      <td>${p.repairs}</td><td>${ms1(p.inference)}</td>
    </tr>`).join("");

  taskRows.innerHTML = tasks.map(t=>`
    <tr>
      <td>${esc(t.participant)}</td><td>${esc(t.id)}</td><td>${t.success?"성공":"실패"}</td>
      <td>${sec(t.durationMs||0)}</td><td>${t.finalPredictions||0}</td><td>${t.retryCount||0}</td>
      <td>${t.candidateRepairs||0}</td><td>${t.manualFallbackUsed?"예":"아니오"}</td>
      <td>${ms1(t.meanInferenceLatencyMs)}</td>
    </tr>`).join("");

  statusEl.textContent = ` ${logs.length}개 로그 · ${tasks.length}개 과제`;
}

filesInput.addEventListener("change", async e => {
  const next = [];
  for (const file of e.target.files || []) {
    try {
      const data = JSON.parse(await file.text());
      if (data.format !== "ieum-pilot-log-v1") throw new Error("지원하지 않는 로그 형식");
      next.push(data);
    } catch (err) {
      console.error(file.name, err);
      alert(`${file.name}: ${err.message}`);
    }
  }
  logs = next;
  render();
});

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replaceAll('"','""')}"` : s;
}

exportCsvBtn.addEventListener("click", () => {
  const rows = [["participant","task_id","success","duration_ms","final_predictions","retry_count","candidate_repairs","manual_fallback","mean_inference_latency_ms"]];
  for (const log of logs) {
    for (const t of log.tasks || []) {
      rows.push([
        log.participant,t.id,t.success,t.durationMs,t.finalPredictions,t.retryCount,
        t.candidateRepairs,t.manualFallbackUsed,t.meanInferenceLatencyMs
      ]);
    }
  }
  const text = rows.map(r=>r.map(csvCell).join(",")).join("\n");
  const blob = new Blob([text], {type:"text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ieum_pilot_tasks_${Date.now()}.csv`;
  a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});

render();

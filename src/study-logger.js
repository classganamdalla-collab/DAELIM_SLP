function finite(n) {
  return Number.isFinite(Number(n)) ? Number(n) : 0;
}

function mean(xs) {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

export class StudyLogger {
  constructor({ enabled = false, participant = "P00" } = {}) {
    this.enabled = Boolean(enabled);
    this.participant = String(participant || "P00").replace(/[^A-Za-z0-9_-]/g, "_");
    this.sessionId = crypto.randomUUID();
    this.startedAt = new Date().toISOString();
    this.events = [];
    this.tasks = [];
    this.currentTask = null;
    this.inferenceLatencies = [];
    this.taskCounter = 0;
  }

  log(type, payload = {}) {
    if (!this.enabled) return;
    this.events.push({
      t: performance.now(),
      iso: new Date().toISOString(),
      type,
      ...payload,
    });
  }

  startTask(taskId = null) {
    if (!this.enabled) return null;
    if (this.currentTask) this.finishTask(false, "interrupted");

    this.taskCounter += 1;
    const id = taskId || `T${String(this.taskCounter).padStart(2, "0")}`;
    this.currentTask = {
      id,
      startedPerf: performance.now(),
      startedAt: new Date().toISOString(),
      finalPredictions: 0,
      rejectedPredictions: 0,
      acceptedPredictions: 0,
      candidateRepairs: 0,
      manualFallbackUsed: false,
      sentencesSpoken: 0,
      inferenceLatencies: [],
    };
    this.log("task_start", { taskId: id });
    return id;
  }

  recordInferenceLatency(ms) {
    if (!this.enabled) return;
    const value = finite(ms);
    this.inferenceLatencies.push(value);
    if (this.currentTask) this.currentTask.inferenceLatencies.push(value);
  }

  recordFinalPrediction(result, accepted) {
    if (!this.enabled) return;
    const payload = {
      taskId: this.currentTask?.id ?? null,
      label: result?.label ?? null,
      score: finite(result?.score),
      margin: finite(result?.margin),
      accepted: Boolean(accepted),
    };
    this.log("final_prediction", payload);

    if (this.currentTask) {
      this.currentTask.finalPredictions += 1;
      if (accepted) this.currentTask.acceptedPredictions += 1;
      else this.currentTask.rejectedPredictions += 1;
    }
  }

  recordCandidateRepair(label) {
    if (!this.enabled) return;
    this.log("candidate_repair", { taskId: this.currentTask?.id ?? null, label });
    if (this.currentTask) this.currentTask.candidateRepairs += 1;
  }

  recordManualFallback() {
    if (!this.enabled) return;
    this.log("manual_fallback", { taskId: this.currentTask?.id ?? null });
    if (this.currentTask) this.currentTask.manualFallbackUsed = true;
  }

  recordSentenceSpoken(source, textLength = 0) {
    if (!this.enabled) return;
    this.log("sentence_spoken", {
      taskId: this.currentTask?.id ?? null,
      source,
      textLength: finite(textLength),
    });
    if (this.currentTask) this.currentTask.sentencesSpoken += 1;
  }

  finishTask(success, note = "") {
    if (!this.enabled || !this.currentTask) return null;
    const finished = performance.now();
    const task = {
      id: this.currentTask.id,
      startedAt: this.currentTask.startedAt,
      finishedAt: new Date().toISOString(),
      success: Boolean(success),
      note: String(note || ""),
      durationMs: finished - this.currentTask.startedPerf,
      finalPredictions: this.currentTask.finalPredictions,
      acceptedPredictions: this.currentTask.acceptedPredictions,
      rejectedPredictions: this.currentTask.rejectedPredictions,
      retryCount: Math.max(0, this.currentTask.finalPredictions - 1),
      candidateRepairs: this.currentTask.candidateRepairs,
      manualFallbackUsed: this.currentTask.manualFallbackUsed,
      sentencesSpoken: this.currentTask.sentencesSpoken,
      meanInferenceLatencyMs: mean(this.currentTask.inferenceLatencies),
    };
    this.tasks.push(task);
    this.log("task_finish", {
      taskId: task.id,
      success: task.success,
      durationMs: task.durationMs,
    });
    this.currentTask = null;
    return task;
  }

  summary() {
    const completed = this.tasks.length;
    const success = this.tasks.filter(t => t.success).length;
    const finalEvents = this.events.filter(e => e.type === "final_prediction");
    const acceptedEvents = finalEvents.filter(e => e.accepted);

    return {
      participant: this.participant,
      completedTasks: completed,
      successfulTasks: success,
      taskSuccessRate: completed ? success / completed : 0,
      meanTaskDurationMs: mean(this.tasks.map(t => t.durationMs)),
      meanRetryCount: mean(this.tasks.map(t => t.retryCount)),
      manualFallbackTasks: this.tasks.filter(t => t.manualFallbackUsed).length,
      candidateRepairCount: this.tasks.reduce((a, t) => a + t.candidateRepairs, 0),
      finalPredictionCount: finalEvents.length,
      acceptedPredictionRate: finalEvents.length ? acceptedEvents.length / finalEvents.length : 0,
      meanInferenceLatencyMs: mean(this.inferenceLatencies),
    };
  }

  payload() {
    return {
      format: "ieum-pilot-log-v1",
      participant: this.participant,
      sessionId: this.sessionId,
      startedAt: this.startedAt,
      exportedAt: new Date().toISOString(),
      summary: this.summary(),
      tasks: this.tasks,
      events: this.events,
      privacy: "No camera frames, raw landmarks, audio, or STT transcripts are stored in this log.",
    };
  }

  download() {
    if (!this.enabled) return;
    const data = JSON.stringify(this.payload(), null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ieum_pilot_${this.participant}_${Date.now()}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

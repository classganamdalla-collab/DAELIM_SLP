#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expand(patterns):
    out = set()
    for pattern in patterns:
        for name in glob.glob(pattern, recursive=True):
            p = Path(name)
            if p.is_file() and p.suffix.lower() == ".json":
                out.add(p.resolve())
    return sorted(out)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs):
    return statistics.fmean(xs) if xs else None


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def nfmt(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


def task_mode(task):
    if not task.get("success"):
        return "failed"
    if task.get("manualFallbackUsed"):
        return "fallback"
    if int(task.get("candidateRepairs") or 0) > 0:
        return "repair"
    return "ai-only"


def collect_pilot(paths):
    logs, surveys = [], []
    for path in paths:
        data = load(path)
        fmt = data.get("format")
        if fmt in {"ieum-pilot-log-v1", "ieum-pilot-log-v2"}:
            logs.append(data)
        elif fmt == "ieum-pilot-survey-v1":
            surveys.append(data)

    if not logs and not surveys:
        return None

    tasks = []
    for log in logs:
        for raw in log.get("tasks") or []:
            t = dict(raw)
            t["participant"] = log.get("participant") or "unknown"
            t["completionMode"] = t.get("completionMode") or task_mode(t)
            tasks.append(t)

    participants = set()
    participants.update(str(x.get("participant") or "unknown") for x in logs)
    participants.update(str(x.get("participant") or "unknown") for x in surveys)
    survey_scores = [
        float(item["score"])
        for survey in surveys
        for item in (survey.get("scores") or [])
        if item.get("score") is not None
    ]
    total = len(tasks)
    return {
        "participants": len(participants),
        "tasks": total,
        "functional_success_rate": sum(bool(t.get("success")) for t in tasks) / total if total else None,
        "ai_only_success_rate": sum(t["completionMode"] == "ai-only" for t in tasks) / total if total else None,
        "repair_success_rate": sum(t["completionMode"] == "repair" for t in tasks) / total if total else None,
        "fallback_success_rate": sum(t["completionMode"] == "fallback" for t in tasks) / total if total else None,
        "mean_duration_ms": mean([float(t.get("durationMs") or 0) for t in tasks]),
        "median_duration_ms": statistics.median([float(t.get("durationMs") or 0) for t in tasks]) if tasks else None,
        "mean_retry_count": mean([float(t.get("retryCount") or 0) for t in tasks]),
        "mean_inference_latency_ms": mean([
            float(t.get("meanInferenceLatencyMs") or 0)
            for t in tasks if float(t.get("meanInferenceLatencyMs") or 0) > 0
        ]),
        "survey_mean_5": mean(survey_scores),
        "survey_item_responses": len(survey_scores),
        "log_files": len(logs),
        "survey_files": len(surveys),
    }


def collect_benchmarks(paths):
    out = []
    for path in paths:
        data = load(path)
        if data.get("format") == "ieum-browser-benchmark-v1":
            out.append(data)
    return out


def collect_model(model_dir):
    metrics = model_dir / "metrics.json"
    if not metrics.exists():
        return None
    result = {"metrics": load(metrics)}
    onnx = model_dir / "onnx_evaluation.json"
    result["onnx"] = load(onnx) if onnx.exists() else None
    return result


def render(model, pilot, benchmarks):
    lines = [
        "# 이음 공모전 실측 근거 요약",
        "",
        "> 실제 생성된 결과 파일만 집계하며 없는 결과는 임의로 채우지 않습니다.",
        "",
        "## 1. v2 모델 평가",
        "",
    ]
    if model is None:
        lines += [
            "- 아직 model/v2/metrics.json이 없습니다.",
            "- 다인원 원시 데이터 수집 및 v2 학습 후 수치가 생성됩니다.",
        ]
    else:
        m = model["metrics"]
        t = m.get("test") or {}
        cis = t.get("confidence_intervals") or {}
        aci = cis.get("accuracy_95ci") or {}
        fci = cis.get("macro_f1_95ci") or {}
        rej = t.get("rejection") or {}
        lines += [
            f"- 선택 모델: {m.get('selected_model', '-')}",
            f"- split 방식: {m.get('split_strategy', '-')}",
            f"- 데이터 표본: {m.get('dataset_samples', '-')}",
            f"- held-out test accuracy: {pct(t.get('accuracy'))}",
            f"- held-out test macro-F1: {nfmt(t.get('macro_f1'))}",
            f"- accuracy 95% CI: {pct(aci.get('low'))} ~ {pct(aci.get('high'))}",
            f"- macro-F1 95% CI: {nfmt(fci.get('low'))} ~ {nfmt(fci.get('high'))}",
            f"- rejection known coverage: {pct(rej.get('known_coverage'))}",
            f"- rejection known selective accuracy: {pct(rej.get('selective_accuracy_known'))}",
            f"- unknown false-accept rate: {pct(rej.get('unknown_false_accept_rate'))}",
        ]
        if model.get("onnx"):
            q = model["onnx"]
            d = q.get("int8_vs_fp32") or {}
            lines += [
                "",
                "### ONNX FP32 대 INT8",
                f"- FP32 test accuracy: {pct((q.get('fp32') or {}).get('accuracy'))}",
                f"- INT8 test accuracy: {pct((q.get('int8') or {}).get('accuracy'))}",
                f"- INT8 macro-F1 감소: {nfmt(d.get('macro_f1_drop'))}",
                f"- argmax agreement: {pct(d.get('argmax_agreement'))}",
                f"- 모델 크기 감소: {nfmt(d.get('size_reduction_percent'), 1)}%",
                f"- INT8 accuracy gate: {'통과' if q.get('int8_accuracy_gate_pass') else '미통과'}",
            ]

    lines += ["", "## 2. 실제 파일럿", ""]
    if pilot is None:
        lines += [
            "- 아직 실제 파일럿 로그/설문 파일이 없습니다.",
            "- 실제 참여 데이터 없이 성공률을 생성하지 않습니다.",
        ]
    else:
        lines += [
            f"- 참여자: {pilot['participants']}명",
            f"- 완료 과제: {pilot['tasks']}개",
            f"- 기능적 성공률: {pct(pilot['functional_success_rate'])}",
            f"- AI 단독 성공률: {pct(pilot['ai_only_success_rate'])}",
            f"- 후보 repair 성공률: {pct(pilot['repair_success_rate'])}",
            f"- 직접입력 fallback 성공률: {pct(pilot['fallback_success_rate'])}",
            f"- 평균 수행시간: {nfmt((pilot['mean_duration_ms'] or 0) / 1000, 2)}초",
            f"- 중앙 수행시간: {nfmt((pilot['median_duration_ms'] or 0) / 1000, 2)}초",
            f"- 평균 재시도: {nfmt(pilot['mean_retry_count'], 2)}회",
            f"- 평균 추론 latency: {nfmt(pilot['mean_inference_latency_ms'], 2)} ms",
            f"- 사용성 평균: {nfmt(pilot['survey_mean_5'], 2)}/5",
        ]

    lines += ["", "## 3. 실제 브라우저 benchmark", ""]
    if not benchmarks:
        lines += [
            "- 아직 브라우저 benchmark 결과 JSON이 없습니다.",
            "- 실제 기기에서 benchmark.html을 실행하고 결과 JSON을 저장해야 합니다.",
        ]
    else:
        for i, data in enumerate(benchmarks, 1):
            device = data.get("device") or {}
            lines += [
                f"### 기기 {i}",
                f"- platform: {device.get('platform', 'unknown')} / mobile={device.get('mobile')}",
                f"- logical cores: {device.get('hardwareConcurrency')}",
                f"- WebGPU: {device.get('webgpuAvailable')}",
            ]
            for eng in data.get("engines") or []:
                lines.append(
                    f"- {eng.get('name')}: mean {nfmt(eng.get('latencyMeanMs'), 2)} ms, "
                    f"p95 {nfmt(eng.get('latencyP95Ms'), 2)} ms, load {nfmt(eng.get('loadMs'), 1)} ms"
                )
            lines.append("")

    lines += [
        "## 4. 해석 원칙",
        "",
        "- v1 98.91%는 과거 내부 validation 수치이며 실제 사용자 성능이 아닙니다.",
        "- v2 수치는 실제 수집 데이터로 학습/평가한 뒤에만 사용합니다.",
        "- 참여자가 실제 한국수어 사용자가 아니면 기술적 사용성 파일럿으로 한정합니다.",
        "- INT8은 정확도, 모델 크기, 실제 브라우저 latency를 함께 보고 채택합니다.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model/v2")
    ap.add_argument("--pilot", nargs="*", default=["reports/input/pilot/*.json"])
    ap.add_argument("--benchmark", nargs="*", default=["reports/input/benchmark/*.json"])
    ap.add_argument("--out-dir", default="reports/local")
    args = ap.parse_args()

    pilot_paths = expand(args.pilot)
    benchmark_paths = expand(args.benchmark)
    model = collect_model(ROOT / args.model_dir)
    pilot = collect_pilot(pilot_paths)
    benchmarks = collect_benchmarks(benchmark_paths)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "pilot": pilot,
        "browser_benchmarks": benchmarks,
        "sources": {
            "pilot": [str(x) for x in pilot_paths],
            "benchmark": [str(x) for x in benchmark_paths],
        },
    }
    (out / "evidence-summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "evidence-summary.md").write_text(render(model, pilot, benchmarks), encoding="utf-8")
    print(out / "evidence-summary.md")
    print(out / "evidence-summary.json")


if __name__ == "__main__":
    main()

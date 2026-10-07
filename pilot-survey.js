const questions = [
  "사용 방법을 이해하기 쉬웠다.",
  "수어 인식 결과를 확인하기 쉬웠다.",
  "인식이 틀렸을 때 수정하거나 다른 방법으로 전달하기 쉬웠다.",
  "상대방과 대화할 때 도움이 될 가능성이 있다고 느꼈다.",
  "실제 카페 같은 상황에서 사용해볼 의향이 있다.",
];

const form = document.getElementById("survey");
questions.forEach((q, qi) => {
  const wrap = document.createElement("div");
  wrap.className = "q";
  const title = document.createElement("div");
  title.className = "q-title";
  title.textContent = `${qi + 1}. ${q}`;
  const scale = document.createElement("div");
  scale.className = "scale";

  for (let score = 1; score <= 5; score++) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="radio" name="q${qi + 1}" value="${score}" required><small>${score}점</small>`;
    scale.appendChild(label);
  }
  wrap.append(title, scale);
  form.appendChild(wrap);
});

document.getElementById("save").addEventListener("click", () => {
  const participant = document.getElementById("participant").value.trim().replace(/[^A-Za-z0-9_-]/g, "_");
  if (!participant) return alert("가명 참여자 코드를 입력해주세요.");
  if (!form.reportValidity()) return;

  const scores = questions.map((question, i) => ({
    item: i + 1,
    question,
    score: Number(new FormData(form).get(`q${i + 1}`)),
  }));

  const payload = {
    format: "ieum-pilot-survey-v1",
    participant,
    createdAt: new Date().toISOString(),
    scale: { min: 1, max: 5 },
    scores,
    meanScore: scores.reduce((a, x) => a + x.score, 0) / scores.length,
    comments: {
      difficult: document.getElementById("difficult").value.trim(),
      useful: document.getElementById("useful").value.trim(),
      improve: document.getElementById("improve").value.trim(),
    },
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ieum_survey_${participant}_${Date.now()}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

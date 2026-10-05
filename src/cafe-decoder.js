function unique(words) {
  return [...new Set(words.filter(Boolean))];
}

export function decodeCafeWords(inputWords) {
  const words = unique(inputWords);
  if (!words.length) return "";

  const has = w => words.includes(w);
  const clauses = [];

  if (has("안녕하세요")) clauses.push("안녕하세요.");

  const orderTokens = ["아메리카노", "아이스", "뜨거운", "2잔", "제일", "큰 걸로", "테이크아웃", "주세요", "해주세요"];
  if (orderTokens.some(has)) {
    const item = [];
    if (has("아이스")) item.push("아이스");
    else if (has("뜨거운")) item.push("따뜻한");

    if (has("아메리카노")) item.push("아메리카노");
    if (has("2잔")) item.push("두 잔");

    const modifiers = [];
    if (has("제일") && has("큰 걸로")) modifiers.push("제일 큰 사이즈로");
    else if (has("큰 걸로")) modifiers.push("큰 사이즈로");
    if (has("테이크아웃")) modifiers.push("테이크아웃으로");

    if (item.length || modifiers.length) {
      const head = item.length ? item.join(" ") : "주문";
      const tail = modifiers.length ? ` ${modifiers.join(" ")}` : "";
      const ending = has("해주세요") || has("테이크아웃") ? " 해주세요." : " 주세요.";
      clauses.push(`${head}${tail}${ending}`.replace(/\s+/g, " ").trim());
    } else if (has("주세요") || has("해주세요")) {
      clauses.push(has("해주세요") ? "해주세요." : "주세요.");
    }
  }

  if (has("와이파이") || has("있나요?")) {
    clauses.push("와이파이 있나요?");
  }

  if (has("포인트")) clauses.push("포인트 할게요.");
  if (has("카드")) clauses.push("카드로 결제할게요.");
  if (has("영수증")) clauses.push("영수증 주세요.");
  if (has("감사합니다")) clauses.push("감사합니다.");

  if (!clauses.length) return words.join(" ");
  return clauses.join(" ").replace(/\s+/g, " ").trim();
}

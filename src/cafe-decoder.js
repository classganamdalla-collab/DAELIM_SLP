function unique(words) {
  return [...new Set(words.filter(Boolean).map(String))];
}

const IDS = Object.freeze({
  HELLO: "안녕하세요",
  AMERICANO: "아메리카노",
  HOT: "뜨거운",
  ICE: "아이스",
  TWO: "2잔",
  PLEASE: "주세요",
  MOST: "제일",
  LARGE: "큰걸로",
  DO_PLEASE: "해주세요",
  TAKEOUT: "테이크아웃",
  POINT: "포인트",
  CARD: "카드",
  RECEIPT: "영수증",
  THANKS: "감사합니다",
  WIFI: "와이파이",
  EXISTS: "있나요",
});

/**
 * Deterministic cafe-domain surface realization.
 *
 * Important: inputWords are model label IDs, not display strings.
 * This function intentionally does not invent content that is absent from
 * the recognized labels. It only reorders or inflects known cafe tokens.
 */
export function decodeCafeWords(inputWords) {
  const words = unique(inputWords);
  if (!words.length) return "";

  const has = id => words.includes(id);
  const clauses = [];

  if (has(IDS.HELLO)) clauses.push("안녕하세요.");

  const orderTokens = [
    IDS.AMERICANO, IDS.ICE, IDS.HOT, IDS.TWO, IDS.MOST,
    IDS.LARGE, IDS.TAKEOUT, IDS.PLEASE, IDS.DO_PLEASE,
  ];

  if (orderTokens.some(has)) {
    const item = [];
    if (has(IDS.ICE)) item.push("아이스");
    else if (has(IDS.HOT)) item.push("따뜻한");

    if (has(IDS.AMERICANO)) item.push("아메리카노");
    if (has(IDS.TWO)) item.push("두 잔");

    const modifiers = [];
    if (has(IDS.MOST) && has(IDS.LARGE)) modifiers.push("제일 큰 사이즈로");
    else if (has(IDS.LARGE)) modifiers.push("큰 사이즈로");
    if (has(IDS.TAKEOUT)) modifiers.push("테이크아웃으로");

    if (item.length || modifiers.length) {
      const head = item.length ? item.join(" ") : "주문";
      const tail = modifiers.length ? ` ${modifiers.join(" ")}` : "";
      const ending = has(IDS.DO_PLEASE) || has(IDS.TAKEOUT) ? " 해주세요." : " 주세요.";
      clauses.push(`${head}${tail}${ending}`.replace(/\s+/g, " ").trim());
    } else if (has(IDS.PLEASE) || has(IDS.DO_PLEASE)) {
      clauses.push(has(IDS.DO_PLEASE) ? "해주세요." : "주세요.");
    }
  }

  if (has(IDS.WIFI) || has(IDS.EXISTS)) {
    clauses.push("와이파이 있나요?");
  }

  if (has(IDS.POINT)) clauses.push("포인트 할게요.");
  if (has(IDS.CARD)) clauses.push("카드로 결제할게요.");
  if (has(IDS.RECEIPT)) clauses.push("영수증 주세요.");
  if (has(IDS.THANKS)) clauses.push("감사합니다.");

  if (!clauses.length) return words.join(" ");
  return clauses.join(" ").replace(/\s+/g, " ").trim();
}
